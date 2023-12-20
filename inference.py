# %% [markdown]
# # Inference

# %%
import os
import json
import time
from argparse import ArgumentParser
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
from dataclasses import dataclass
from loguru import logger

import evaluate
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, DataCollatorForSeq2Seq

from src.dataset import get_dataset
# %%

parser = ArgumentParser("LLM inference")
# testing arguments
parser.add_argument('-m', "--model", type=str, required=True)
parser.add_argument('-t', "--tokenizer", type=str, required=True)
parser.add_argument('-d', "--data", type=str, default="samsum")
parser.add_argument('--metric', type=str, default='rouge',  choices=['rouge', 'accuracy'])
parser.add_argument("--seed", type=int, default=2023)
# generation arguments
parser.add_argument("--max-tokens", type=int, default=256)
parser.add_argument("--top-k", type=int, default=-1)
parser.add_argument("--top-p", type=float, default=1.0)
parser.add_argument("--temperature", type=float, default=1.0)

args = parser.parse_args()

SEED = args.seed
evaluation_metric = args.metric

__import__('random').seed(SEED)
__import__('numpy').random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# %%
MODEL_PATH = Path(args.model)  # Path("./results/samsum/gpt2-base-sft/checkpoint-8736/")
TOKENIZER_PATH = Path(args.tokenizer)  # Path("../models/gpt2/base")
WORK_DIR = Path(args.model)  # Path('results/samsum/gpt2-base-sft/checkpoint-8736/')

DATA_NAME = args.data

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
tokenizer.padding_side = "left"
tokenizer.pad_token_id = tokenizer.eos_token_id
logger.debug(f"The padding token id is {tokenizer.pad_token_id}")
    
_data_class = get_dataset(DATA_NAME)
LABEL_SPLIT = _data_class.info.label_split
label_column = _data_class.info.label_column
test_data = _data_class.get_test(tokenizer)

logger.debug(f"Test data: {test_data}")
# __import__('ipdb').set_trace()
# %%
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    load_in_8bit=False, 
    torch_dtype=torch.float16
)
import tensor_parallel as tp
model = tp.tensor_parallel(model)
model.cuda()

# %%
label_pad_token_id = -100

@dataclass
class idCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        # print(features)
        _id = [feature.pop('_id') for feature in features] if "_id" in features[0].keys() else None
        return _id, super().__call__(features, return_tensors)

data_collator = idCollator(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    return_tensors="pt",
    pad_to_multiple_of=8
)

# %%
dataloader = DataLoader(test_data, collate_fn=data_collator, batch_size=4)

# %%
generation_config = GenerationConfig(
    # do_sample=True,
    # top_p=1.5,
    # top_k= 3,
    # temperature=1.0,
    # no_repeat_ngram_size=6,
    # repetition_penalty=None,
    max_length=1024,
    max_new_token=512,
    min_length=None,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    return_dict_in_generate=True,
    output_scores=False
)

# %%
results = []
for it, (_ids, data) in tqdm(enumerate(dataloader), total=len(dataloader)):
    for k, v in data.items():
        data[k] = v.to(model.device)
    out = model.generate(**data, generation_config=generation_config).sequences
    # print(out)
    results.extend([{"id":_id, "summary": summary} for _id, summary in zip(_ids, tokenizer.batch_decode(out))])

# %%
predictions = []
for result in tqdm(results):
    if LABEL_SPLIT in result['summary']:
        predictions.append({
            'id': result['id'],
            'summary': result['summary'].replace(tokenizer.pad_token,'').split(LABEL_SPLIT)[-1]
        })

# %%
suffix = time.time_ns()
with open(WORK_DIR/f"predictions-{suffix}.json", 'w') as f:
    json.dump(predictions, f)

# %%
dataset = _data_class.dataset['test']
pred_ref = pd.merge(
    pd.DataFrame.from_records(predictions),
    pd.DataFrame.from_records(dataset, columns=['id', label_column]),
    on='id',
    how='inner',
    suffixes=['_pred', '_ref']
)

# %%
if evaluation_metric == 'rouge':
    rouge = evaluate.load("src/metrics/rouge")
    metrics = rouge.compute(predictions=pred_ref[label_column+'_pred'], 
                            references=pred_ref[label_column+'_ref'])
elif evaluation_metric == 'accuracy':
    acc = evaluate.load("src/metrics/accuracy")
    def ans_parse(ans_str: str) -> int:
        try:
            ans = int(ans_str.split("####")[-1].strip().replace(',',''))
        except Exception:
            ans = -9999
        return ans
                
    metrics = acc.compute(predictions=pred_ref[label_column+'_pred'].map(ans_parse), 
                            references=pred_ref[label_column+'_ref'].map(ans_parse))

logger.info(metrics)

# %%
with open(WORK_DIR/f"metrics-{suffix}.json", 'w') as f:
    json.dump(metrics, f)


