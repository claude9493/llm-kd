# %% [markdown]
# # Inference using vllm
# CUDA_VISIBLE_DEVICES=7 python inference-vllm.py

import json
# %%
import os
import time
from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer

import evaluate
import pandas as pd
import torch
import vllm
from datasets import load_from_disk
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformers import DataCollatorForSeq2Seq

# %%
SEED = 2023

__import__('random').seed(SEED)
__import__('numpy').random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# %%
DATA_PATH = Path("./data/samsum")
MODEL_PATH = Path("./results/samsum/gpt2-xlarge-sft/checkpoint-4600/")
TOKENIZER_PATH = Path("../models/gpt2/xlarge")
WORK_DIR = Path('results/samsum/gpt2-xlarge-sft')

# %%
dataset = load_from_disk(str(DATA_PATH))
logger.debug(dataset)

prompt_template = """[INST] <<SYS>>
Use the Input to provide a summary of a conversation.
<</SYS>>

Input:
{dialogue}

Summary:
{summary}
"""
LABEL_SPLIT = "Summary:\n"
logger.debug("Train data example:\n" + prompt_template.format(**dataset['train'][0]))

# %%
# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, load_in_8bit=False)
model = vllm.LLM(model=str(MODEL_PATH), 
                 tokenizer=str(TOKENIZER_PATH),
                 seed=SEED)

# %%
tokenizer = model.get_tokenizer()
tokenizer.padding_side = "left"
tokenizer.pad_token_id = tokenizer.eos_token_id

# %%
@dataclass
class idCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        # Raw data
        _id = [feature.pop('id') for feature in features] if "id" in features[0].keys() else None
        prompts = [prompt_template.format(**instance).split(LABEL_SPLIT)[0] + LABEL_SPLIT for instance in features]
        return _id, prompts

data_collator = idCollator(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    return_tensors="pt",
    pad_to_multiple_of=8
)

# %%
dataloader = DataLoader(dataset['test'], collate_fn=data_collator, batch_size=64)

# %%
beam_search_params = vllm.SamplingParams(
    max_tokens=256,
    temperature=0,
    use_beam_search=True,
    length_penalty=0.75,
    best_of=3,
    top_p=1.0, 
    top_k=-1
)

sampling_params = vllm.SamplingParams(  # same as minillm
    max_tokens=256,
    top_k=-1,
    top_p=1.0,
    temperature=1.0,
)

# %%
torch.cuda.empty_cache()
predictions = []

start = timer()
for it, (_ids, data) in tqdm(enumerate(dataloader)):
    results = model.generate(data, sampling_params, use_tqdm=False)
    predictions.extend([{"id":_id, "summary": summary.strip()} for _id, summary in zip(_ids, [result.outputs[0].text for result in results])])

end = timer()
logger.info(f"Testing time: {end-start:.6f}s.")

# %%
logger.info(f"Example prediction: {predictions[0]}")

# %%
suffix = time.time_ns()
with open(WORK_DIR/f"predictions-{suffix}.json", 'w') as f:
    json.dump(predictions, f)

# %%
pred_ref = pd.merge(
    pd.DataFrame.from_records(predictions),
    pd.DataFrame.from_records(dataset['test'], columns=['id', 'summary']),
    on='id',
    how='inner',
    suffixes=['_pred', '_ref']
)

# %%
rouge = evaluate.load("rouge")
metrics = rouge.compute(predictions=pred_ref['summary_pred'], references=pred_ref['summary_ref'])
logger.info(metrics)

# %%
with open(WORK_DIR/f"metrics-{suffix}.json", 'w') as f:
    json.dump(metrics, f)
