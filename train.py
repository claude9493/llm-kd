# NCCL_P2P_DISABLE=1 deepspeed --include localhost:4,5,6,7 train.py

import os
from loguru import logger
from pathlib import Path
from transformers import AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, load_from_disk
import deepspeed

DATA_PATH = Path("./data/samsum")
MODEL_PATH = Path("../models/gpt2/base/")
WORK_DIR = Path('results/samsum/gpt2-base-sft')

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

logger.debug("Train data example:\n" + prompt_template.format(**dataset['train'][0]))


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
tokenizer.pad_token_id = tokenizer.eos_token_id

logger.debug(f"The padding token id is {tokenizer.pad_token_id}")

CUTOFF_LEN = 512
LABEL_SPLIT = "Summary:\n"

def generate_and_tokenize_prompt(instance, is_test=False):
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=CUTOFF_LEN,
            padding=True,
            return_tensors=None
        )
        if(
            result['input_ids'][-1] != tokenizer.eos_token_id
            and len(result['input_ids']) < CUTOFF_LEN
            and add_eos_token
        ):
            result['input_ids'].append(tokenizer.eos_token_id)
            result['attention_mask'].append(1)
        result['labels'] = result['input_ids'].copy()
        return result
    tokenized_full_prompt = tokenize(prompt_template.format(**instance))
    tokenized_user_prompt = tokenize(prompt_template.format(**instance).split(LABEL_SPLIT)[0] + LABEL_SPLIT, add_eos_token=False)
    user_prompt_len = len(tokenized_user_prompt['input_ids'])
    tokenized_full_prompt['labels'] = [-100]*user_prompt_len + tokenized_full_prompt['labels'][user_prompt_len:]
    if is_test:
        tokenized_user_prompt['_id'] = instance['id']
        return tokenized_user_prompt
    
    len_labels = len(tokenizer(instance['summary'])['input_ids'])
    tokenized_full_prompt['is_label_complete'] = len(tokenized_full_prompt['labels'][user_prompt_len:]) >= len_labels
    return tokenized_full_prompt

columns = ['input_ids', 'attention_mask', 'labels']

train_data = dataset['train'].map(generate_and_tokenize_prompt, num_proc=1) \
                             .filter(lambda instance: instance['is_label_complete']) \
                             .select_columns(columns) \
                             .with_format(type='torch')
                           
val_data = dataset['test'].map(generate_and_tokenize_prompt, num_proc=1) \
                          .filter(lambda instance: instance['is_label_complete']) \
                          .select_columns(columns) \
                          .with_format(type='torch', columns=columns)

logger.debug(f"Training data usage: {train_data.num_rows}/{dataset['train'].num_rows}.")
logger.debug(f"Validation data usage: {val_data.num_rows}/{dataset['validation'].num_rows}.")                     
                        
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, 
                                             torch_dtype=torch.float16, 
                                             load_in_8bit=False,
                                             use_cache=False) 

logger.debug(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model,
    label_pad_token_id=tokenizer.pad_token_id,
    return_tensors="pt",
    pad_to_multiple_of=8
)

N_EPOCHS = 20
LR = 5e-4  # base
# LR = 5e-5  # xlarge

PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
# N_GPUS = torch.cuda.device_count()
# TRAIN_LENGTH = len(train_data)
# N_STEPS = N_EPOCHS * TRAIN_LENGTH / (PER_DEVICE_BATCH_SIZE * N_GPUS)
# N_EVAL_PER_EPOCH =  2
# EVAL_STEPS = int(TRAIN_LENGTH / (PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * N_GPUS) / N_EVAL_PER_EPOCH)
# logger.debug(f"Evaluate per {EVAL_STEPS} steps")

training_args = Seq2SeqTrainingArguments(
    output_dir=str(WORK_DIR),
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    weight_decay=1e-2,
    optim="adamw_torch",
    num_train_epochs=N_EPOCHS,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    save_strategy="steps",
    evaluation_strategy="steps",
    eval_steps=0.5/N_EPOCHS,
    save_steps=0.5/N_EPOCHS,
    logging_dir=str(WORK_DIR/"logs"),
    logging_strategy="steps",
    logging_steps=0.05/N_EPOCHS,
    save_total_limit=4,
    load_best_model_at_end=True,
    report_to="tensorboard",
    deepspeed="ds_config/ds_config_zero1.json"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=val_data
)

trainer.train()