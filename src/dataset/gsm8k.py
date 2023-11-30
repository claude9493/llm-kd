# from transformers.utils import logging
from loguru import logger
from pathlib import Path
from functools import partial
from datasets import load_dataset, load_from_disk
from .base import generate_and_tokenize_prompt, columns, DataInfo

NAME = "GSM8K"

prompt_template = """Question:
{question}

Answer:
{answer}
"""

info = DataInfo(
    name="GSM8K",
    path=Path("./data/gsm8k"),
    prompt_template=prompt_template,
    label_split="Answer:\n",
    label_column="answer",
    cutoff_len=256
)

generate_and_tokenize_prompt = partial(generate_and_tokenize_prompt, info=info)

dataset = load_from_disk(info.path)
logger.debug(f"Dataset: {dataset}")

def get_train(tokenizer):
    train_data = dataset['train'].map(partial(generate_and_tokenize_prompt, tokenizer=tokenizer), num_proc=1) \
                             .filter(lambda instance: instance['is_label_complete']) \
                             .select_columns(columns) \
                             .with_format(type='torch')
    logger.debug("Train data example:\n" + prompt_template.format(**dataset['train'][0]))
    logger.debug(f"Training data usage: {train_data.num_rows}/{dataset['train'].num_rows}.")
    return train_data

def get_val(tokenizer):
    val_data = dataset['validation'].map(partial(generate_and_tokenize_prompt, tokenizer=tokenizer), num_proc=1) \
                          .filter(lambda instance: instance['is_label_complete']) \
                          .select_columns(columns) \
                          .with_format(type='torch', columns=columns)
    logger.debug(f"Validation data usage: {val_data.num_rows}/{dataset['validation'].num_rows}.")      
    return val_data