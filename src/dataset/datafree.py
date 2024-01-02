# from transformers.utils import logging
from copy import deepcopy
from loguru import logger
from pathlib import Path
from functools import partial
from datasets import load_dataset, load_from_disk
from .base import DataInfo, generate_and_tokenize_prompt, columns

# logger = logging.get_logger(__name__)
NAME = "DataFree"

info = DataInfo(
    name="datafree",
    path=None,
    prompt_template=None,
    label_split=None,
    label_column=None,
    cutoff_len=512
)


generate_and_tokenize_prompt = partial(generate_and_tokenize_prompt, info=info)

# dataset = load_from_disk(info.path)
# logger.debug(f"Dataset: {dataset}")

# def get_train(tokenizer):
#     train_data = dataset['train'].map(partial(generate_and_tokenize_prompt, tokenizer=tokenizer), num_proc=1) \
#                              .filter(lambda instance: instance['is_label_complete']) \
#                              .select_columns(columns) \
#                              .with_format(type='torch')
#     logger.debug("Train data example:\n" + prompt_template.format(**dataset['train'][0]))
#     logger.debug(f"Training data usage: {train_data.num_rows}/{dataset['train'].num_rows}.")
#     return train_data
