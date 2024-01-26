# from transformers.utils import logging
from loguru import logger
from copy import deepcopy
from pathlib import Path
from functools import partial
from datasets import load_dataset, load_from_disk, concatenate_datasets
from .base import generate_and_tokenize_prompt, columns, DataInfo

NAME = "Self-instruct-eval"

prompt_template = "{text}"


info = DataInfo(
    name="self-instruct-eval",
    path=Path("./data/self-instruct-eval"),
    prompt_template=prompt_template,
    label_split="### Response:\n",
    label_column="output",
    cutoff_len=512
)

# generate_and_tokenize_prompt = partial(generate_and_tokenize_prompt, info=info)

dataset = load_from_disk(info.path)
logger.debug(f"Dataset: {dataset}")


def get_test(tokenizer):
    columns_test = deepcopy(columns)
    columns_test.append('id')
    gtp = partial(generate_and_tokenize_prompt, info=info, tokenizer=tokenizer, is_test=True)
    if 'id' not in dataset['test'].column_names:
        dataset['test'] = dataset['test'].add_column('id', list(range(dataset['test'].num_rows)))
    test_data = dataset['test'].map(gtp, num_proc=1) \
            .select_columns(columns_test) \
            .with_format(type='torch', columns=columns, output_all_columns=True)

    logger.debug(f"Testing data usage: {test_data.num_rows}/{dataset['test'].num_rows}.")      
    return test_data