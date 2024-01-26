# from transformers.utils import logging
from loguru import logger
from copy import deepcopy
from pathlib import Path
from functools import partial
from datasets import load_dataset, load_from_disk
from .base import generate_and_tokenize_prompt, columns, DataInfo

NAME = "MBPP"

three_shots = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\nWrite a function to find squares of individual elements in a list using lambda function.\n\n"
    "### Response:\ndef square_nums(nums):\r\n square_nums = list(map(lambda x: x ** 2, nums))\r\n return square_nums\n\n"
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\nWrite a python function to find the minimum number of rotations required to get the same string.\n\n"
    "### Response:\ndef find_Rotations(str): \r\n    tmp = str + str\r\n    n = len(str) \r\n    for i in range(1,n + 1): \r\n        substring = tmp[i: i+n] \r\n        if (str == substring): \r\n            return i \r\n    return n \n\n"
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\nWrite a function to get the n smallest items from a dataset.\n\n"
    "### Response:\nimport heapq\r\ndef small_nnum(list1,n):\r\n  smallest=heapq.nsmallest(n,list1)\r\n  return smallest\n\n"
)

prompt_template = three_shots+(
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{text}\n\n### Response:\n{code}"
)


info = DataInfo(
    name="MBPP",
    path=Path("./data/mbpp"),
    prompt_template=prompt_template,
    label_split="### Response:\n",
    label_column="code",
    cutoff_len=512
)

generate_and_tokenize_prompt = partial(generate_and_tokenize_prompt, info=info)

dataset = load_from_disk(info.path)
# if 'id' not in dataset.column_names:
#     test_data = dataset.add_column('_id', list(range(dataset.num_rows)))
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

def get_test(tokenizer):
    columns_test = deepcopy(columns)
    columns_test.append('id')
    if 'id' not in dataset['test'].column_names:
        dataset['test'] = dataset['test'].add_column('id', list(range(dataset['test'].num_rows)))

    test_data = dataset['test'].map(partial(generate_and_tokenize_prompt, tokenizer=tokenizer, is_test=True), num_proc=1) \
                          .select_columns(columns_test) \
                          .with_format(type='torch', columns=columns, output_all_columns=True)
    logger.debug(f"Test data usage: {test_data.num_rows}/{dataset['test'].num_rows}.")
    return test_data