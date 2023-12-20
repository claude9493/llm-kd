# from transformers.utils import logging
from loguru import logger
from copy import deepcopy
from pathlib import Path
from functools import partial
from datasets import load_dataset, load_from_disk, concatenate_datasets
from .base import generate_and_tokenize_prompt, columns, DataInfo

NAME = "Dolly"

prompt_template = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n{response}"
)

prompt_template2 = (  # No context
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n{response}"
)

info = DataInfo(
    name="dolly",
    path=Path("./data/dolly"),
    prompt_template=prompt_template,
    label_split="### Response:\n",
    label_column="response",
    cutoff_len=512
)

info2 = DataInfo(
    name="dolly",
    path=Path("./data/dolly"),
    prompt_template=prompt_template2,
    label_split="### Response:\n",
    label_column="response",
    cutoff_len=512
)

# generate_and_tokenize_prompt = partial(generate_and_tokenize_prompt, info=info)

dataset = load_from_disk(info.path)
logger.debug(f"Dataset: {dataset}")

def get_train(tokenizer):
    gtp = partial(generate_and_tokenize_prompt, info=info, tokenizer=tokenizer)
    gtp2 = partial(generate_and_tokenize_prompt, info=info2, tokenizer=tokenizer)
    train_data = concatenate_datasets([
        dataset['train'].filter(lambda x: len(x['context'])!=0).map(gtp, num_proc=1),
        dataset['train'].filter(lambda x: len(x['context'])==0).map(gtp2, num_proc=1)]) \
            .filter(lambda instance: instance['is_label_complete']) \
            .select_columns(columns) \
            .with_format(type='torch')
    logger.debug("Train data example:\n" + prompt_template.format(**dataset['train'][0]))
    logger.debug(f"Training data usage: {train_data.num_rows}/{dataset['train'].num_rows}.")
    return train_data

def get_val(tokenizer):
    gtp = partial(generate_and_tokenize_prompt, info=info, tokenizer=tokenizer)
    gtp2 = partial(generate_and_tokenize_prompt, info=info2, tokenizer=tokenizer)
    val_data = concatenate_datasets([
        dataset['validation'].filter(lambda x: len(x['context'])!=0).map(gtp, num_proc=1),
        dataset['validation'].filter(lambda x: len(x['context'])==0).map(gtp2, num_proc=1)]) \
            .filter(lambda instance: instance['is_label_complete']) \
            .select_columns(columns) \
            .with_format(type='torch')
    logger.debug(f"Validation data usage: {val_data.num_rows}/{dataset['validation'].num_rows}.")      
    return val_data

def get_test(tokenizer):
    columns_test = deepcopy(columns)
    columns_test.append('_id')
    gtp = partial(generate_and_tokenize_prompt, info=info, tokenizer=tokenizer, is_test=True)
    gtp2 = partial(generate_and_tokenize_prompt, info=info2, tokenizer=tokenizer, is_test=True)
    if 'id' not in dataset['test'].column_names:
        dataset['test'] = dataset['test'].add_column('id', list(range(dataset['test'].num_rows)))
    test_data = concatenate_datasets([
        dataset['test'].filter(lambda x: len(x['context'])!=0).map(gtp, num_proc=1),
        dataset['test'].filter(lambda x: len(x['context'])==0).map(gtp2, num_proc=1)]) \
            .select_columns(columns_test) \
            .with_format(type='torch', columns=columns_test)

    logger.debug(f"Testing data usage: {test_data.num_rows}/{dataset['validation'].num_rows}.")      
    return test_data