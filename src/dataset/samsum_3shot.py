# from transformers.utils import logging
from copy import deepcopy
from loguru import logger
from pathlib import Path
from functools import partial
from datasets import load_dataset, load_from_disk
from .base import DataInfo, generate_and_tokenize_prompt, columns

# logger = logging.get_logger(__name__)
NAME = "Samsum-3shot"

prompt_template = """[INST] <<SYS>>
Use the Input to provide a summary of a conversation. For example,
----------
Input:
Amanda: I baked cookies. Do you want some? 
Jerry: Sure! 
Amanda: I'll bring you tomorrow :-)	

Summary: Amanda baked cookies and will bring Jerry some tomorrow.
----------
Input:
Sam: I'm so sorry. I can't make it on time. 
Sandra: Should we start without you? 
Sam: Please do. I'll be 30 min late. 
Staś: Ok	

Summary: Sam will be 30 minutes late. Sandra and Staś will start without Sam.
----------
Input:
Mark: I just shipped the goods Mark: Tomorrow I'll send you the tracking number 
George: Thanks!	

Summary: Mark just shipped the goods and he will send George the tracking number tomorrow.
----------
<</SYS>>

Input:
{dialogue}

Summary:
{summary}
"""

info = DataInfo(
    name="Samsum-3shot",
    path=Path("./data/samsum"),
    prompt_template=prompt_template,
    label_split="Summary:\n",
    label_column="summary",
    cutoff_len=768
)


generate_and_tokenize_prompt = partial(generate_and_tokenize_prompt, info=info)

dataset = load_from_disk(info.path)
logger.debug(f"Dataset: {dataset}")


def get_val(tokenizer):
    val_data = dataset['validation'].map(partial(generate_and_tokenize_prompt, tokenizer=tokenizer), num_proc=1) \
                          .filter(lambda instance: instance['is_label_complete']) \
                          .select_columns(columns) \
                          .with_format(type='torch', columns=columns)
    logger.debug(f"Validation data usage: {val_data.num_rows}/{dataset['validation'].num_rows}.")      
    return val_data

def get_test(tokenizer):
    columns_test = deepcopy(columns)
    columns_test.append('_id')
    test_data = dataset['test'].map(partial(generate_and_tokenize_prompt, tokenizer=tokenizer, is_test=True), num_proc=1) \
                          .select_columns(columns_test) \
                          .with_format(type='torch', columns=columns_test)
    logger.debug(f"Test data usage: {test_data.num_rows}/{dataset['test'].num_rows}.")      
    return test_data