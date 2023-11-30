# NCCL_P2P_DISABLE=1 deepspeed --include localhost:6,7 train.py

import os
from transformers.utils import logging
# from loguru import logger
from pathlib import Path
from transformers import AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from src.dataset import get_dataset
import deepspeed

logger = logging.get_logger(__name__)

DATA_NAME = "gsm8k"
MODEL_PATH = Path("../models/gpt2/xlarge/")
WORK_DIR = Path(f'results/{DATA_NAME}/gpt2-xlarge-sft')

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
tokenizer.pad_token_id = tokenizer.eos_token_id

logger.debug(f"The padding token id is {tokenizer.pad_token_id}")

_data_class = get_dataset(DATA_NAME)
train_data = _data_class.get_train(tokenizer)
val_data = _data_class.get_val(tokenizer)
                        
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

N_EPOCHS = 10
# LR = 5e-4  # base
LR = 5e-5  # xlarge
BATCH_SIZE=32  # 4*4*2
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4

training_args = Seq2SeqTrainingArguments(
    output_dir=str(WORK_DIR),
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    fp16=True,
    adam_epsilon=1e-3,
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
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="tensorboard",
    deepspeed="ds_config/ds_config_zero1.json",
    # auto_find_batch_size=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=val_data
)

trainer.train()