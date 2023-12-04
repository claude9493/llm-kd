# NCCL_P2P_DISABLE=1 deepspeed --include localhost:6,7 kd.py

import os
# from transformers.utils import logging
from loguru import logger
from pathlib import Path
from transformers import AutoTokenizer
import torch
from copy import deepcopy
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, TrainerCallback
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, load_from_disk
# import deepspeed

from src.trainer import Seq2SeqKDArguments, Seq2SeqKDTrainer
from src.trainer import Seq2SeqLDKDArguments, Seq2SeqLDKDTrainer
from src.dataset import get_dataset

# os.environ['CUDA_VISIBLE_DEVICES'] = "6"

# logger = logging.get_logger(__name__)
DATA_NAME = "samsum"
MODEL_PATH = Path("../models/gpt2/base/")
TEACHER_MODEL_PATH = Path("results/samsum/gpt2-xlarge-sft/checkpoint-4256")
WORK_DIR = Path("./temp")  # Path("./results/samsum/gpt2-base-kd") 

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

teacher_model = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL_PATH,
                                                     torch_dtype=torch.float16,
                                                     load_in_8bit=False,
                                                     use_cache=False)
teacher_model.eval()

logger.debug(f"Student model #Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
logger.debug(f"Teacher model #Params: {sum(p.numel() for p in teacher_model.parameters() if p.requires_grad):,}")

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model,
    label_pad_token_id=tokenizer.pad_token_id,
    return_tensors="pt",
    pad_to_multiple_of=8
)

N_EPOCHS = 20
LR = 5e-4  # base
# LR = 1e-7  # Debug
# LR = 5e-5  # xlarge
BATCH_SIZE = 32  # 4*4*2
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4

training_args = Seq2SeqTrainingArguments(
    output_dir=str(WORK_DIR),
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    learning_rate=LR,
    lr_scheduler_type="cosine",
    # fp16=True,
    bf16=True,
    warmup_steps=100,
    weight_decay=1e-2,
    optim="adamw_torch",
    adam_epsilon=1e-3,  # thanks to https://zhuanlan.zhihu.com/p/507889212
    num_train_epochs=N_EPOCHS,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    save_strategy="steps",
    evaluation_strategy="steps",
    eval_steps=0.5/N_EPOCHS,
    save_steps=0.5/N_EPOCHS,
    logging_strategy="steps",
    logging_steps=0.05/N_EPOCHS,
    logging_dir=str(WORK_DIR/"logs"),
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="tensorboard",
    deepspeed="ds_config/ds_config_zero1.json"
)
kd_args = Seq2SeqLDKDArguments(
    ldkd_alpha=1,
    ldkd_beta=2,
    ldkd_top_ratio=0.99,
    kd_temperature=2.5
)

class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_step_end(self, args, state, control, **kwargs):
        if control.should_log:
            self._trainer.log(self._trainer.loss_dict)
        self._trainer.loss_dict = dict()


trainer = Seq2SeqLDKDTrainer(
    model=model,
    teacher_model=teacher_model,
    tokenizer=tokenizer,
    args=training_args,
    kd_args=kd_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=val_data
)
trainer.add_callback(CustomCallback(trainer))

logger.info("Start training!!!")
trainer.train()