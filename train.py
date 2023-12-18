# NCCL_P2P_DISABLE=1 deepspeed --include localhost:6,7 train.py

import os
import sys
from loguru import logger
from pathlib import Path
import torch
from transformers import HfArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from src.dataset import get_dataset

from arguments import ModelArguments, DataArguments

parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))

# if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
if sys.argv[-1].endswith(".yaml"):
    model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[-1]), allow_extra_keys=False)
    logger.debug(f"Config file: {os.path.abspath(sys.argv[-1])}")
else:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# Handle arguments
MODEL_PATH = model_args.model_name_or_path

WORK_DIR = Path('results')/data_args.dataset_name/training_args.output_dir
training_args.output_dir = str(WORK_DIR)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, **model_args.tokenizer_kwargs)
# tokenizer.padding_side = "left"  # for GPT2
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token_id = tokenizer.eos_token_id  # GPT2

logger.debug(f"The padding token id is {tokenizer.pad_token_id}")

# Load data
_data_class = get_dataset(data_args.dataset_name)
train_data = _data_class.get_train(tokenizer)
val_data = _data_class.get_val(tokenizer)

# Load model
torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, 
                                             torch_dtype=torch_dtype, 
                                             load_in_8bit=False,
                                             use_cache=False) 

logger.debug(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

if model_args.lora_config is not None:
    from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
    from src.utils import find_all_linear_names
    target_modules = find_all_linear_names(model)
    target_modules = ["q_proj", "k_proj", "v_proj"]
    logger.debug(f"Lora target modules: {target_modules}")
    lora_config = LoraConfig(**model_args.lora_config, 
                             target_modules=target_modules)
    model = get_peft_model(prepare_model_for_kbit_training(model), lora_config)
    model.print_trainable_parameters()
    # model.train()
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model,
    label_pad_token_id=tokenizer.pad_token_id,
    return_tensors="pt",
    pad_to_multiple_of=8
)

# Trainer
training_args.eval_steps /= training_args.num_train_epochs
training_args.save_steps /= training_args.num_train_epochs
training_args.logging_steps /= training_args.num_train_epochs
training_args.label_names = ['labels']

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=val_data
)

# __import__('ipdb').set_trace()
logger.debug("Start Training!!!")
trainer.train(resume_from_checkpoint=model_args.checkpoint)