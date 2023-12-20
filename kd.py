# NCCL_P2P_DISABLE=1 deepspeed --include localhost:6,7 kd.py

import os
import sys
# from transformers.utils import logging
from loguru import logger
from pathlib import Path
import torch
from copy import deepcopy
from transformers import HfArgumentParser
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments
import deepspeed

from src.trainer import Seq2SeqKDTrainer, Seq2SeqLDKDTrainer, KDLoggingCallback
from arguments import ModelArguments, DataArguments, KDArguments
from src.dataset import get_dataset
from src.utils.archive import ArchiveScriptCallback

# os.environ['CUDA_VISIBLE_DEVICES'] = "6"


parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, KDArguments))

print(sys.argv)

# if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
if sys.argv[-1].endswith(".yaml"):  # [-1] in case deepspeed is used
    model_args, data_args, training_args, kd_args = parser.parse_yaml_file(
        yaml_file=os.path.abspath(sys.argv[-1]), 
        allow_extra_keys=False)
else:
    model_args, data_args, training_args, kd_args = parser.parse_args_into_dataclasses()

# Handle arguments
MODEL_PATH = model_args.model_name_or_path
WORK_DIR = Path('results')/data_args.dataset_name/training_args.output_dir
training_args.output_dir = str(WORK_DIR)
training_args.logging_dir = str(WORK_DIR/__import__('transformers').training_args.default_logdir())

# logger = logging.get_logger(__name__)
# DATA_NAME = "samsum"
# MODEL_PATH = Path("../models/gpt2/base/")
# TEACHER_MODEL_PATH = Path("results/samsum/gpt2-xlarge-sft/checkpoint-4256")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
tokenizer.pad_token_id = tokenizer.eos_token_id

logger.debug(f"The padding token id is {tokenizer.pad_token_id}")

# __import__('ipdb').set_trace()

_data_class = get_dataset(data_args.dataset_name)
train_data = _data_class.get_train(tokenizer)
val_data = _data_class.get_val(tokenizer)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, load_in_8bit=False, use_cache=False
)

teacher_model = AutoModelForCausalLM.from_pretrained(
    kd_args.teacher_model_path,
    torch_dtype=torch.float16,
    load_in_8bit=False,
    use_cache=False,
)
teacher_model.eval()

# kd_args = kd_args.kd_args  # stupid but necessary

logger.debug(f"Student model #Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
logger.debug(f"Teacher model #Params: {sum(p.numel() for p in teacher_model.parameters() if p.requires_grad):,}")

# from deepspeed.inference.config import DeepSpeedInferenceConfig, DeepSpeedTPConfig

# ds_inference_cfg = DeepSpeedInferenceConfig(
#     tensor_parallel = DeepSpeedTPConfig(
#         tp_size=torch.cuda.device_count()
#     )
# )

ds_inference_cfg = dict(
    tensor_parallel = dict(
        tp_size=torch.cuda.device_count()
    )
)


deepspeed.init_inference(teacher_model, 
                         config=ds_inference_cfg)

logger.debug(f"[RANK {os.environ['LOCAL_RANK']}] Teacher model device: {teacher_model.device}")
logger.debug(f"[RANK {os.environ['LOCAL_RANK']}] Teacher model forward propagation test: {teacher_model(**train_data[0])}")

# __import__('ipdb').set_trace()

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model,
    label_pad_token_id=tokenizer.pad_token_id,
    return_tensors="pt",
    pad_to_multiple_of=8
)


trainer = Seq2SeqLDKDTrainer(
    model=model,
    teacher_model=teacher_model,
    tokenizer=tokenizer,
    args=training_args,
    kd_args=kd_args.kd_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=val_data
)

trainer.add_callback(KDLoggingCallback(trainer))
trainer.add_callback(ArchiveScriptCallback(training_args.output_dir))

logger.info("Start training!!!")
trainer.train(resume_from_checkpoint=model_args.checkpoint)