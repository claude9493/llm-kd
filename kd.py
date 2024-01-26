# NCCL_P2P_DISABLE=1 deepspeed --include localhost:0,1 kd.py

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

from src.trainer import KD_TRAINERS_DICT, KDLoggingCallback
from arguments import ModelArguments, DataArguments, KDArguments
from src.dataset import get_dataset
from src.utils.archive import ArchiveScriptCallback

# os.environ['CUDA_VISIBLE_DEVICES'] = "6"


parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, KDArguments))

print(sys.argv)
config_file = None
for _arg in sys.argv:
    if _arg.endswith(".yaml") and _arg.startswith("configs"):
        config_file = os.path.abspath(_arg)
        break

# if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
# if sys.argv[-1].endswith(".yaml"):  # [-1] in case deepspeed is used
if config_file:
    model_args, data_args, training_args, kd_args = parser.parse_yaml_file(
        yaml_file=config_file, allow_extra_keys=False)
    logger.debug(f"Config file: {config_file}")
else:
    model_args, data_args, training_args, kd_args = parser.parse_args_into_dataclasses()

# Handle arguments
MODEL_PATH = model_args.model_name_or_path
WORK_DIR = Path('results')/data_args.dataset_name/training_args.output_dir
training_args.output_dir = str(WORK_DIR) # results/samsum/openllama-3bv2-kd
training_args.logging_dir = str(WORK_DIR/__import__('transformers').training_args.default_logdir()) #results/samsum/openllama-3bv2-kd/runs/Jan02_15-06-59_notebook-69bbf15b-61f9-47bb-84ef-16a148cb8d60

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, **model_args.tokenizer_kwargs) # tokenizer from stu
tokenizer.padding_side = "left"
tokenizer.pad_token_id = tokenizer.eos_token_id

logger.debug(f"The padding token id is {tokenizer.pad_token_id}")


_data_class = get_dataset(data_args.dataset_name)
train_data = _data_class.get_train(tokenizer)# .select(list(range(200)))
val_data = _data_class.get_val(tokenizer)# .select(list(range(50)))

print(train_data)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, load_in_8bit=False, use_cache=False
)
logger.debug(f"Student model loaded: {MODEL_PATH}. #Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

if model_args.lora_config is not None:
    from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
    from src.utils import find_all_linear_names
    peft_model_path = model_args.lora_config.get("peft_model_path", None)
    if peft_model_path:
        model = prepare_model_for_kbit_training(model)
        model.load_adapter(peft_model_path)
        model.enable_adapters()
        logger.debug(f"Load adapter from {peft_model_path}")
        # __import__('ipdb').set_trace()
    else:
        model_args.lora_config.pop("peft_model_path", None)
        lora_config = LoraConfig(**model_args.lora_config)
        if lora_config.target_modules == "all":
            lora_config.target_modules = find_all_linear_names(model)
        logger.debug(f"Lora target modules: {lora_config.target_modules}")
        model = get_peft_model(prepare_model_for_kbit_training(model), lora_config)
        trainable_params, all_param = model.get_nb_trainable_parameters()
        logger.debug(f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}")

teacher_model = None

if kd_args.teacher_model_path:
    teacher_model = AutoModelForCausalLM.from_pretrained(
        kd_args.teacher_model_path,
        torch_dtype=torch.float16,
        load_in_8bit=False,
        use_cache=False,
    )
    teacher_model.eval()
    logger.debug(f"Teacher model loaded: {kd_args.teacher_model_path}. #Params: {sum(p.numel() for p in teacher_model.parameters() if p.requires_grad):,}")

# from deepspeed.inference.config import DeepSpeedInferenceConfig, DeepSpeedTPConfig

# ds_inference_cfg = DeepSpeedInferenceConfig(
#     tensor_parallel = DeepSpeedTPConfig(
#         tp_size=torch.cuda.device_count()
#     )
# )

# ds_inference_cfg = dict(
#     tensor_parallel = dict(
#         tp_size=torch.cuda.device_count()
#     )
# )


# deepspeed.init_inference(teacher_model, 
#                          config=ds_inference_cfg)

# logger.debug(f"[RANK {os.environ['LOCAL_RANK']}] Teacher model device: {teacher_model.device}")
# logger.debug(f"[RANK {os.environ['LOCAL_RANK']}] Teacher model forward propagation test: {teacher_model(**train_data[0])}")

# __import__('ipdb').set_trace()

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
logger.debug(f"Evaluation per {training_args.eval_steps} steps. Saving per {training_args.save_steps} steps. Logging per {training_args.logging_steps} steps.")
training_args.label_names = ['labels']  # If eval loss is not logged, try uncomment this line

# training_args.eval_steps = 1  # For Debug only

if kd_args.kd_type == "dfkd":
    data_collator = __import__("transformers").default_data_collator

trainer = KD_TRAINERS_DICT[kd_args.kd_type](
    model=model,
    teacher_model=teacher_model,
    tokenizer=tokenizer,
    args=training_args,
    kd_args=kd_args.kd_args,
    data_collator=data_collator,
    train_dataset=train_data,
    eval_dataset=val_data,
    tensor_parallel=kd_args.tensor_parallel
)

archive_script_callback = ArchiveScriptCallback(
    training_args.output_dir, 
    config_file=config_file
)

trainer.add_callback(KDLoggingCallback(trainer))
trainer.add_callback(archive_script_callback)
logger.debug("Trainer callbacks: " + trainer.callback_handler.callback_list.replace('\n', ', '))


logger.info("Start training!!!")
trainer.train(resume_from_checkpoint=model_args.checkpoint)