import os
import yaml
from copy import deepcopy
from itertools import product
from functools import reduce
from operator import getitem
import subprocess

CONFIG_TEMPLATE_PATH = "configs/sweep/dolly-tinyllama-sft-lora8.yaml"
with open(CONFIG_TEMPLATE_PATH, 'r') as f:
    template = yaml.full_load(f)

CONFIG_LOCATION, CONFIG_NAME = os.path.split(CONFIG_TEMPLATE_PATH)
CONFIG_NAME, CONFIG_EXT = CONFIG_NAME.split('.')
os.makedirs(os.path.join(CONFIG_LOCATION, CONFIG_NAME), exist_ok=True)

TRAIN_COMMAND = "NCCL_P2P_DISABLE=1 deepspeed --include localhost:0,1 train.py {config}"
TEST_COMMAND = "accelerate launch inference-accelerate.py -m {ckpt} -t ../models/llama/7B -d dolly --metric rouge"

OUTPUT_DIR = "result/dolly/"

hp_suggests = {
    "lora_config.r": [4], #[4, 8, 16, 32],
    "lora_config.lora_alpha": [1, 4, 8]# [1, 4, 8, 16, 32, 64]
}

hp_space = [dict(zip(hp_suggests.keys(), element)) for element in product(*list(hp_suggests.values()))]

hp_alias = {
    "lora_config.r": "r",
    "lora_config.lora_alpha": "a"
}

def _new_config(template: dict , 
                suggest: dict):
    config = deepcopy(template)
    for k, v in suggest.items():
        sub_config = reduce(getitem, k.split('.')[:-1], config)
        sub_config[k.split('.')[-1]] = v
    config['output_dir'] += '-' + '-'.join([f"{hp_alias[k]}{v}" for k, v in suggest.items()])
    return config

for _id, suggest in enumerate(hp_space):
    config = _new_config(template, suggest)
    config_path = os.path.join(CONFIG_LOCATION, CONFIG_NAME, '-'.join([f"{hp_alias[k]}{v}" for k, v in suggest.items()])) + '.' + CONFIG_EXT
    output_dir = os.path.join(OUTPUT_DIR, config['output_dir'])
    with open(config_path, 'w+') as f:
        yaml.dump(config, f)
    train_command = TRAIN_COMMAND.format(config = config_path)
    with open("CURRENT_COMMAND", 'w+') as f:
        f.write(f"{_id}/{len(hp_space)}\n")
        f.write(train_command)
    subprocess.call(train_command, shell=True)
    ckpts = [dir for dir in os.listdir(output_dir) if "checkpoint" in dir]
    latest_ckpt = os.path.join(output_dir, f"checkpoint-{max([int(dir.split('-')[-1]) for dir in ckpts])}")
    test_command = TEST_COMMAND.format(ckpt=latest_ckpt)
    with open("CURRENT_COMMAND", 'w+') as f:
        f.write(f"{_id}/{len(hp_space)}\n")
        f.write(test_command)
    subprocess.call(train_command, shell=True)