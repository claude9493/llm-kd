# %% [markdown]
# # Load Data and Model

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, DataCollatorForSeq2Seq
import torch
import torch.nn.functional as F
import transformers

import datasets
from torch.utils.data import DataLoader
from dataclasses import dataclass

from src.dataset import get_dataset
from peft import PeftConfig, LoraConfig, prepare_model_for_kbit_training, get_peft_model

MODEL_PATH = "../models/llama2/7B/"
MODEL_PATH = "../models/tinyllama/tiny_llama_2_5T/"
# MODEL_PATH = "results/datafree/tinyllama-2_5T-kd/checkpoint-24/"

# %%
tokenizer = AutoTokenizer.from_pretrained("../models/tinyllama/tiny_llama_2_5T/")
tokenizer.pad_token_id = tokenizer.eos_token_id

# %%
dataset_name = "dolly"
_data_class = get_dataset(dataset_name)
train_data = _data_class.get_train(tokenizer).select(list(range(10)))
val_data = _data_class.get_val(tokenizer).select(list(range(5)))
train_data

# %%
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, load_in_8bit=False, use_cache=False
)

# %%
adapter_config = LoraConfig(**dict(
    r = 8,
    lora_alpha = 16,
    lora_dropout = 0.05,
    target_modules = ["q_proj", "k_proj", "v_proj"]
))
adapters = ["adapter-0", "adapter-1"]


def set_adapter(model, adapter):
    model.disable_adapter()
    model.set_adapter(adapter)
    for name, param in model.named_parameters():
         if adapter in name:
             param.requires_grad = True

# %% [markdown]
# ## Setup adapters

# %%
model = get_peft_model(prepare_model_for_kbit_training(base_model), adapter_config, adapter_name=adapters[0])#, mixed=True)
trainable_params, all_param = model.get_nb_trainable_parameters()

# logger.debug(f"{list_trainable_parameters(model)}")
model.add_adapter(adapters[1], adapter_config)
set_adapter(model, adapters[1])
trainable_params, all_param = model.get_nb_trainable_parameters()

# logger.debug(f"{list_trainable_parameters(model)}")
# model.enable_adapters()
set_adapter(model, adapters[0])

# %%
model.cuda(0)

# %%
for name, p in model.named_parameters():
    if p.requires_grad:
        print(name)

# %%
from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=5e-4, eps=1e-4)

# %% [markdown]
# # Train step by step

# %% [markdown]
# ## Set train and test data

# %%
label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    return_tensors="pt",
    pad_to_multiple_of=8
)

dataloader = DataLoader(train_data, 
                        collate_fn=data_collator, 
                        batch_size=2)

val_loader = DataLoader(val_data,
                        collate_fn=data_collator,
                        batch_size=4)

# %%
data = next(dataloader._get_iterator())

for k, v in data.items():
    data[k] = v.cuda(0)

# %%
test_data = next(val_loader._get_iterator())
for k, v in test_data.items():
    test_data[k] = v.cuda(0)

# %% [markdown]
# ## SFT two adapters

# %%
model.base_model.model.model.layers[1].self_attn.q_proj.lora_A["adapter-0"].weight

# %%
model.base_model.model.model.layers[1].self_attn.q_proj.lora_A["adapter-1"].weight

# %%
optimizer.zero_grad()
model.set_adapter(adapters[0])
print(f"{model.active_adapter}:")
print(f"Eval Loss: {model(**test_data).loss}")

output_0 = model(**data)
loss_0 = output_0.loss
print(f"Train Loss: {loss_0}")

# %%
loss_0.backward()
optimizer.step()

print(f"Train Loss: {model(**data).loss}")
print(f"Eval Loss: {model(**test_data).loss}")

# %%
optimizer.zero_grad()
model.set_adapter(adapters[1])
print(f"{model.active_adapter}:")
print(f"Eval Loss: {model(**test_data).loss}")

output_1 = model(**data)
loss_1 = output_1.loss
print(f"Train Loss: {loss_1}")

# %%
loss_1.backward()
optimizer.step()

print(f"Train Loss: {model(**data).loss}")
print(f"Eval Loss: {model(**test_data).loss}")

# %% [markdown]
# ## SFT-0 + KD-1

# %%
import torch.nn.functional as F

# %%
optimizer.zero_grad()
model.set_adapter(adapters[0])
with torch.no_grad():
    logits_0 = model(**data).logits
model.set_adapter(adapters[1])
outputs = model(**data)
loss_gt = outputs.loss
logits_1 = outputs.logits

tmpt = 1
loss_mask = torch.where(((data['labels'] < 0) | (data['labels'] == tokenizer.pad_token_id)), 0, 1).unsqueeze(-1)
input, target = logits_1, logits_0
input = F.log_softmax(input/tmpt, dim=-1)  # , dtype=torch.float32)
target = F.softmax(target/tmpt, dim=-1)  # , dtype=torch.float32)

loss_kd = F.kl_div((input*loss_mask).nan_to_num(0,0,0), (target*loss_mask).nan_to_num(0,0,0), reduction="mean") * tmpt**2

print(f"loss_gt: {loss_gt}\tloss_kd: {loss_kd}")

# %%
loss = (loss_gt + loss_kd) / 2

print(f"Train Loss: {loss}")
print(f"Eval Loss: {model(**test_data).loss}")

# %%
loss.backward()
optimizer.step()

print(f"Train Loss: {model(**data).loss}")
print(f"Eval Loss: {model(**test_data).loss}")

# %%
model.merge_and_unload(adapter_names=['adapter-1'])

# %%
model.save_pretrained("./results/debug/round1")

# %%
model.merge_and_unload(safe_merge=True, adapter_names=["adapter-1"])


