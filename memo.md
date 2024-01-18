```bash
conda create -n llmkd python==3.10.13
python -m pip install --upgrade pip
ls -d dependencies/* |grep whl | xargs -I {} python -m pip install {}
cat requirements.txt | xargs -I {} python -m pip install {}

DS_BUILD_CPU_ADAM=1  BUILD_UTILS=1  pip install deepspeed -U

python -c "from deepspeed.ops.op_builder import CPUAdamBuilder; CPUAdamBuilder().load()"
```


ls -d  results/gsm8k/llama-7b-sft-lora8/*/*| grep "global_step\|rng_state" | xargs -I {} rm -r {}