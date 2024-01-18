import os
from copy import deepcopy
import math 
from torch.nn.modules import Module
from loguru import logger
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import tensor_parallel as tp

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from peft import PeftConfig, LoraConfig, prepare_model_for_kbit_training, get_peft_model

from transformers import Seq2SeqTrainer, TrainerCallback, GenerationConfig
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from transformers.trainer import (
    is_sagemaker_mp_enabled, 
    is_peft_available, 
    unwrap_model, 
    is_apex_available,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
)

if is_apex_available():
    from apex import amp

if TYPE_CHECKING:
    from transformers.data.data_collator import DataCollator
    from transformers.modeling_utils import PreTrainedModel
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    from transformers.trainer_callback import TrainerCallback
    from transformers.trainer_utils import EvalPrediction, PredictionOutput
    from transformers.training_args import TrainingArguments

from .kd_arguments import *
from .kd_trainer import kld_loss

class AltKDCallback(TrainerCallback):
    def __init__(self, switch_freq) -> None:
        self.switch_freq = switch_freq
        super().__init__()
        
    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.epoch > 0 and math.ceil(state.epoch) % self.switch_freq == 0:
            control.train_state = "kd" if control.train_state == "sft" else "sft"
        if state.is_world_process_zero:
            logger.debug(f"Epoch: {state.epoch}. Training state: {control.train_state}")
        return super().on_epoch_begin(args, state, control, **kwargs)
    

class Seq2SeqAltKDTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", nn.Module] = None,
        teacher_model = None,
        args: "TrainingArguments" = None,
        kd_args: "Seq2SeqKDArguments" = None,
        data_collator: Optional["DataCollator"] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
        compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
        callbacks: Optional[List["TrainerCallback"]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        tensor_parallel: bool = False
    ):
        self.adapters = ("adapter1", "adapter2")  # (sft, kd)

        adapter_config = LoraConfig(**kd_args.altkd_adapter_config)
        model = self._setup_adapters(model, adapter_config)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        
        # Load Teacher model
        self.tensor_parallel = tensor_parallel

        if tensor_parallel:
            self.model = tp.tensor_parallel(self.model)
            # print(type(self.model))
            logger.debug("The model is tensor-parallized.")
        else:
            self._move_model_to_device(self.model, args.device)
            logger.debug(f"Models are moved to device {args.device}")
        
        self.kd_args = kd_args
        self.loss_dict = dict()
        
        self.control.train_state = "sft"  # (sft, kd)
        self.add_callback(AltKDCallback(kd_args.altkd_switch_freq))

    def _setup_adapters(self, model: Module, adapter_config: PeftConfig):
        assert len(self.adapters) == 2, "Only support 2 adapters' alternative KD right now."
        # model = prepare_model_for_kbit_training(model)
        model = get_peft_model(prepare_model_for_kbit_training(model), adapter_config, adapter_name=self.adapters[0])
        # for name in self.adapters:
            # model.add_adapter(adapter_name=name, adapter_config=adapter_config)
        model.add_adapter(self.adapters[1], adapter_config)
        # model.enable_adapters()
        model.set_adapter(self.adapters[0])
        logger.debug(f"{model.peft_config}")
        trainable_params, all_param = model.get_nb_trainable_parameters()
        logger.debug(f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}")
        return model
    
    def training_step(self, model: Module, inputs: Dict[str, Tensor | Any]) -> Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps
    
    def compute_loss(self, model, inputs, return_outputs=False):
        if self.control.should_evaluate:
            return self.compute_eval_loss(model, inputs, return_outputs)
        train_state = self.control.train_state
        
        model.set_adapter(self.adapters[0])  # Teacher

        if train_state == "sft":
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        elif train_state == "kd":
            # logger.debug("Start KD!")
            tmpt = self.kd_args.kd_temperature
            kd_ratio = self.kd_args.kd_ratio
            loss_mask = torch.where(((inputs['labels'] < 0) | (inputs['labels'] == self.tokenizer.pad_token_id)), 0, 1).unsqueeze(-1)

            with torch.no_grad():
                outputs_t = model(**inputs)
            logits_t = outputs_t['logits']
            # logger.debug("Teacher logits computed.")

            model.set_adapter(self.adapters[1])  # Student
            outputs = model(**inputs)
            loss_gt = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
            # logger.debug("Student logits computed.")

            # Compute KD Loss
            input, target = outputs["logits"], logits_t
            if self.kd_args.reverse_kld:
                input, target = target, input
            
            input = F.log_softmax(input/tmpt, dim=-1)  # , dtype=torch.float32)
            target = F.softmax(target/tmpt, dim=-1)  # , dtype=torch.float32)

            loss_kd = kld_loss(input, target, loss_mask, reduction="mean") * tmpt**2

            loss = (1 - kd_ratio) * loss_gt + kd_ratio * loss_kd
        else:
            raise ValueError(f"Non-recognized training state {train_state}")
        
        # Loss dict
        self.loss_dict["loss"] = round(loss.mean().item(), 4)
        if train_state == "kd":
            self.loss_dict["loss_gt"] = round(loss_gt.mean().item(), 4)
            self.loss_dict["loss_kd"] = round(loss_kd.mean().item(), 4)

        return (loss, outputs) if return_outputs else loss
    

    def compute_eval_loss(self, model, inputs, return_outputs=False):
        if self.control.train_state == "sft":
            adapter = self.adapters[0]
        else:
            adapter = self.adapters[1]

        # adapter = self.adapters[1] if not adapter else adapter
        model.set_adapter(adapter)
        
        outputs = model(**inputs)
        
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]


        # logger.debug(f"Evaluation outputs: {outputs}")
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        logger.debug(f"At evaluation, outputs: {outputs}")
        # loss = outputs.loss
        self.loss_dict = dict(
            loss=round(loss.mean().item(), 4)
        )
        return (loss, outputs) if return_outputs else loss
    

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        self.model.set_adapter(self.adapters[1])
        super()._save(output_dir, state_dict)

        self.model.set_adapter(self.adapters[0])
        super()._save(os.path.join(output_dir, "adapter-t"), state_dict)