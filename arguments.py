from dataclasses import dataclass, field
from typing import Optional
from src.trainer import Seq2SeqKDArguments, Seq2SeqLDKDArguments

KD_TYPE_DICT = dict(
    kd = Seq2SeqKDArguments,
    ldkd = Seq2SeqLDKDArguments
)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    reference: https://github.com/huggingface/transformers/blob/235e5d4991e8a0984aa78db91087b49622c7740e/examples/pytorch/language-modeling/run_clm.py#L71
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )

    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "Resume training from specified checkpoint."
        }
    )
    # def __post_init__(self):
    #     if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
    #         raise ValueError(
    #             "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
    #         )


@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(
        default=None, metadata={
            "help": "The name of the dataset to use (via the src.dataset).",
            "choice": ["dolly","samsum","gmk8k"]
        }
    )

@dataclass
class KDArguments:
    type: str = field(
        default=None, metadata={
            "help": "Type of the KD trainer.",
            "choice": list(KD_TYPE_DICT.keys())
        }
    )
    args: dict = field(
        default=None, metadata={"help": "Key-value pairs of the KD arguments."}
    )
    
    def __post_init__(self):
        if self.type == None:
            return
        self.type = self.type.lower()
        assert self.type in KD_TYPE_DICT.keys(), f"Unknown KD trainer type {self.type}."
        self.args = KD_TYPE_DICT[self.type].parse_dict(self.args)