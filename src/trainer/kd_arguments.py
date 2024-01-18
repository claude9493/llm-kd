from dataclasses import dataclass, field

@dataclass
class Seq2SeqKDArguments:
    reverse_kld: bool = field(default=False, metadata={"help": "Whether to use reverse KL divergence."}) 
    kd_ratio: float = field(default=0.5, metadata={"help": "Weight for KD loss."}) 
    kd_temperature: float = field(default=1.0, metadata={"help": "Teamperature for computing KL divergence."})


@dataclass
class Seq2SeqLDKDArguments(Seq2SeqKDArguments):
    ldkd_alpha: float = field(default=1.0, metadata={"help":"Weight for top classes"})
    ldkd_beta: float = field(default=1.0, metadata={"help":"Weight for remaining classes"})
    ldkd_top_ratio: float = field(default=0.9, metadata={"help":"Top percentage."})
    ldkd_gen_perc: float = field(default=0.75)

@dataclass
class Seq2SeqDataFreeKDArguments(Seq2SeqKDArguments):
    seqs_per_seed: int = field(default=1, metadata={"help": "number of sequences to generate from each seeding word"})

@dataclass
class Seq2SeqAltKDArguments(Seq2SeqKDArguments):
    altkd_adapter_config: dict = field(
        default=None,
        metadata={
            "help": "Config for the LoRA."
        }
    )
    altkd_switch_freq: int = field(
        default = 1,
        metadata={
            "help": "Frequency to switch adapters."
        }
    )