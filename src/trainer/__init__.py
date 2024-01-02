from .kd_trainer import Seq2SeqKDTrainer, Seq2SeqLDKDTrainer, Seq2SeqDataFreeKDTrainer
from .kd_arguments import Seq2SeqKDArguments, Seq2SeqLDKDArguments, Seq2SeqDataFreeKDArguments
from .kd_trainer import KDLoggingCallback

KD_TRAINERS_DICT = dict(
    kd = Seq2SeqKDTrainer,
    ldkd = Seq2SeqLDKDTrainer,
    dfkd = Seq2SeqDataFreeKDTrainer
)

KD_ARGS_DICT = dict(
    kd = Seq2SeqKDArguments,
    ldkd = Seq2SeqLDKDArguments,
    dfkd = Seq2SeqDataFreeKDArguments
)