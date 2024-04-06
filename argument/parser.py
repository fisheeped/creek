from transformers import HfArgumentParser,TrainingArguments
from typing import Tuple
from . import ModelArgs,DataArgs
def parser() -> Tuple[ModelArgs,DataArgs,TrainingArguments]:
    parser = HfArgumentParser((ModelArgs,DataArgs,TrainingArguments))
    a,b,c  = parser.parse_args_into_dataclasses()
    return a,b,c

