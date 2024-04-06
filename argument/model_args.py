#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    : model_arg.py
@Time    : 2024/01/19 09:54:46
@Author  : YuYang
@Contact : fisheepman@gmail.com
@License : Apache License Version 2.0
@Describe: for model_args like model_path
@refer   : https://github.com/huggingface/trl/blob/main/examples/research_projects/toxicity/scripts/gpt-j-6b-toxicity.py
'''

from typing import Literal, NewType, Any
from dataclasses import dataclass,field

from .baseargs import BaseArgs

DataClass = NewType("DataClass", Any)

# current sup type
torch_dtype = Literal['float16','bfloat16','float']



# TODO 待优化继承，但是dataclass继承不太稳定
@dataclass
class ModelArgs(BaseArgs):
    model_type: torch_dtype = field(default="bfloat16")
    # save_per_epochs: int = 2
    model_path: str =  "/ceph2/yuyang06/note/myllama/creek"
    use_fast_tokenizer: bool = True
    # output_path: str = field(default="./output/ppotest",metadata={"help":"path to output"})
    # global_epochs: int = 3
    # device: str = "cuda:0"
