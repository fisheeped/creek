#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    : model.py
@Time    : 2023/11/24 17:20:25
@Author  : YuYang
@Contact : fisheepman@gmail.com
@License : Apache License Version 2.0
'''

import logging
import sys


logging.basicConfig(
    format='[%(asctime)s %(pathname)s:%(lineno)s %(levelno)s]\t%(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import torch
from tokenizer_creek import CreekTokenizerFast
from model_creek import creekForCausalLM
from configuration_creek import creekConfig,creekGenerationConfig
from transformers import TextStreamer
import os
# os.environ['CUDA_VISIBLE_DEVICES']='2'



# 很重要的包 TextIteratorStreamer
# from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

def init_creek(zoom_size:float = 0.25):
    device = 'cuda'
    tokenizer = CreekTokenizerFast(tokenizer_file="/ceph2/yuyang06/note/myllama/model/tokenizer.json",legacy = False)
    tokenizer.register_for_auto_class()

    vocab_size = len(tokenizer)
    creekconfig = creekConfig(
            vocab_size = vocab_size,
            hidden_size = int(4096 * zoom_size),
            intermediate_size = int(11008 * zoom_size),
            num_hidden_layers = int(32 * zoom_size),
            num_attention_heads = int(32 * zoom_size),
            max_position_embeddings = int(2048),
        )
    creekconfig.register_for_auto_class("AutoConfig")
    genConfig = creekGenerationConfig()
    model = creekForCausalLM(creekconfig).to(device).bfloat16()
    model.generation_config = genConfig
    model.register_for_auto_class("AutoModelForCausalLM") # 注册至自动类到config中
    streamer = TextStreamer(tokenizer)
    # input_ids = torch.randint(low = 0, high = vocab_size, size = (4, 32))
    # res =  model(input_ids=input_ids)
    qes = "who are u?"
    logger.info(f"question:{qes}")
    data = tokenizer(qes, return_tensors = "pt").to(device)
    res = model.generate(**data,streamer=streamer) # res.shape bsz*out_seq_len
    response = tokenizer.decode(res[0])
    logger.info(f"response:{response}")
    # 参数量
    tmp_data = model.num_parameters()
    logger.info(f"模型参数量{tmp_data}") # 1074944000
    # 最大占用 理论最优显存分块策略可达
    tmp_data = torch.cuda.max_memory_allocated()/1023**3
    logger.info(f"最大占用G:{tmp_data}") # 2.203125
    # 最大预留
    tmp_data = torch.cuda.max_memory_reserved()/1023**3
    logger.info(f"最大预留G:{tmp_data}") # 2.209592
    tokenizer.save_pretrained("/ceph2/yuyang06/note/myllama/creek")
    model.save_pretrained("/ceph2/yuyang06/note/myllama/creek")
    logger.info("ok!")    

if __name__ == "__main__":
    init_creek()
    