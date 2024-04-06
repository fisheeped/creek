#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    : train_tokenizer.py
@Time    : 2024/04/01 16:10:03
@Author  : YuYang
@Contact : fisheepman@gmail.com
@License : Apache License Version 2.0
@Describe: 
@refer   : https://github.com/zhaibowen/Retriever/blob/main/tokenization_v2.py
'''


import os
import json
import torch
import tokenizers
from tokenizers import (
    models,
    normalizers,
    pre_tokenizers,
    decoders, 
    Regex
)
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast
import datasets


import logging
import sys


logging.basicConfig(
    format='[%(asctime)s %(pathname)s:%(lineno)s %(levelno)s]\t%(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def bpe_tokenization(model_root,data_path):
    tmp_tk = os.path.join(model_root, "tmp_tokenizer.json")
    final_tk = os.path.join(model_root, "tokenizer.json")

    trainer = BpeTrainer(
        special_tokens=["<unk>", "<s>", "</s>"], 
        vocab_size=32000-256,
        limit_alphabet=4000,
        )
    # special_tokens_map = {}
    # for idx,special_token in enumerate(trainer.special_tokens):
    #     special_tokens_map[special_token.content] = idx
    tokenizer = tokenizers.Tokenizer(
        models.BPE(
            unk_token="<unk>",
            fuse_unk=True,
            byte_fallback=True
        ))
    tokenizer.add_special_tokens(trainer.special_tokens)
    tokenizer.normalizer = normalizers.Sequence([
        # normalizers.NFKC(),
        normalizers.Prepend('▁'), # 句首加▁,挺重要的
        normalizers.Replace(' ', '▁')
        ]) 
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex('▁'), behavior='merged_with_next'),
        # pre_tokenizers.Split(Regex('\d|[\u2E80-\u2FDF\u3040-\u318F\u31A0-\u31BF\u31F0-\u31FF\u3400-\u4DB5\u4E00-\u9FFF\uA960-\uA97F\uAC00-\uD7FF]'), behavior='isolated'),
        pre_tokenizers.Split(Regex('▁*(\w+|[^\w\s]+)'), behavior='isolated')
        ])
    tokenizer.decoder = decoders.Sequence([
        decoders.Replace('▁', ' '),
        decoders.ByteFallback(),
        decoders.Fuse(),
        decoders.Strip(" ", 1, 0)
        ])
    
    # 数据流处理
    data = datasets.load_dataset(data_path,trust_remote_code=True,split=datasets.Split.TRAIN,num_proc=96)
    def batch_text(ds,batch_size=1000):
        # 优化内存占用，防止一次性读取大文件
        for i in range(0, len(ds), batch_size):
            yield ds[i : i + batch_size]["text"]
    iter_ds = batch_text(data)

    tokenizer.train_from_iterator(iterator = iter_ds, trainer=trainer,length=len(data))
    tokenizer.save(tmp_tk)

    with open(tmp_tk, 'r') as f:
        x = json.load(f)
        new_vocab = {}
        new_vocab['<unk>'] = 0
        new_vocab['<s>'] = 1
        new_vocab['</s>'] = 2

        for i in range(256):
            hexn = hex(i)[2:].upper()
            s = f"<0x{hexn:>02s}>"
            new_vocab[s] = i + 3

        for k, v in x['model']['vocab'].items():
            if k not in ['<unk>', '<s>', '</s>']:
                new_vocab[k] = v + 256
        x['model']['vocab'] = new_vocab

    with open(final_tk, 'w') as f:
        json.dump(x, f, indent=2, ensure_ascii=False)
    
    tokenizer = tokenizers.Tokenizer.from_file(final_tk)
    output = tokenizer.encode("Hello. y'all! Nice to meet to you 😁\n您好，我是小弟 Ã?\nwhat is a nice today ")
    logger.info(f"encode sentence:{output.tokens}")
    output = tokenizer.decode(output.ids)
    logger.info(f"decode token:{output}")

def bpe_validate(model_root, data_path):
    final_tk = os.path.join(model_root, "tokenizer.json")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=final_tk)

    query = "Hello, y'all! How are you 😁\nwhat is the weather like today?\nThis year is 2023"
    query2 = "what is your name"
    output = tokenizer([query, query2], max_length=10, return_overflowing_tokens=True, stride=0, truncation=False)['input_ids']
    # print(output.tokens)
    # output = tokenizer.decode(output.ids)
    logger.info(f"{output}")

    data = datasets.load_dataset(data_path,trust_remote_code=True,split=datasets.Split.TRAIN)
    def map_tokenizer(example):
        example['len_tk'] = len(tokenizer.encode(example['text']))
        example['len_text'] = len(example['text'])
        return example
    data = data.select(range(10000)).map(map_tokenizer)
    sum_tokens = sum(data['len_tk'])
    sum_texts = sum(data['len_text'])
    char_to_tokens = sum_tokens / sum_texts
    logger.info(f"char_to_tokens  is {char_to_tokens}")

if __name__ == "__main__":
   data_path = "/ceph2/yuyang06/note/creek/data/all_jsonl"
   model_root = "/ceph2/yuyang06/note/creek/model_init"

   bpe_tokenization(model_root,data_path)
#    bpe_validate(model_root,data_path)