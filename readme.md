# 从零创建生成大模型
预训练A800*8，每张卡显存占用70G+


**train tokenizer** 
```shell
python model_init/tokenization/train_eval_tokenizer.py
```
**init model** 
```shell
python model_init/model_init.py
```
**pretrain**
```shell
bash pretrain.sh
```
**finetune**
```shell
bash sft.sh
```

