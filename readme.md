# 从零创建生成大模型
下面代码中的参数，资源占用：预训练A800*8，每张卡显存占用70G+。
可以调小batch_size,max_length，最少单卡12G显存应该能完成下面步骤。


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

