from functools import partial
from itertools import chain
import logging
import math
import os
import sys
import datasets
from argument import parser
from torchdata.datapipes.iter import IterableWrapper
import transformers
import evaluate
import glob


IGNORE_INDEX = -100




def main():
    model_args,data_args,train_args = parser()
    logging.basicConfig(
        format='[%(asctime)s %(pathname)s:%(lineno)s %(levelno)s]\t%(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if train_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = train_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_path, 
        trust_remote_code=True,
        revision = "main",
        use_fast =  model_args.use_fast_tokenizer,
        padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_path, 
        revision = "main",
        trust_remote_code=True)

    def dsMapFunction(
            examples, 
            tokenizer, 
            block_size:int = 1024):
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        
        messages = [
            {'role':'user','content':""},
            {'role':'assistant','content':""},
            ]
        label_idx = []
        for idx in range(len(examples['instruction'])):
            messages[0]['content'] = examples['instruction'][idx]
            messages[1]['content'] = examples['output'][idx]
            tokens = tokenizer.apply_chat_template(
                messages,
                return_dict = True,
                max_length = block_size,
                truncation = True)
            model_inputs['input_ids'].append(tokens['input_ids'])
            model_inputs['attention_mask'].append(tokens['attention_mask'])
            len_source = len(tokenizer.encode(examples['instruction'][idx], add_special_tokens=False)) + 6
            len_label = len(tokenizer.encode(examples['output'][idx], add_special_tokens=False)) + 2
            label_idx.append([len_source,len_label])
        for idx, lidx in enumerate(label_idx):
            encode_len = len(model_inputs["input_ids"][idx])
            len_source, len_label = lidx
            label = [IGNORE_INDEX] * encode_len
            if len_source + len_label < block_size:
                label[len_source:len_source + len_label] = model_inputs["input_ids"][idx][len_source:len_source + len_label] # 除了answer其它全部遮住
            elif len_source - 4 < block_size: # 不减6是因为前面有bos + white space
                model_inputs["input_ids"][idx][len_source-4]  = tokenizer.eos_token_id  # 不加空格，麻烦。-4是第一位answer token 需要转eos
                label[:len_source-3] = model_inputs["input_ids"][idx][:len_source-3] # 把刚刚加的eos加进去,当作pretrain
            else:
                label = model_inputs["input_ids"][idx] # 不加eos了，因为预训练的时候也没加，对齐一下
            model_inputs["labels"].append(label.copy())

        return model_inputs

    cache_path = os.path.join(data_args.data_path,".cache")
    
    with train_args.main_process_first(desc="dataset map tokenization"):
        if data_args.cache and os.path.exists(cache_path):
            ds = datasets.load_from_disk(cache_path)
        else:
            ds = datasets.load_dataset(data_args.data_path,trust_remote_code=True,split=datasets.Split.TRAIN,num_proc= data_args.num_proc)
            ds = ds.shuffle()
            column_names = ds.column_names
            
            dsmap = partial(
                dsMapFunction, 
                tokenizer=tokenizer, 
                block_size = data_args.block_size)

            logger.info(f"{ds[0]}")
            ds = ds.map(
                dsmap, 
                batched=True, 
                batch_size=56 * 8, # 和训练的size有关
                remove_columns=column_names, 
                num_proc = data_args.num_proc,
                desc = f"tokenizer dataset with {data_args.block_size}")
            logger.info(f"{ds[0]}")
            logger.info(f"dataset cache path:{cache_path}")
            ds = ds.train_test_split(test_size=0.01,train_size=0.99)
            ds.save_to_disk(cache_path)

    if train_args.do_train:
        train_dataset = ds['train']
        if data_args.max_num_data:
            train_dataset = train_dataset.select(range(data_args.max_num_data))
    compute_metrics = None
    preprocess_logits_for_metrics = None
    if train_args.do_eval:
        eval_dataset = ds['test']
        def preprocess_logits_for_metrics(logits, labels):
                if isinstance(logits, tuple):
                    logits = logits[0]
                return logits.argmax(dim=-1)
        metric = evaluate.load("accuracy.py")
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)
    if train_args.do_train: 
        data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            pad_to_multiple_of=8 ,  # for shift short attention
            label_pad_token_id=IGNORE_INDEX,
            padding=True
        )   
        trainer = transformers.Trainer(
            model=model,
            args=train_args,
            train_dataset= IterableWrapper(train_dataset) if train_args.do_train else None,
            eval_dataset= IterableWrapper(eval_dataset) if train_args.do_eval else None,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it.
            data_collator = data_collator,
            compute_metrics = compute_metrics ,
            preprocess_logits_for_metrics = preprocess_logits_for_metrics
        )
        

    if train_args.do_train:
        logger.info(f"local_rank:{train_args.local_rank} start train")
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


    if train_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()