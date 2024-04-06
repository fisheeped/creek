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
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
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
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_path, 
        revision = "main",
        trust_remote_code=True)

    def dsMapFunction(
            examples, 
            tokenizer, 
            block_size:int = 1024,
            data_col:str = "text"):
        eos_text = [tokenizer.bos_token + ' ' + text + ' ' + tokenizer.eos_token for text in examples[data_col]]
        tokenized_examples = tokenizer(eos_text, add_special_tokens=False)
        concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
        total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    cache_path = os.path.join(data_args.data_path,".cache")
    with train_args.main_process_first(desc="dataset map tokenization"):
        if data_args.cache and os.path.exists(cache_path):
            ds = datasets.load_from_disk(cache_path)
        else:
            ds = datasets.load_dataset(data_args.data_path,trust_remote_code=True,split=datasets.Split.TRAIN)
            ds = ds.shuffle()
            column_names = ds.column_names
            assert data_args.text_column_name in column_names, f"data_args.text_column_name: {data_args.text_column_name} is not in dataset column_names !"
            dsmap = partial(
                dsMapFunction, 
                tokenizer=tokenizer, 
                block_size = data_args.block_size, 
                data_col = data_args.text_column_name)

            logger.info(f"{ds[0][data_args.text_column_name]}")

            ds = ds.map(
                dsmap, 
                batched=True, 
                remove_columns=column_names, 
                num_proc = data_args.num_proc,
                desc = f"tokenizer dataset with {data_args.block_size}")

            logger.info(f"{ds[0]}")
            ds = ds.train_test_split(test_size=0.01,train_size=0.99)
            logger.info(f"dataset cache path:{cache_path}")
            ds.save_to_disk(cache_path)

    if train_args.do_train:
        train_dataset = ds["train"]
        
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
        trainer = transformers.Trainer(
            model=model,
            args=train_args,
            train_dataset= IterableWrapper(train_dataset) if train_args.do_train else None,
            eval_dataset= IterableWrapper(eval_dataset) if train_args.do_eval else None,
            tokenizer=tokenizer,
            # Data collator will default to DataCollatorWithPadding, so we change it.
            data_collator = transformers.default_data_collator,
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