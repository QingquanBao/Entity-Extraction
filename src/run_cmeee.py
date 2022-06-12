
import json
from lib2to3.pgen2.token import NOTEQUAL
from os.path import join
from typing import List

from sklearn.metrics import precision_recall_fscore_support
from transformers import AdamW, get_scheduler
from transformers.models.bert.modeling_bert import BertEmbeddings, BertAttention
from transformers import set_seed, BertTokenizer, Trainer, HfArgumentParser, TrainingArguments, BertLayer
from torch.optim.swa_utils import AveragedModel, SWALR

from args import ModelConstructArgs, CBLUEDataArgs
from logger import get_logger
from ee_data import EE_label2id2, EEDataset, EE_NUM_LABELS1, EE_NUM_LABELS2, EE_NUM_LABELS, CollateFnForEE, \
    EE_label2id1, NER_PAD, EE_label2id
from model import BertForCRFHeadNER, BertForLinearHeadNER, BertForLinearHeadNestedNER, BertForCRFHeadNestedNER, CRFClassifier, LinearClassifier
from metrics import ComputeMetricsForNER, ComputeMetricsForNestedNER, extract_entities
from torch.nn import LSTM

MODEL_CLASS = {
    'linear': BertForLinearHeadNER, 
    'linear_nested': BertForLinearHeadNestedNER,
    'crf': BertForCRFHeadNER,
    'crf_nested': BertForCRFHeadNestedNER
}

def get_logger_and_args(logger_name: str, _args: List[str] = None):
    parser = HfArgumentParser([TrainingArguments, ModelConstructArgs, CBLUEDataArgs])
    train_args, model_args, data_args = parser.parse_args_into_dataclasses(_args)

    # ===== Get logger =====
    logger = get_logger(logger_name, exp_dir=train_args.logging_dir, rank=train_args.local_rank)
    for _log_name, _logger in logger.manager.loggerDict.items():
        # 在4.6.0版本的transformers中无效
        if _log_name.startswith("transformers.trainer"):
            # Redirect other loggers' output
            _logger.addHandler(logger.handlers[0])

    logger.info(f"==== Train Arguments ==== {train_args.to_json_string()}")
    logger.info(f"==== Model Arguments ==== {model_args.to_json_string()}")
    logger.info(f"==== Data Arguments ==== {data_args.to_json_string()}")

    return logger, train_args, model_args, data_args


def get_model_with_tokenizer(model_args):
    model_class = MODEL_CLASS[model_args.head_type]

    if 'nested' not in model_args.head_type:
        model = model_class.from_pretrained(model_args.model_path, num_labels1=EE_NUM_LABELS)
    else:
        model = model_class.from_pretrained(model_args.model_path, num_labels1=EE_NUM_LABELS1, num_labels2=EE_NUM_LABELS2)
    
    tokenizer = BertTokenizer.from_pretrained(model_args.model_path)
    return model, tokenizer


def generate_testing_results(train_args, logger, predictions, test_dataset, for_nested_ner=False):
    assert len(predictions) == len(test_dataset.examples), \
        f"Length mismatch: predictions({len(predictions)}), test examples({len(test_dataset.examples)})"

    if not for_nested_ner:
        pred_entities1 = extract_entities(predictions[:, 1:], for_nested_ner=False)
        pred_entities2 = [[]] * len(pred_entities1)
    else:
        pred_entities1 = extract_entities(predictions[:, 1:, 0], for_nested_ner=True, first_labels=True)
        pred_entities2 = extract_entities(predictions[:, 1:, 1], for_nested_ner=True, first_labels=False)

    final_answer = []

    for p1, p2, example in zip(pred_entities1, pred_entities2, test_dataset.examples):
        text = example.text
        entities = []
        for start_idx, end_idx, entity_type in p1 + p2:
            entities.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "type": entity_type,
                "entity": text[start_idx: end_idx + 1],
            })
        final_answer.append({"text": text, "entities": entities})

    with open(join(train_args.output_dir, "CMeEE_test.json"), "w", encoding="utf8") as f:
        json.dump(final_answer, f, indent=2, ensure_ascii=False)
        logger.info(f"`CMeEE_test.json` saved")
        
def AdamW_grouped_LLRD(model,init_lr):
    ## ref: https://zhuanlan.zhihu.com/p/412889866
    ## Base in the source ref is roberta
    
    opt_parameters = []       # To be passed to the optimizer (only parameters of the layers you want to update).
    named_parameters = list(model.named_parameters()) 

    # According to AAAMLP book by A. Thakur, we generally do not use any decay 
    # for bias and LayerNorm.weight layers.
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    set_2 = ["layer.4", "layer.5", "layer.6", "layer.7"]
    set_3 = ["layer.8", "layer.9", "layer.10", "layer.11"]
    # init_lr = 1e-6

    for i, (name, params) in enumerate(named_parameters):  
        print(name)
        weight_decay = 0.0 if any(p in name for p in no_decay) else 0.01

        if name.startswith("bert.embeddings") or name.startswith("bert.encoder"):            
            # For first set, set lr to 1e-6 (i.e. 0.000001)
            lr = init_lr       

            # For set_2, increase lr to 0.00000175
            lr = init_lr * 1.75 if any(p in name for p in set_2) else lr

            # For set_3, increase lr to 0.0000035 
            lr = init_lr * 3.5 if any(p in name for p in set_3) else lr

            opt_parameters.append({"params": params,
                                   "weight_decay": weight_decay,
                                   "lr": lr})  

        # For regressor and pooler, set lr to 0.0000036 (slightly higher than the top layer).                
        if name.startswith("regressor") or name.startswith("bert.pooler"):               
            lr = init_lr * 3.6 

            opt_parameters.append({"params": params,
                                   "weight_decay": weight_decay,
                                   "lr": lr})    

    return AdamW(opt_parameters, lr=init_lr)

def main(_args: List[str] = None):
    # ===== Parse arguments =====
    logger, train_args, model_args, data_args = get_logger_and_args(__name__, _args)

    # ===== Set random seed =====
    set_seed(train_args.seed)

    # ===== Get models =====
    model, tokenizer = get_model_with_tokenizer(model_args)
    for_nested_ner = 'nested' in model_args.head_type

    # ===== Get datasets =====
    if train_args.do_train:
        train_dataset = EEDataset(data_args.cblue_root, "train", data_args.max_length, tokenizer, for_nested_ner=for_nested_ner)
        dev_dataset = EEDataset(data_args.cblue_root, "dev", data_args.max_length, tokenizer, for_nested_ner=for_nested_ner)
        logger.info(f"Trainset: {len(train_dataset)} samples")
        logger.info(f"Devset: {len(dev_dataset)} samples")
    else:
        train_dataset = dev_dataset = None

    # ===== Trainer =====
    compute_metrics = ComputeMetricsForNestedNER() if for_nested_ner else ComputeMetricsForNER()
    
    if model_args.lr_decay:
        ## ref: https://zhuanlan.zhihu.com/p/412889866, decay & warm_up
        optimizer = AdamW_grouped_LLRD(model,train_args.learning_rate)
        scheduler = get_scheduler(
                    "cosine",
                    optimizer = optimizer,
                    num_warmup_steps = 50,
                    num_training_steps = int(1e6)
        )
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=train_args,
            data_collator=CollateFnForEE(tokenizer.pad_token_id, for_nested_ner=for_nested_ner),
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=compute_metrics,
            optimizers = (optimizer,scheduler)
        )
    else:
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=train_args,
            data_collator=CollateFnForEE(tokenizer.pad_token_id, for_nested_ner=for_nested_ner),
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=compute_metrics,
        )

    if train_args.do_train:
        try:
            trainer.train()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt")

    if train_args.do_predict:
        test_dataset = EEDataset(data_args.cblue_root, "test", data_args.max_length, tokenizer, for_nested_ner=for_nested_ner)
        logger.info(f"Testset: {len(test_dataset)} samples")

        # np.ndarray, None, None
        predictions, _labels, _metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
        generate_testing_results(train_args, logger, predictions, test_dataset, for_nested_ner=for_nested_ner)


if __name__ == '__main__':
    main()
