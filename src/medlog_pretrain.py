import math
from transformers import set_seed, BertTokenizer, Trainer, HfArgumentParser, TrainingArguments, BertLayer, BertModel, BertForMaskedLM

import json
from datasets import Dataset, load_from_disk
from transformers import DataCollatorForLanguageModeling


def tokenize_function(examples):
    result = tokenizer(examples["train"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def group_texts(examples):
    chunk_size = 128

    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

def cat_list(example):
    out_ = ''
    for d in example:
        out_ += str(d)
    return out_.replace('病人：', '').replace('医生：', '')

## Dataset def 
#filename = '../data/train_data.json'
#with open(filename, encoding="utf8") as f:
#    data = json.load(f)
#
#data1 = [ cat_list(d) for d in data]
#data2 = {'train': data1}
#dataset = Dataset.from_dict(data2)


tokenizer = BertTokenizer.from_pretrained('../bert-base-chinese')

#tokenized_datasets = dataset.map(
#    tokenize_function, batched=True,remove_columns=["train"],load_from_cache_file=True,writer_batch_size=10000
#)
#lm_datasets = tokenized_datasets.map(group_texts, batched=True,load_from_cache_file=True,writer_batch_size=10000)
#
#lm_datasets.save_to_disk('../data/lm_cache')
lm_datasets = load_from_disk('../data/lm_cache', keep_in_memory=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)


downsampled_dataset = lm_datasets.train_test_split(
    train_size=0.9, test_size=0.1, seed=42
)
print(downsampled_dataset)

batch_size = 256
# Show the training loss with every epoch
logging_steps = max(len(downsampled_dataset["train"]) // batch_size, 1)
model_name = 'bert'

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-medlog",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=True,
    fp16=True,
    logging_steps=logging_steps,
    save_steps=logging_steps // 4,
    num_train_epochs=10,
    resume_from_checkpoint='/dssg/home/acct-stu/stu907/cmeee/src/bert-finetuned-medlog/checkpoint-70000'
)

model = BertForMaskedLM.from_pretrained('/dssg/home/acct-stu/stu907/cmeee/src/bert-finetuned-medlog/checkpoint-70000')

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,
)

#eval_results = trainer.evaluate()
#print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

trainer.train(resume_from_checkpoint=True)
eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

torch.save(model.bert.state_dict(), 'my_pretrain/01.bin')