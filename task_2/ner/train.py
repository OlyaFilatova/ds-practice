"""Train a NER model to identify animal names in text."""
import json
import logging
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

BASE_DIR = Path(__file__).resolve().parent
with open(BASE_DIR / "../data/ner/train.json", encoding="utf-8") as f:
    train_data = json.load(f)
with open(BASE_DIR / "../data/ner/val.json", encoding="utf-8") as f:
    val_data = json.load(f)

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

label_list = ["O", "B-ANIMAL"]
label2id = {l:i for i,l in enumerate(label_list)}
id2label = {i:l for l,i in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def log_as_json(message, **kwargs):
    """Log messages in JSON format."""
    logging.info(json.dumps({"message": message, **kwargs}))

def tokenize_and_align_labels(examples):
    """Tokenize input texts and align NER labels with tokenized outputs."""
    log_as_json("Tokenizing and aligning labels", num_examples=len(examples["tokens"]))
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=128
    )

    labels = []
    for i in range(len(examples["tokens"])):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_ids.append(label2id[examples["ner_tags"][i][word_idx]])
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs



train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

training_args = TrainingArguments(
    output_dir=BASE_DIR / "../models/ner",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

trainer.train()
trainer.save_model(BASE_DIR / "../models/ner")
