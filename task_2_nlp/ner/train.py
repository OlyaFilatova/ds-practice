"""Train a NER model to identify animal names in text."""

import json
import logging
from pathlib import Path

from datasets import Dataset
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")

BASE_DIR = Path(__file__).resolve().parent
train_file = BASE_DIR / "../data/ner/train.json"
val_file = BASE_DIR / "../data/ner/val.json"

# -----------------------------
# Load dataset
# -----------------------------
with open(train_file, encoding="utf-8") as f:
    train_data = json.load(f)
with open(val_file, encoding="utf-8") as f:
    val_data = json.load(f)

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# -----------------------------
# Labels
# -----------------------------
label_list = ["O", "B-ANIMAL"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

# -----------------------------
# Tokenizer
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")


# -----------------------------
# Tokenize & align labels
# -----------------------------
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=128,
    )

    labels = []
    for i, word_ids in enumerate(
        map(
            tokenized_inputs.word_ids,
            range(len(tokenized_inputs["input_ids"])),
        )
    ):
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                # Label all subwords with the same entity
                label_ids.append(label2id[examples["ner_tags"][i][word_idx]])
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)

# -----------------------------
# Model
# -----------------------------
model = AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-cased",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)


# -----------------------------
# Compute metrics for entity-level F1
# -----------------------------
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids
    preds_list, labels_list = [], []

    for label, p_ in zip(labels, preds):
        l_new, p_new = [], []
        for ll, pp in zip(label, p_):
            if ll != -100:
                l_new.append(id2label[ll])
                p_new.append(id2label[pp])
        labels_list.append(l_new)
        preds_list.append(p_new)

    return {
        "precision": precision_score(labels_list, preds_list),
        "recall": recall_score(labels_list, preds_list),
        "f1": f1_score(labels_list, preds_list),
    }


# -----------------------------
# Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir=str(BASE_DIR / "../models/ner"),
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,  # Early stopping based on eval F1
    metric_for_best_model="f1",
    greater_is_better=True,
)

# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# -----------------------------
# Train & Save
# -----------------------------
trainer.train()
trainer.save_model(BASE_DIR / "../models/ner")
