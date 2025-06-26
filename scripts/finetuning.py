from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
from seqeval.metrics import classification_report

# Step 1: Load and Parse CoNLL
def read_conll_file(filepath):
    sentences = []
    tokens, labels = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                token, label = line.strip().split()
                tokens.append(token)
                labels.append(label)
            else:
                sentences.append({'tokens': tokens, 'ner_tags': labels})
                tokens, labels = [], []
    return sentences

data = read_conll_file("C:\\Users\\Rahel\\Desktop\\KAIM5\\Week4\\labels\\labeled_data.conll")

# Step 2: Label mapping
label_list = sorted(list({label for s in data for label in s['ner_tags']}))
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}

# Step 3: Tokenize
model_checkpoint = "Davlan/xlm-roberta-base-ner-hrl"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []

    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            else:
                label_ids.append(label_to_id[label[word_idx]] if label[word_idx].startswith("I-") else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Step 4: Dataset preparation
dataset = Dataset.from_list(data).train_test_split(test_size=0.2)
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

# Step 5: Load model
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

# Step 6: Training args
args = TrainingArguments(
    output_dir="amharic-ner-model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Step 7: Trainer setup
data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Step 8: Train the model
trainer.train()

# Step 9: Save
trainer.save_model("amharic-ner-model")
tokenizer.save_pretrained("amharic-ner-model")

print("Model fine-tuning complete and saved.")
