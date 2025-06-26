from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from datasets import Dataset
import numpy as np
from seqeval.metrics import classification_report

# Define labels exactly as in your training
LABEL_LIST = ["O", "B-LOC", "B-PRICE", "B-Product", "I-LOC", "I-PRICE", "I-Product"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

# Paths
MODEL_PATH = "amharic-ner-model"
LABEL_PATH = "C:\\Users\\Rahel\\Desktop\\KAIM5\\Week4\\labels\\labeled_data.conll"

def load_conll_data(filepath):
    sentences = []
    tokens, tags = [], []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tok, label = line.strip().split()
                tokens.append(tok)
                tags.append(label)
            else:
                if tokens:
                    sentences.append({"tokens": tokens, "ner_tags": tags})
                    tokens, tags = [], []
    # Catch last sentence if no trailing newline
    if tokens:
        sentences.append({"tokens": tokens, "ner_tags": tags})
    return sentences

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True,
        padding=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens get -100
            elif word_idx != previous_word_idx:
                label_ids.append(LABEL_TO_ID[label[word_idx]])
            else:
                # For subword tokens inside the same word, label with I- if applicable else -100
                current_label = label[word_idx]
                if current_label.startswith("I-"):
                    label_ids.append(LABEL_TO_ID[current_label])
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def main():
    # Load labeled data
    data = load_conll_data(LABEL_PATH)

    # Split dataset to train/test or just use all for evaluation (you can split if you want)
    dataset = Dataset.from_list(data)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

    # Tokenize and align labels
    tokenized_dataset = dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)

    # Setup Trainer for evaluation only
    trainer = Trainer(model=model, tokenizer=tokenizer)

    # Predict
    predictions_output = trainer.predict(tokenized_dataset)
    preds = np.argmax(predictions_output.predictions, axis=2)
    labels = predictions_output.label_ids

    # Convert predictions and labels to label names for seqeval
    true_preds = []
    true_labels = []
    for pred, label in zip(preds, labels):
        true_pred = []
        true_label = []
        for p, l in zip(pred, label):
            if l != -100:
                true_label.append(ID_TO_LABEL[l])
                true_pred.append(ID_TO_LABEL[p])
        true_preds.append(true_pred)
        true_labels.append(true_label)

    # Print classification report
    print("Named Entity Recognition Evaluation Report:")
    print(classification_report(true_labels, true_preds))

if __name__ == "__main__":
    main()
