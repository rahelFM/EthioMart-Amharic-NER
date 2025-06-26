
# # EthioMart Amharic Named Entity Recognition (NER)

This project focuses on fine-tuning a Named Entity Recognition (NER) model for the **Amharic language**, specifically targeting e-commerce-related entities such as **Product Names**, **Prices**, and **Locations** from Telegram channel messages of the fictional marketplace "EthioMart".

## Project Structure

```bash
EthioMart-Amharic-NER/
├── data/
│   ├── raw_data.csv
│   └── sample_for_labeling.csv
├── labels/
│   └── labeled_data.conll
├── scripts/
│   ├── preprocessing.py
│   ├── labeling_app.py
│   ├── finetuning.py
│   └── evaluation.py
├── amharic-ner-model/  # fine-tuned model saved here
└── README.md
✅ Tasks
✅ Task 1: Data Collection & Cleaning
Extracted 50 product-related messages from Telegram exports.

Cleaned and normalized Amharic text.

✅ Task 2: Manual Annotation
Built an interactive Streamlit labeling tool (labeling_app.py).

Annotated tokens with BIO format: B-Product, I-Product, B-LOC, B-PRICE, etc.

Saved in CoNLL format at labels/labeled_data.conll.

✅ Task 3: Fine-Tuning
Used xlm-roberta-base pretrained on multilingual NER.

Fine-tuned on labeled Amharic product data.

Script: scripts/finetuning.py
Model Info
Base Model: Davlan/xlm-roberta-base-ner-hrl

Training Epochs: 3

Labels:

B-Product, I-Product

B-LOC, I-LOC

B-PRICE, I-PRICE

O (outside entity)

Dependencies

pip install pandas streamlit transformers datasets seqeval
Running the Project
1. Label Tokens

streamlit run scripts/labeling_app.py
2. Fine-Tune Model

python scripts/finetuning.py
3. Evaluate Model

python scripts/evaluation.py
Authors
Rahel Sileshi Abdisa

License
MIT License
