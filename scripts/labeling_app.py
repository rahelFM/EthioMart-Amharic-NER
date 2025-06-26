import streamlit as st 
import pandas as pd
import os

# Load your sample data
df = pd.read_csv("C:\\Users\\Rahel\\Desktop\\KAIM5\\Week4\\EthioMart-Amharic-NER\\data\\sample_for_labeling.csv") 

# Entity tag options
entity_tags = ["O", "B-Product", "I-Product", "B-LOC", "I-LOC", "B-PRICE", "I-PRICE"]

st.title("Amharic NER Labeling Tool")

# Initialize labeled data storage
labeled_data = []

# Loop through each message
for idx, row in df.iterrows():
    st.subheader(f"Message {idx + 1}")
    tokens = str(row['clean_text']).split()
    sentence = []

    for i, token in enumerate(tokens):
        label = st.selectbox(
            f"Label for '{token}'", entity_tags, key=f"{idx}-{i}-{token}"
        )
        sentence.append((token, label))

    labeled_data.append(sentence)
    st.markdown("---")

# Save button in the sidebar
with st.sidebar:
    if st.button("Save CoNLL File"):
        output_path = "labels/labeled_data.conll"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for sentence in labeled_data:
                for token, label in sentence:
                    f.write(f"{token} {label}\n")
                f.write("\n")

        st.success(f"File saved at: {output_path}")
