import pandas as pd
import re

# Load data
df = pd.read_csv("c:\\Users\\Rahel\\Desktop\\KAIM5\\Week4\\EthioMart-Amharic-NER\\data\\raw\\telegram_data.csv")

# Define Amharic text cleaner
def clean_amharic_text(text):
    text = str(text)
    text = re.sub(r"[A-Za-z0-9]", " ", text) 
    text = re.sub(r"[^\u1200-\u137F\s]", " ", text) 
    text = re.sub(r"\s+", " ", text).strip()  
    return text

# Apply cleaning using correct column name
df['clean_text'] = df['Text'].apply(clean_amharic_text)

# Save cleaned version
df.to_csv("c:\\Users\\Rahel\\Desktop\\KAIM5\\Week4\\EthioMart-Amharic-NER\\data\\preprocessed_data.csv", index=False)

print("Preprocessing completed successfully.")
