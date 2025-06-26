import pandas as pd

# Load preprocessed data
df = pd.read_csv("C:\\Users\\Rahel\\Desktop\\KAIM5\\Week4\\EthioMart-Amharic-NER\\data\\preprocessed_data.csv")

# Sample 50 messages
subset = df[['clean_text']].dropna().sample(n=50, random_state=42)

# Save sample for labeling
subset.to_csv("c:\\Users\\Rahel\\Desktop\\KAIM5\\Week4\\EthioMart-Amharic-NER\\data\\sample_for_labeling.csv", index=False)

print("50 messages saved for labeling.")
