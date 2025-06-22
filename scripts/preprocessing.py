import pandas as pd
df = pd.read_csv('C:\\Users\\Rahel\\Desktop\\KAIM 5-6\\Week 4\\EthioMart-Amharic-NER\\data\\raw\\telegram_data.csv')
df['Text'] = df['Text'].str.replace(r'http\S+', '', regex=True)  # Remove URLs