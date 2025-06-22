# EthioMart-Amharic-NER  
Telegram-based e-commerce data pipeline for Amharic NER.

##  Current Progress  
### Data Scraping  
- **Script**: `scripts/telegram_scrapper.py`  
- **Output**:  
  - `data/raw/telegram_data.csv` (structured messages)  
  - `data/telegram_media/` (downloaded images)  

### Preprocessing  
1. **URL Removal**:  
   ```python
   df['Text'] = df['Text'].str.replace(r'http\S+', '', regex=True)
Data Validation:

Run script/preprocessing.ipynb to check for missing values.

 Files Submitted
text
EthioMart-Amharic-NER/  
├── data/  
│   ├── raw/telegram_data.csv          # Raw scraped data  
│   └── telegram_media/                # Product images  
├── scripts/  
│   ├── telegram_scrapper.py           # Scraper code  
│   └── preprocessing.py                  # URL cleaning script  
└── README.md  
🚀 Next Steps
Label data using Label Studio.

Fine-tune afro-xlmr-base for Amharic NER.
