import pandas as pd
import json
from pathlib import Path

# Configuration
INPUT_PATH = 'data/clean/processed_telegram_data.csv'
OUTPUT_PATH = 'data/for_annotation/label_studio_tasks.jsonl'
TEXT_COLUMN = 'Cleaned_Message'  # Column with cleaned Amharic text
INCLUDE_METADATA = True  # Whether to include channel info and timestamps

def prepare_annotation_tasks():
    """Convert processed data to Label Studio JSONL format"""
    # Ensure output directory exists
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    # Load data with explicit UTF-8 encoding
    df = pd.read_csv(INPUT_PATH, encoding='utf-8')
    
    # Filter and clean messages
    messages = df[TEXT_COLUMN].dropna()
    messages = messages[messages.str.strip() != '']  # Remove empty strings
    
    # Export to JSONL
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for idx, msg in enumerate(messages):
            task = {
                "data": {
                    "text": msg,
                    # Include metadata if enabled
                    **({"metadata": {
                        "source": df.loc[idx, 'Channel Title'],
                        "date": df.loc[idx, 'Date']
                    }} if INCLUDE_METADATA else {})
                }
            }
            f.write(json.dumps(task, ensure_ascii=False) + '\n')
    
    print(f"✅ Successfully exported {len(messages)} tasks to {OUTPUT_PATH}")

if __name__ == '__main__':
    prepare_annotation_tasks()