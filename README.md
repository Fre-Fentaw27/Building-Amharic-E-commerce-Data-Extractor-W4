# Building-Amharic-E-commerce-Data-Extractor-W4

# Building an Amharic E-commerce Data Extractor

![Telegram Logo](https://upload.wikimedia.org/wikipedia/commons/8/82/Telegram_logo.svg)
![Ethiopia Flag](https://upload.wikimedia.org/wikipedia/commons/7/71/Flag_of_Ethiopia.svg)

A system for extracting structured product information from Ethiopian Telegram e-commerce channels, with specialized Amharic NLP processing.

## 📌 Project Overview

This project enables EthioMart to:

1. **Scrape** product listings from Ethiopian Telegram channels
2. **Process** Amharic text with specialized normalization
3. **Extract** key entities (products, prices, locations)
4. **Structure** data for centralized e-commerce catalog

## 🛠️ Technical Components

### Core Modules

- **Telegram Scraper**: Collects messages/media from channels
- **Amharic Preprocessor**: Handles Ethiopic script normalization
- **Labeling Tool**: Creates CoNLL-format datasets for NER

### Key Features

- Amharic-specific text cleaning
- Price pattern recognition (ብር/birr conversions)
- Location entity detection (Addis Ababa neighborhoods)
- Media download with metadata preservation

## 📂 Repository Structure

Building-Amharic-E-commerce-Data-Extractor-W4/
├── .venv/
├── data/
│ ├── clean/
│ ├── raw/
│ ├── for_annotation/
├── notebooks/
│ ├── preprocessing_task1.ipynb
├── scripts/
│ ├── telegram_scraper.py
│ ├── realtime_ingest.py
│ ├── preprocess_telegramdata.py
│ ├── labelstudio_annotation.py
│ ├── convert_to_ls_json.py
├── .env # API credentials
└── README.md
└── .gitignore
└── requirements.txt

## 🚀 Quick Start

### 1. Prerequisites

````bash
pip install -r requirements.txt

## ⚙️ Configuration

### 1. Environment Setup

Create a `.env` file in your project root:

```ini
# Telegram API credentials (from https://my.telegram.org)
TG_API_ID=your_api_id_here
TG_API_HASH=your_api_hash_here
PHONE_NUMBER=+251XXXXXXXXXX  # Ethiopian number preferred

# Optional settings
MAX_MESSAGES=500  # Per channel
MEDIA_DOWNLOAD=True

### 2.  Run Pipeline
# Step 1: Data Collection
  - python scripts/telegram_scraper.py

# Step 2: Text Processing
  - python scripts/preprocess_telegramdata.py
````

## Task 1: Data Ingestion & Preprocessing

## Task 2: Dataset Labeling

- Steps Completed:
- Sampled 50 diverse messages from unlabeled_conll.txt
- CoNLL-format annotation with:

```
B-PRODUCT: "ሽያጭ"
B-PRICE: "100"
I-PRICE: "ብር"

```
