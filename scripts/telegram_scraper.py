from telethon import TelegramClient
import csv
import os
from dotenv import load_dotenv
import pandas as pd

# Debugging: Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Debugging: Check if .env file exists
dotenv_path = os.path.join(os.getcwd(), '.env')
if os.path.exists(dotenv_path):
    print(f".env file found at: {dotenv_path}")
    # Try to read its content directly (for debugging purposes)
    try:
        with open(dotenv_path, 'r', encoding='utf-8') as f:
            print("\n--- .env file content (for debugging) ---")
            for line in f:
                print(line.strip())
            print("--- End .env file content ---\n")
    except Exception as e:
        print(f"Error reading .env file for debugging: {e}")
else:
    print(f"Error: .env file NOT found at: {dotenv_path}")
    print("Please ensure your .env file is in the same directory as your script, or specify its full path.")
    exit() # Exit if .env is not found, as we can't proceed without it


# Load environment variables once
load_dotenv('.env', override=True) # This line attempts to load the variables
print("Attempted to load .env variables.")

api_id = os.getenv('TG_API_ID')
api_hash = os.getenv('TG_API_HASH')
phone = os.getenv('phone')

# Debugging: Print what os.getenv actually returns
print(f"TG_API_ID from os.getenv: '{api_id}' (Type: {type(api_id)})")
print(f"TG_API_HASH from os.getenv: '{api_hash}' (Type: {type(api_hash)})")
print(f"phone from os.getenv: '{phone}' (Type: {type(phone)})")


# Convert api_id to an integer because TelegramClient expects it
try:
    if api_id is None:
        raise ValueError("TG_API_ID environment variable is not set.")
    api_id = int(api_id)
except (ValueError, TypeError) as e:
    print(f"Error: TG_API_ID in .env is not a valid integer or is missing. Details: {e}")
    print("Please ensure your .env file contains TG_API_ID=your_actual_api_id_number (without quotes).")
    exit() # Exit if conversion fails

# Function to scrape data from a single channel
async def scrape_channel(client, channel_username, writer, media_dir):
    try:
        entity = await client.get_entity(channel_username)
        channel_title = entity.title  # Extract the channel's title
        print(f"Scraping channel: {channel_title} ({channel_username})")
        async for message in client.iter_messages(entity, limit=500):
            media_path = None
            if message.media and hasattr(message.media, 'photo'):
                filename = f"{channel_username}_{message.id}.jpg"
                media_path = os.path.join(media_dir, filename)
                await client.download_media(message.media, media_path)
            
            writer.writerow([channel_title, channel_username, message.id, message.message, message.date, media_path])
    except Exception as e:
        print(f"Error scraping channel {channel_username}: {e}")
        # Optionally, you can log the error and continue with the next channel

# Initialize the client once
client = TelegramClient('scraping_session', api_id, api_hash)

async def main():
    await client.start()
    
    media_dir = 'photos'
    os.makedirs(media_dir, exist_ok=True)

    with open('telegram_data.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Channel Title', 'Channel Username', 'ID', 'Message', 'Date', 'Media Path'])
        
        channels_df = pd.read_excel("data/raw/5_channels_to_crawl.xlsx", header=None)
        channels = channels_df[0].dropna().tolist()
        print("Channels to scrape:", channels)
        
        for channel in channels:
            await scrape_channel(client, channel, writer, media_dir)
            print(f"Finished scraping data from {channel}")

with client:
    client.loop.run_until_complete(main())