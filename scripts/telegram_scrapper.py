from telethon import TelegramClient, errors
import csv
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)

# Load .env from the same directory as the script
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)
api_id = os.getenv('TG_API_ID')
api_hash = os.getenv('TG_API_HASH')
phone = os.getenv('PHONE')  # Note: Capitalized to match standard .env convention

# Validate environment variables
if not all([api_id, api_hash, phone]):
    raise ValueError("Missing required environment variables in .env file")

async def scrape_channel(client, channel_username, writer, media_dir):
    try:
        entity = await client.get_entity(channel_username)
        channel_title = entity.title
        
        async for message in client.iter_messages(entity, limit=10000):
            try:
                media_path = None
                if message.media and hasattr(message.media, 'photo'):
                    filename = f"{channel_username}_{message.id}_{int(datetime.now().timestamp())}.jpg"
                    media_path = os.path.join(media_dir, filename)
                    await client.download_media(message.media, file=media_path)
                
                writer.writerow([
                    channel_title,
                    channel_username,
                    message.id,
                    message.text or "",  # Handle None messages
                    message.date.isoformat() if message.date else "",
                    media_path or ""
                ])
                
            except Exception as e:
                logging.error(f"Error processing message {message.id} in {channel_username}: {str(e)}")
                continue
                
    except errors.ChannelPrivateError:
        logging.error(f"Channel {channel_username} is private and cannot be accessed")
    except errors.ChannelInvalidError:
        logging.error(f"Channel {channel_username} is invalid or doesn't exist")
    except Exception as e:
        logging.error(f"Unexpected error scraping {channel_username}: {str(e)}")

async def main():
    try:
        client = TelegramClient('scraping_session', int(api_id), api_hash)
        await client.start(phone)
        
        if not await client.is_user_authorized():
            raise Exception("Client not authorized. Please check your phone number and login.")
        
        media_dir = 'telegram_media'
        os.makedirs(media_dir, exist_ok=True)

        with open('telegram_data.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Channel Title', 'Channel Username', 'Message ID', 'Text', 'Date', 'Media Path'])
            
            channels = [
                '@Shageronlinestore',
                # Add more channels here
                # '@ExampleChannel1',
                # '@ExampleChannel2',
            ]
            
            for channel in channels:
                logging.info(f"Starting scrape for {channel}")
                await scrape_channel(client, channel, writer, media_dir)
                logging.info(f"Completed scrape for {channel}")
                
    except Exception as e:
        logging.error(f"Fatal error in main: {str(e)}")
    finally:
        if 'client' in locals():
            await client.disconnect()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())