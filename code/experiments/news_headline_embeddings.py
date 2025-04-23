import json
import logging
from google import genai
import os
from dotenv import load_dotenv
import numpy as np
import time
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Set up Gemini to process textual templates
load_dotenv()

# Load news data
with open(r'C:\Users\Tammy\Documents\GitHub\multimodal_stockprice_prediction\data\clean\multimodal_inputs\llm_text_data_processed_headline_AAPL.json', 'r') as f:
    news_data = json.load(f)
# Set up Gemini to process textual templates
API_KEY = os.getenv("API_KEY")  # Replace with your API key
client = genai.Client(api_key=API_KEY)

# Maximum requests per minute
MAX_REQUESTS_PER_MINUTE = 140

# Rate limiting and request tracking variables
requests_in_minute = 0
start_time = time.time()

# Concatenate the embeddings to create a single representation of the day's news
def get_text_embeddings(text_templates, dates):
    global requests_in_minute, start_time
    text_embeddings = {}

    for i, text in enumerate(text_templates):
        if not text.strip():  # Skip empty headlines (and their corresponding dates)
            continue

        # Check if we have exceeded the request limit (140 requests per minute)
        current_time = time.time()
        
        # Reset the request count every minute
        if current_time - start_time >= 60:
            requests_in_minute = 0
            start_time = current_time
        
        # If we've reached the limit, wait until the next minute
        if requests_in_minute >= MAX_REQUESTS_PER_MINUTE:
            time_to_wait = 60 - (current_time - start_time)
            logger.info(f"Request limit reached. Waiting for {time_to_wait:.2f} seconds to reset the rate limit.")
            time.sleep(time_to_wait)  # Sleep until the next minute

        try:
            # Log the start of embedding process
            logger.info(f"Fetching embedding for date: {dates[i]} (Article {i+1}/{len(text_templates)})")

            # Fetch embeddings for each article
            response = client.models.embed_content(
                model='text-embedding-004',
                contents=text
            )
            
            # Extract the embedding values
            embedding = np.array(response.embeddings[0].values)  # Get the embedding for the current article
            
            # Concatenate embeddings for each day
            if dates[i] in text_embeddings:
                text_embeddings[dates[i]].append(embedding)  # Append the current embedding to the list
            else:
                text_embeddings[dates[i]] = [embedding]  # Initialize with the first article's embedding
            
            # Increment the request counter
            requests_in_minute += 1

        except Exception as e:
            logger.error(f"Error occurred while fetching embedding for date {dates[i]}: {e}")
            continue  # Skip to the next article

    # Now concatenate the embeddings for each day
    logger.info("Concatenating the embeddings for each day...")
    for date in text_embeddings:
        text_embeddings[date] = np.concatenate(text_embeddings[date], axis=0).tolist()  # Concatenate embeddings for each day and convert to list for JSON compatibility

    return text_embeddings

# Process all available data (no limit)
def create_text_templates(news_data):
    text_templates = []
    dates = []
    logger.info("Creating text templates from news data...")

    for date, headlines in news_data.items():
        if not headlines:  # Skip dates with no headlines
            continue

        dates.append(date)  # Store the date
        text = " ".join(headlines)  # Combine all headlines for a given date into one string
        text_templates.append(text)
    
    return text_templates, dates

# Use the create_text_templates function without a limit
text_templates, dates = create_text_templates(news_data)

# Get textual embeddings using Gemini
logger.info("Fetching textual embeddings...")
text_embeddings = get_text_embeddings(text_templates, dates)

# Save the embeddings to a JSON file for verification
output_file = "news_headlines_embeddings_concat.json"
logger.info(f"Saving embeddings to {output_file}...")
with open(output_file, 'w') as f:
    json.dump(text_embeddings, f)

logger.info(f"Embeddings saved to {output_file}")
