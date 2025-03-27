import json
import os
import pandas as pd
import time
from datetime import datetime
from google.genai import types
from tqdm import tqdm
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv

# Set up Gemini to process textual templates
load_dotenv()

# Load environment variables from the .env file
API_KEY = os.getenv("API_KEY")
client = genai.Client(api_key=API_KEY)  # Insert your API key here

# Load the dataset (no longer using df, we'll use a dictionary)
df = pd.read_csv(r'C:\Users\Tammy\Documents\GitHub\multimodal_stockprice_prediction\data\clean\multimodal_inputs\baseline_transformed_dataset_AAPL.csv')

# Load the news data from the JSON file
with open(r'C:\Users\Tammy\Documents\GitHub\multimodal_stockprice_prediction\data\clean\multimodal_inputs\llm_text_data_processed_headline_AAPL.json', 'r') as f:
    news_data = json.load(f)

# Define the lookback window (5 days)
lookback_window = 5

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Set 'Date' column as the index
df.set_index('Date', inplace=True)

# Function to create textual templates for each rolling window
def create_text_template(df, start_idx, lookback_window, news_data):
    end_idx = start_idx + lookback_window
    price_window = df.iloc[start_idx:end_idx]
    
    # Calculate statistical details
    min_price = price_window['AAPL(t)'].min()
    max_price = price_window['AAPL(t)'].max()
    avg_price = price_window['AAPL(t)'].mean()
    rate_of_change = (price_window['AAPL(t)'].iloc[-1] - price_window['AAPL(t)'].iloc[0]) / price_window['AAPL(t)'].iloc[0] * 100
    
    # Construct the "Trend Analysis" section
    trend_analysis = f"""
Trend Analysis:
- Minimum price: {min_price:.2f}
- Maximum price: {max_price:.2f}
- Moving average: {avg_price:.2f}
- Rate of change: {rate_of_change:.2f}%
"""

    # Define the prediction date (the next valid trading day after the window)
    prediction_date = df.index[end_idx]  # This is the next day (next row) after the rolling window
    prediction_date_str = prediction_date.strftime('%Y-%m-%d')

    # Construct the "Prediction" section
    prediction_section = f"Instructions:\n- Predict the price for {prediction_date_str}.\n"

    # Get the relevant news headlines for the range (from start_idx to the day before prediction date)
    news_start_date = price_window.index[0].strftime('%Y-%m-%d')
    news_end_date = (prediction_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')  # End at the day before the prediction date

    # Aggregate news headlines from the news_data dictionary
    aggregated_news = []
    for single_date in pd.date_range(start=news_start_date, end=news_end_date).strftime('%Y-%m-%d'):
        if single_date in news_data:
            aggregated_news.extend(news_data[single_date])
    
    # Construct the "News" section
    news_section = f"News from {news_start_date} to {news_end_date}:\n"
    if aggregated_news:
        news_section += "\n".join([f"  - {headline}" for headline in aggregated_news])  # Indented list format
    else:
        news_section += "  - No news available during this period."

    # Combine all sections (Prediction, Trend Analysis, and News)
    template = f"""Context:
- This is a daily stock from {news_start_date} to {price_window.index[-1].strftime('%Y-%m-%d')}.

{trend_analysis}{news_section}\n{prediction_section}
"""
    
    return template


class StockPrediction(BaseModel):
    Date: str  # Date as a string (e.g., 'YYYY-MM-DD')
    PredictedPrice: str  # Predicted price as a string (could be float)

def gemini_predict(prompt):
    try:
        # Generate content from the model
        response = client.models.generate_content(
            model='gemini-2.0-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are an expert in stock price prediction, using technical indicators and market sentiment. Predict the stock price for Apple Inc. (AAPL) based on the provided context. This context includes statistical details such as minimum price, maximum price, average price, and rate of change, along with relevant news articles.",
                response_mime_type='application/json',
                response_schema=StockPrediction,  # Use the Pydantic model here
            )
        )

        # Check if the response is successful
        # The GenerateContentResponse does not have a status_code, so we check if response.text exists
        if response.text:
            # Assuming the response is a JSON string, parse it
            response_data = json.loads(response.text)  # Parse the response text

            # If the response is a list, extract the first prediction
            if isinstance(response_data, list):
                prediction = response_data[0]
            elif isinstance(response_data, dict):
                prediction = response_data
            else:
                raise ValueError("Unexpected response format")

            # Convert the response data into a StockPrediction model
            stock_prediction = StockPrediction(**prediction)

            # Extract and print the prediction data
            print(f"Predicted Date: {stock_prediction.Date}")
            print(f"Predicted Price: {stock_prediction.PredictedPrice}")
            return stock_prediction.Date, stock_prediction.PredictedPrice  # Optionally return the data

        else:
            print(f"Error: Empty response text")
            return None, None

    except Exception as e:
        # Catch any other exceptions (network, timeout, etc.)
        print(f"An error occurred: {e}")
        return None, None


# Function to generate stock predictions and store them in a JSON file
def generate_predictions_to_json(stock_data, news_data, lookback_window):
    predictions = []  # List to hold the predictions

    total_iterations = len(stock_data) - lookback_window
    print(f"Starting the prediction process for {total_iterations} windows...")

    # Initialize a progress bar (optional)
    for i in tqdm(range(total_iterations), desc="Generating predictions", unit="window"):
        # Generate the textual template
        template = create_text_template(df, i, lookback_window, news_data)

        # Print progress every 10 iterations
        if i % 10 == 0:
            print(f"Processing window {i+1}/{total_iterations}...")

        # Start timing the API call for better visibility on its duration
        start_time = time.time()

        # Get the stock prediction from the model
        prediction = gemini_predict(template)

        # Calculate the time taken for the API call
        api_duration = time.time() - start_time

        if prediction != (None, None):
            predictions.append(prediction)
            print(f"Prediction {i+1}: {prediction} (Time: {api_duration:.2f}s)")
        else:
            print(f"Error generating prediction for template starting at index {i}")

        # Print progress at each API call
        print(f"Time taken for API call at index {i}: {api_duration:.2f} seconds")
    
    # Save predictions to a JSON file
    with open(r'predictions.json', 'w') as f:
        json.dump(predictions, f, indent=4)
    print(f"Predictions saved to 'predictions.json'.")

# Generate predictions and store them in JSON format
generate_predictions_to_json(df, news_data, lookback_window)