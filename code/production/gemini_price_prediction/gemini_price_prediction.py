import json
from google import genai
from dotenv import load_dotenv
import os
from datetime import datetime
from google.genai import types
import pandas as pd
import time
from tqdm import tqdm
from pydantic import BaseModel
from pathlib import Path


# Set up Gemini to process textual templates

# Load environment variables from the .env file
API_KEY = os.getenv("API_KEY")
client = genai.Client(api_key=API_KEY)  # Insert your API key here

load_dotenv(".env")
load_dotenv(r"C:\Users\Tammy\Documents\GitHub\multimodal_stockprice_prediction\code\production\modelling\.llm.env")

#  Load filepaths
stock_data_filepath = os.getenv("stock_data_filepath")  # This is the base directory for your files
target_stock = os.getenv("target_stock")  
llm_text_output_filepath = os.getenv("llm_text_output_filepath")
company_name = os.getenv('company_name')
text_type = os.getenv("text_type")

# Prepare file paths
output_path = Path(f"../../../data/raw")
full_output_path = output_path / f"gemini_price_predictions_{target_stock}.json"

# Construct the full file path using the target_stock variable
file_path = f"{stock_data_filepath}{target_stock}.csv"

# Load the DataFrame from the dynamically constructed file path
df = pd.read_csv(file_path)

# Construct the full file path for the news data (assuming the file is named with the stock ticker)
json_file_path = f"{llm_text_output_filepath}{text_type}_{target_stock}.json"

# Load the news data from the dynamically constructed JSON file path
with open(json_file_path, 'r') as f:
    news_data = json.load(f)

# Define the lookback window (5 days)
lookback_window = 5

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Set 'Date' column as the index
df.set_index('Date', inplace=True)

def calculate_short_term_indicators(price_series):
    """
    Calculate focused technical indicators using only price data
    
    Args:
        price_series (pd.Series): Recent price data
    
    Returns:
        dict: Concentrated technical indicators
    """
    import numpy as np
    
    # Simple and Exponential Moving Averages
    sma_5 = price_series.rolling(window=5).mean().iloc[-1]
    ema_5 = price_series.ewm(span=5, adjust=False).mean().iloc[-1]
    
    # Simple Momentum Indicator
    def calculate_momentum(data, periods=5):
        return (data.iloc[-1] - data.iloc[0]) / data.iloc[0] * 100
    
    # Volatility Approximation
    def calculate_volatility(data, periods=5):
        return data.rolling(window=periods).std() / data.mean() * 100
    
    # Rate of Change
    rate_of_change = calculate_momentum(price_series)
    
    # Volatility
    volatility = calculate_volatility(price_series).iloc[-1]
    
    return {
        'sma_5': sma_5,
        'ema_5': ema_5,
        'rate_of_change': rate_of_change,
        'volatility': volatility
    }

def create_text_template(df, start_idx, lookback_window, news_data):
    end_idx = start_idx + lookback_window
    price_window = df.iloc[start_idx:end_idx]
    
    # Calculate statistical details
    min_price = price_window[f"{target_stock}(t)"].min()
    max_price = price_window[f"{target_stock}(t)"].max()
    # avg_price = price_window['AAPL(t)'].mean()
    
    # Calculate technical indicators
    technical_indicators = calculate_short_term_indicators(price_window[f"{target_stock}(t)"])
    
    # Construct the "Trend Analysis" section
    trend_analysis = f"""
Trend Analysis:
- Minimum price: {min_price:.2f}
- Maximum price: {max_price:.2f}
- 5-Day Simple Moving Average: {technical_indicators['sma_5']:.2f}
- 5-Day Exponential Moving Average: {technical_indicators['ema_5']:.2f}
- Rate of Change: {technical_indicators['rate_of_change']:.2f}%
- Price Volatility: {technical_indicators['volatility']:.2f}%
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
- This is a stock analysis for {company_name} ({target_stock}) from {news_start_date} to {price_window.index[-1].strftime('%Y-%m-%d')}.
{trend_analysis}
{news_section}\n
{prediction_section}
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
                system_instruction=f"""
                You are a quantitative financial analyst specializing in short-term stock price forecasting. 
                Analyze the provided technical indicators, statistical metrics, and recent news to generate a precise, data-driven prediction for {company_name} ({target_stock}) stock price.

                IMPORTANT: Your task is to create a balanced analysis where both statistical indicators AND news sentiment are thoroughly evaluated. Neither should completely override the other, but you must not be blindsided by focusing solely on technical metrics.

                Input Format:
                The provided input will be structured as follows:
                - "Context": This section gives an overview of the stock analysis period.
                - "Trend Analysis": This section contains numerical details like minimum, maximum, moving averages, and rate of change.
                - "News": This section contains relevant news headlines for the given date range. If present, these must be thoroughly analyzed for sentiment and potential market impact.
                - "Prediction": This section requests a predicted stock price for a future date based on the given data.

                When making predictions:
                1. Evaluate both statistical indicators and news sentiment as complementary signals
                2. Consider how news might confirm, amplify, or contradict the statistical trends
                3. Be alert to significant news that could indicate trend changes not yet visible in the statistical data
                4. Provide a nuanced prediction that explains how you've weighted both quantitative metrics and qualitative news factors

                Your prediction should demonstrate a sophisticated integration of both data sources, avoiding overreliance on either statistical indicators or news sentiment alone.
                """,
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
    with open(full_output_path, 'w') as f:
        json.dump(predictions, f, indent=4)
    print(f"Predictions saved to '{full_output_path}'.")

# Generate predictions and store them in JSON format
generate_predictions_to_json(df, news_data, lookback_window)