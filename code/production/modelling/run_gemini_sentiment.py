# Run headlines for all companies

#import dependencies
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv
from os import getenv
import re
from google import genai
from google.genai import types
import json

def run_gemini_sentiment(input_path, output_path, ticker_mapping, text_type='headline'):
    
    # Set up gemini client
    client = setup_gemini()

    # Run through files for all companies
    for file in input_path.rglob('*'):

        # Check if valid file
        if file.is_file() and file.name.endswith('.csv'):
            company = file.name.removesuffix("_text_data.csv")
            ticker = ticker_mapping[company]
            output_name = 'gemini_sentiment_analysis_results_' + text_type + '_' + ticker +  '.csv'

            # Check if valid file and if file already exists
            if ticker != '^DJI': # not os.path.exists(output_path / output_name):
                print(file.name)

                # Process headlines/abstract
                sentiment_df = process_text(input_path / file.name, ticker, text_type)

                # Run models
                sentiment_df = apply_without_delay(sentiment_df, client, text_type, ticker)

                sentiment_df.to_csv(output_path / output_name, index=False)
    

def apply_without_delay(df, client, text_type, ticker):
    sentiment_list = []
    print(df.columns)
    for idx, text in enumerate(df[text_type]):
        print(f"Processing {idx + 1}/{len(df)}: {text}")  # Progress indicator
        
        sentiment = find_sentiment_few_shot(text, client, text_type, ticker).lower()  # Gemini prediction in lowercase
        sentiment_list.append(sentiment)
    
    # Add Gemini's predictions while keeping original labels intact
    df['gemini_sentiment'] = sentiment_list
    return df


def process_text(path, ticker, text_type='headline'):
    df = pd.read_csv(path, encoding='utf-8')

    # format datetime again
    df['pub_date'] = pd.to_datetime(df['pub_date']) 
    df['pub_date'] = df['pub_date'].dt.date

    # Cleaning
    df[text_type] = df[text_type].apply(clean_text)

    # adding ticker in case of DF merge
    df['ticker'] = ticker
    df = df[['ticker'] + [col for col in df.columns if col != 'ticker']]

    sentiment_df = df[['ticker', 'pub_date', text_type]]
    sentiment_df = sentiment_df.copy()  # Make a copy to avoid modifications on a slice

    return sentiment_df

def clean_text(text):
    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and special characters (except spaces)
    text = re.sub(r'[^\w\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def setup_gemini():

    # Load environment variables from the .env file
    load_dotenv()
    API_KEY = os.getenv("API_KEY")
    client = genai.Client(api_key=API_KEY)#insert api key

    return client

def find_sentiment_few_shot(text, client, text_type, ticker):
    prompt = f"""
    Classify the sentiment of the financial {text_type} with respect to {ticker}'s stock price as 'Positive', 'Negative', or 'Neutral'. 
    Consider both direct impacts on {ticker} and indirect effects from industry or market-wide events. For instance, if the {text_type} is about {ticker}'s new product, 
    market competition, or financial performance, focus on those. If the {text_type} is about the tech industry as a whole, global economy, or competitor news, 
    consider how it may impact {ticker}'s stock.

    Return the result in JSON format:

    Example:
    {{ "Sentiment": "Positive" }}

    Example 1:
    {text_type}: "Company Close to Finalizing Its 40 billion dollar funding." 
    Sentiment: Positive

    Example 2:
    {text_type}: "Trump blocks 10% of funds for key agency in US-China Tech Race."
    Sentiment: Negative

    Example 3:
    {text_type}: "Why Company B could be a key to a Company C's Deal."
    Sentiment: Neutral

    Example 4:
    {text_type}: "Artificial intelligence boom might help mitigate some tariff pain."
    Sentiment: Neutral

    Example 5:
    {text_type}: "Major banks face regulatory hurdles, impacting earnings outlook."
    Sentiment: Negative

    Example 6:
    {text_type}: "Company’s $32 billion deal may signal a turning point for slow IPO, M&A markets."
    Sentiment: Positive

    Example 7:
    {text_type}: "Company X enters a strategic partnership with Company Y to expand its operations in Asia."
    Sentiment: Positive

    {text_type}: "{text}"
    Response (in JSON format):
    {{ "Sentiment":"""
    
    sentiment = gemini_predict(prompt, client, text_type, ticker)
    return sentiment


def gemini_predict(prompt, client, text_type, ticker):

    # system_instruction = (
    # "You are a financial analyst with expertise in evaluating the sentiment of financial news, especially in the context of stock prices. "
    # "While analyzing sentiment for any company, including {}, consider both the direct and indirect impacts. Apple is a major player in the tech industry, "
    # "and its stock is influenced not only by its own developments but also by broader market trends, competitor actions, and industry news. "
    # "Your task is to classify the sentiment of {} with respect to Apple's stock price, considering both direct and market-wide influences. "
    # "Classify the sentiment as 'Positive', 'Negative', or 'Neutral'."
    # ).format(ticker, text_type)

    system_instruction = f"""
    "You are a financial analyst with expertise in evaluating the sentiment of financial news, especially in the context of stock prices. "
    "While analyzing sentiment for any company, including {ticker}, consider both the direct and indirect impacts. {ticker} is a major player in the tech industry, "
    "and its stock is influenced not only by its own developments but also by broader market trends, competitor actions, and industry news. "
    "Your task is to classify the sentiment of the {text_type} with respect to {ticker}'s stock price, considering both direct and market-wide influences. "
    "Classify the sentiment as 'Positive', 'Negative', or 'Neutral'."
    """

    try:
        # Generate content from the model
        response = client.models.generate_content(
            model='gemini-2.0-flash-lite', 
            contents=prompt,
            config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            max_output_tokens=60,  # Label only
            temperature=0.5,      # More flexibility
            top_k=5,              # Limit to top 5 choices
            top_p=0.7,            # Consider tokens covering 70% probability mass
            response_mime_type='application/json',
            stop_sequences=['}']   # No stop sequence to avoid premature stops
            )
        )
        
        response_text = response.text.strip()
        
        # Attempt to fix incomplete JSON by appending a missing closing brace
        if not response_text.endswith('}'):
            response_text += '}'
        
        # Attempt to parse JSON
        try:
            response_json = json.loads(response_text)  # Parse JSON
            sentiment = response_json.get("Sentiment", "").strip()
            
            if sentiment not in ['Positive', 'Negative', 'Neutral']:
                sentiment = 'Neutral'
            
            return sentiment
        
        except json.JSONDecodeError:
            print(f"JSON Parsing Error. Response: {response_text}")
            return 'Error'
    
    except Exception as e:
        print(f"API Error: {e}")
        return 'Error'


if __name__ == '__main__':
    # load_dotenv('../modeling/.ensemble.env')
    text_type = 'headline'

    input_path = Path('../../../data/clean/stock_text_data')
    output_path = Path('../../../data/clean/sentiment_analysis_results')

    ticker_mapping = {'Amazon.com_Inc':'AMZN',
                      'Apple_Inc': 'AAPL',
                      'International_Business_Machines_Corporation':'IBM',
                      'Microsoft_Corp':'MSFT',
                      'Nvidia_Corporation':'NVDA',
                      'Salesforce.com_Inc':'CRM',
                      '^DJI':'^DJI'
                    }

    run_gemini_sentiment(input_path, output_path, ticker_mapping, text_type)