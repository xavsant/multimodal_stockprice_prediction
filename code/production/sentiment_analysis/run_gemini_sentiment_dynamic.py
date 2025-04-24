from pathlib import Path
import pandas as pd
import re
import json
import os
from tqdm import tqdm
from google import genai
from dotenv import load_dotenv
from pydantic import BaseModel
from google.genai import types

# Pydantic models for response validation
class SentimentTrendPrediction(BaseModel):
    SentimentScore: float

class DurationPrediction(BaseModel):
    PotentialImpactDays: int

# Cleaning function
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

# Function to call Gemini API for sentiment prediction
def gemini_predict_sentiment_trend(prompt, return_json=False):
    try:
        # Get API key
        API_KEY = os.getenv("API_KEY")
        client = genai.Client(api_key=API_KEY)
        
        # Generate content from the model
        response = client.models.generate_content(
            model='gemini-2.0-flash-lite', 
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=f"""
                You are a financial analyst specializing in both quantitative market sentiment and trend analysis. Your task is to analyze each {text_type} and provide:

                **Sentiment Score (-1.0 to 1.0)**:
                - -1.0: Extremely negative news (bankruptcy, massive fraud, severe regulatory action)
                - 0.0: Neutral or balanced news (mixed results, status quo maintained)
                - 1.0: Extremely positive news (breakthrough products, industry-changing acquisitions)
                - Consider both **direct effects on the stock price** and **general market or sector trends**.

                CRITICAL INSTRUCTIONS:
                - Evaluate stock movement within the broader industry context and the overall market sentiment.
                - Consider broader market trends, including economic events, industry-wide shifts, or global developments that may indirectly impact the company or its sector. 
                - If the {text_type} does not pertain to the company or its sector but has potential ripple effects, assign a neutral sentiment (0.0) or assess if it has any relevant impact on market sentiment as a whole.
                - For ambiguous {text_type}s, default to low scores (-0.1 to +0.1 range).
                - Return ONLY one value in JSON format like:
                {{ "SentimentScore": 0.7 }}
                """,
                max_output_tokens=75,
                temperature=0.5,
                top_k=5,
                top_p=0.7,
                response_mime_type='application/json',
                response_schema=SentimentTrendPrediction
            )
        )
        
        # Check if the response is valid
        response_text = response.text.strip()
        if not response_text:
            print("Error: No response text found.")
            return 'Error' if not return_json else {}

        # Try parsing the response as JSON
        try:
            response_json = json.loads(response_text)
            if return_json:
                return response_json
            return response_json
        except json.JSONDecodeError:
            print(f"Error parsing JSON. Response: {response_text}")
            return 'Error' if not return_json else {}

    except Exception as e:
        print(f"API Error: {e}")
        return 'Error' if not return_json else {}

# Function to analyze sentiment trend with examples
def analyze_sentiment_trend(text, ticker):
    company_context = f"company related to ticker {ticker}"
    
    prompt = f"""
    Analyze this financial {text_type} about {company_context} and provide: 
    Sentiment Score: from -1.0 (extremely negative) to 1.0 (extremely positive)

    Return results in JSON format like:
    {{ "SentimentScore": X.X }}

    Here are some examples:

    Example 1: 
    {text_type}: "Company Close to Finalizing Its 40 billion dollar funding." 
    {{ "SentimentScore": 0.9}}

    Example 2: 
    {text_type}: "Regulatory authorities block 10% of funds for key agency in US-China Tech Race."
    {{ "SentimentScore": -0.8 }}

    Example 3: 
    {text_type}: "Why Company B could be a key to a Company C's Deal."
    {{ "SentimentScore": 0.2 }}

    Example 4: 
    {text_type}: "Artificial intelligence boom might help mitigate some tariff pain."
    {{ "SentimentScore": 0.3}}

    Example 5: 
    {text_type}: "Major banks face regulatory hurdles, impacting earnings outlook."
    {{ "SentimentScore": -0.6 }}

    Example 6: 
    {text_type}: "Company's $32 billion deal may signal a turning point for slow IPO, M&A markets."
    {{ "SentimentScore": 0.8 }}

    Example 7: 
    {text_type}: "Company X enters a strategic partnership with Company Y to expand its operations in Asia."
    {{ "SentimentScore": 0.7 }}

    Now, analyze this {text_type}:

    {text_type}: "{text}"
    Response (in JSON format):
    {{ "SentimentScore": }}"""
    
    result = gemini_predict_sentiment_trend(prompt)
    return result

# Function to call Gemini API for duration prediction
def gemini_predict_duration(prompt, return_json=False):
    try:
        # Get API key
        API_KEY = os.getenv("API_KEY")
        client = genai.Client(api_key=API_KEY)
        
        # Generate content from the model
        response = client.models.generate_content(
            model='gemini-2.0-flash-lite', 
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=f"""
                You are a financial market expert specializing in estimating the impact duration of financial news. Your task is to analyze each {text_type} and estimate how long its impact will last on the stock market.

                Consider:
                - **1-3 days**: Short-lived news (minor developments, non-urgent reactions).
                - **4-7 days**: Moderately impactful news (earnings reports, key regulatory actions).
                - **8-14 days**: Significant developments (corporate mergers, product announcements).
                - **15-30 days**: Major structural changes (industry shifts, global economic impacts).

                For company-specific news, consider historical precedents for how similar news has impacted the stock in the past. For market-wide or industry news, estimate how long the ripple effects might last, considering the company's role in its sector.

                Return only the result in JSON format like:
                {{ "PotentialImpactDays": Z }}
                """,
                max_output_tokens=50,
                temperature=0.5,
                top_k=5,
                top_p=0.7,
                response_mime_type='application/json',
                response_schema=DurationPrediction
            )
        )
        
        # Check if the response is valid
        response_text = response.text.strip()
        if not response_text:
            print("Error: No response text found.")
            return 'Error' if not return_json else {}

        # Try parsing the response as JSON
        try:
            response_json = json.loads(response_text)
            if return_json:
                return response_json
            return response_json
        except json.JSONDecodeError:
            print(f"Error parsing JSON. Response: {response_text}")
            return 'Error' if not return_json else {}

    except Exception as e:
        print(f"API Error: {e}")
        return 'Error' if not return_json else {}

# Function to analyze duration with examples
def analyze_duration(text, ticker):
    company_context = f"company related to ticker {ticker}"
    
    prompt = f"""
    Analyze this financial {text_type} about {company_context} and estimate how many days (1-30) this news might impact the market.
    Consider: 
    - 1-3 days for short-lived news
    - 4-7 days for moderately impactful news 
    - 8-14 days for significant developments
    - 15-30 days for major structural changes or significant corporate events
    
    For company-specific news (e.g., product launches, earnings results), consider the historical reaction of the stock and the potential market sentiment based on previous similar events.

    For market-wide news (e.g., interest rate changes, regulatory updates), estimate how it will ripple through the sector and its potential impact on the company's stock.

    Return result in JSON format like:
    {{ "PotentialImpactDays": Z }}

    Examples:

    Example 1: 
    {text_type}: "Company close to finalizing its 40 billion dollar funding." 
    {{ "PotentialImpactDays": 14 }}

    Example 2: 
    {text_type}: "Regulatory authorities block 10% of funds for key agency in US-China Tech Race."
    {{ "PotentialImpactDays": 7 }}

    Example 3: 
    {text_type}: "Why Company B could be a key to a Company C's Deal."
    {{ "PotentialImpactDays": 3 }}

    Example 4: 
    {text_type}: "Artificial intelligence boom might help mitigate some tariff pain."
    {{ "PotentialImpactDays": 5 }}

    Example 5: 
    {text_type}: "Major banks face regulatory hurdles, impacting earnings outlook."
    {{ "PotentialImpactDays": 10 }}

    Example 6: 
    {text_type}: "Company's $32 billion deal may signal a turning point for slow IPO, M&A markets."
    {{ "PotentialImpactDays": 10 }}

    Now analyze this {text_type}:
    {text_type}: "{text}"
    """
    
    result = gemini_predict_duration(prompt)
    return result

# Function to process a single file
def process_file(file_path, output_path, ticker, text_type):
    print(f"Processing {file_path}...")
    
    # Load the CSV file
    df = pd.read_csv(file_path, encoding='utf-8')
    
    # Format datetime
    if 'pub_date' in df.columns:
        df['pub_date'] = pd.to_datetime(df['pub_date'])
        df['pub_date'] = df['pub_date'].dt.date
    
    # Clean the text
    df[text_type] = df[text_type].apply(clean_text)
    
    # Select only the 'pub_date' and text_type columns
    if 'pub_date' in df.columns:
        df_new = df[['pub_date', text_type]]
    else:
        df_new = df[[text_type]]
    
    sentiment_score_list = []
    impact_days_list = []
    
    # Process with first model (sentiment)
    for idx, text in tqdm(enumerate(df_new[text_type]), total=len(df_new), desc=f"Processing Sentiment for {ticker}", unit="text"):
        # Get sentiment predictions
        result = analyze_sentiment_trend(text, ticker)
        
        if isinstance(result, dict) and 'SentimentScore' in result:
            sentiment_score = result['SentimentScore']
            sentiment_score_list.append(sentiment_score)
        else:
            sentiment_score_list.append(None)
    
    # Add sentiment to DataFrame
    df_new['sentiment_score'] = sentiment_score_list
    
    # Process with second model (duration)
    for idx, text in tqdm(enumerate(df_new[text_type]), total=len(df_new), desc=f"Processing Duration for {ticker}", unit="text"):
        # Get duration prediction
        result = analyze_duration(text, ticker)
        
        if isinstance(result, dict) and 'PotentialImpactDays' in result:
            impact_days = result['PotentialImpactDays']
            impact_days_list.append(impact_days)
        else:
            impact_days_list.append(None)
    
    # Add duration to DataFrame
    df_new['impact_days'] = impact_days_list
    
    # Create output filename
    output_file = output_path / f"gemini_{text_type}_features_predictions_{ticker}.csv"
    
    # Save to CSV
    df_new.to_csv(output_file, index=False)
    print(f"Analysis complete for {ticker}. Results saved to {output_file}")
    
    return df_new

def run_gemini_sentiment(input_path, output_path, ticker_mapping, text_type):
    """
    Run sentiment analysis on stock text data for multiple tickers
    
    Args:
        input_path: Path to input directory containing CSV files
        output_path: Path to output directory for results
        ticker_mapping: Dictionary mapping company names to ticker symbols
        text_type: Type of text to analyze ('headline' or 'abstract')
    """
    # Make sure output path exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each ticker
    for company_name, ticker in ticker_mapping.items():
        # Construct file path
        file_name = f"{company_name}_text_data.csv"
        file_path = input_path / file_name
        
        if file_path.exists():
            # Process the file
            process_file(file_path, output_path, ticker, text_type)
        else:
            print(f"Warning: File not found: {file_path}")

if __name__ == '__main__':
    # load_dotenv('../modeling/.ensemble.env')
    load_dotenv()  # Load environment variables from .env file
    
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