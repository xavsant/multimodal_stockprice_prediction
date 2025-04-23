# Run headlines for all companies

#import dependencies
import pandas as pd
import numpy as np 
import os
from pathlib import Path
from dotenv import load_dotenv
from os import getenv

# VADER
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
# nltk.download('vader_lexicon')

# DistilRoberta and deberta
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def run_merged_models(input_path, output_path, ticker_mapping, text_type='headline'):
    
    # Run through files for all companies
    for file in input_path.rglob('*'):
        
        # Check if valid file
        if file.is_file() and file.name.endswith('.csv'):
            company = file.name.removesuffix("_text_data.csv")
            ticker = ticker_mapping[company]

            # Output
            output_name = 'finetuned_sentiment_analysis_' + text_type + '_' + ticker +  '.csv'
            print(output_name)

            # File does not exist yet
            if not os.path.exists(output_path / output_name):
                print(file.name)
                
                # Process headlines/abstract
                if text_type == 'headline':
                    sentiment_df = process_headlines(input_path / file.name, ticker)
                else:
                    sentiment_df = process_abstract(input_path / file.name)

                # Run models
                sentiment_df = run_vader(sentiment_df)
                print("vader done")
                sentiment_df = run_distilRoberta(sentiment_df)
                print("distil done")
                sentiment_df = run_deberta(sentiment_df)
                print("deberta done")

                sentiment_df.to_csv(output_path / output_name, index=False)
    
def process_abstract(path):
    # TO DO
    """
    For abstract, the file to be processed is in json format
    """
    return None


def process_headlines(path, ticker):
    df = pd.read_csv(path)

    # format datetime again
    df['pub_date'] = pd.to_datetime(df['pub_date']) 
    df['pub_date'] = df['pub_date'].dt.date

    # adding ticker in case of DF merge
    df['ticker'] = ticker
    df = df[['ticker'] + [col for col in df.columns if col != 'ticker']]

    sentiment_df = df[['ticker', 'pub_date', 'headline']]
    sentiment_df = sentiment_df.copy()  # Make a copy to avoid modifications on a slice

    return sentiment_df

def run_vader(sentiment_df):
    # Run model
    sia = SentimentIntensityAnalyzer()

    # Function to get full sentiment scores (including probabilities)
    def get_sentiment(text):
        # Get the sentiment scores dictionary (positive, neutral, negative, and compound scores)
        sentiment_scores = sia.polarity_scores(text)
        return sentiment_scores

    # Apply the sentiment analysis to the 'headline' column
    sentiment_df['sentiment_scores'] = sentiment_df['headline'].apply(get_sentiment)

    # Separate out the individual sentiment probabilities into new columns
    sentiment_df['vader_pos'] = sentiment_df['sentiment_scores'].apply(lambda x: x['pos'])
    sentiment_df['vader_neu'] = sentiment_df['sentiment_scores'].apply(lambda x: x['neu'])
    sentiment_df['vader_neg'] = sentiment_df['sentiment_scores'].apply(lambda x: x['neg'])

    # Function to classify sentiment into Positive, Negative, or Neutral based on the compound score
    def classify_sentiment(score):
        if score > 0.05:
            return 'Positive'
        elif score < -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    # Apply the compound score to classify sentiment
    sentiment_df['vader_label'] = sentiment_df['sentiment_scores'].apply(lambda x: classify_sentiment(x['compound']))

    # Display the result
    # Drop the 'sentiment_scores' column from the DataFrame
    sentiment_df = sentiment_df.drop(columns=['sentiment_scores'])

    return sentiment_df


# DistilRoberta
def run_distilRoberta(sentiment_df):
    tokenizer = AutoTokenizer.from_pretrained("frostedtrees/Fin_distilroberta")
    drob_model = AutoModelForSequenceClassification.from_pretrained("frostedtrees/Fin_distilroberta", num_labels=3)  # Assuming 3 sentiments: positive, neutral, negative

    def predict_sentiment(headline):
        inputs = tokenizer(headline, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = drob_model(**inputs)
        
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
        
        return probs[0].cpu().numpy()  # Returns an array of probabilities

    # Apply prediction to DataFrame
    sentiment_probs = sentiment_df['headline'].apply(predict_sentiment)

    # Create separate columns for sentiment probabilities
    sentiment_df[['drob_neg', 'drob_neu', 'drob_pos']] = pd.DataFrame(sentiment_probs.tolist(), index=sentiment_probs.index)

    # Determine final sentiment label based on highest probability and simplify label mapping
    sentiment_df['drob_label'] = sentiment_df[['drob_neg', 'drob_neu', 'drob_pos']].idxmax(axis=1)
    sentiment_df['drob_label'] = sentiment_df['drob_label'].replace({
        'drob_pos': 'Positive',
        'drob_neu': 'Neutral',
        'drob_neg': 'Negative'
    })

    return sentiment_df

def run_deberta(sentiment_df):
    tokenizer = AutoTokenizer.from_pretrained("tammiloveshf/Fin_DeBerta")
    deb_model = AutoModelForSequenceClassification.from_pretrained("tammiloveshf/Fin_DeBerta")

    # Function to get sentiment and probabilities for negative, neutral, and positive
    def get_sentiment(headline):
        # Tokenize the input headline
        encoded_text = tokenizer(headline, return_tensors='pt', padding=True, truncation=True)
        
        # Get the model's output (logits)
        output = deb_model(**encoded_text)
        
        # Apply softmax to get probabilities for each class (negative, neutral, positive)
        probs = torch.nn.functional.softmax(output.logits, dim=-1)
        
        # Extract sentiment scores (negative, neutral, positive)
        deb_neg, deb_neu, deb_pos = probs[0].detach().numpy()
        
        # Return sentiment scores as well as the label
        return deb_neg, deb_neu, deb_pos, 'Negative' if deb_neg > deb_neu and deb_neg > deb_pos else ('Neutral' if deb_neu > deb_pos else 'Positive')

    # Apply the function to the 'headline' column and unpack the values into separate columns
    sentiment_df[['deb_pos', 'deb_neg', 'deb_neu', 'deb_label']] = sentiment_df['headline'].apply(get_sentiment).apply(pd.Series)

    return sentiment_df


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

    run_merged_models(input_path, output_path, ticker_mapping, text_type)