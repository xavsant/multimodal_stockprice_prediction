# Note: requires stock data - need to reference dates for sentiment aggregation 
# Run after process_stock_data.py

# Imports
from pandas import read_csv, to_datetime
from dotenv import load_dotenv
from os import getenv
from datetime import timedelta

# Function to aggregate sentiment using a voting system
def map_sentiment(row):
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    print(row)
    return sentiment_map[row['gemini_sentiment']]

# Function to calculate weighted rolling average
def weighted_rolling_avg(price_df, sent_df, window_size=7, sentiment_effect=0.0001):
    """
    Calculate rolling average for sentiment data over x number of days before (window size), weighted by recency

    Parameters:
    - price_df: processed stock data
    - sent_df: sentiment data
    - window_size: number of days before current stock price date to aggregate sentiment for
    - sentiment_effect: impact of the sentiment
    
    """
    sent_df = sent_df.copy()
    price_df = price_df.copy()

    sent_df['aggregated_sentiment'] = sent_df.apply(map_sentiment, axis=1)
    sent_df['pub_date'] = to_datetime(sent_df['pub_date'])
    price_df.index = to_datetime(price_df.index)

    valid_dates = price_df.index
    weighted_sums = []

    for current_date in valid_dates:
        start_date = current_date - timedelta(days=window_size)
        end_date = current_date - timedelta(days=1)
        past_data = sent_df[(sent_df['pub_date'] >= start_date) & (sent_df['pub_date'] <= end_date)]

        if not past_data.empty:
            past_data['weighted'] = (window_size - (current_date - past_data['pub_date']).dt.days + 1) * sentiment_effect * past_data['aggregated_sentiment']
            weighted_sum = past_data['weighted'].sum()
        else:
            weighted_sum = 0 # neutral if no sentiment
        weighted_sums.append(weighted_sum)
    
    price_df['aggregated_sentiment'] = weighted_sums

    return price_df[['aggregated_sentiment']] # return index of price df + aggregated sentiment in df format


if __name__ == '__main__':
    load_dotenv('../modelling/.llm.env')
    target_stock = getenv('target_stock')
    stock_data_filepath = getenv('stock_data_filepath') + target_stock + '.csv'
    text_type = getenv('text_type')
    sentiment_input_filepath = '../../../data/clean/sentiment_analysis_results/gemini_sentiment_analysis_results_' + text_type + '_' + target_stock + '.csv'
    llm_sentiment_output_weighted_filepath = getenv('llm_sentiment_output_weighted_filepath') + text_type + '_' + target_stock + '.csv'

    # Initialise key variables
    window_size = 7
    sentiment_effect = 0.0001
    
    lstm_df = read_csv(stock_data_filepath, index_col='Date')
    sentiment_df = read_csv(sentiment_input_filepath)

    # Aggregate
    aggregate_sentiment_df = weighted_rolling_avg(lstm_df, sentiment_df, window_size=window_size, sentiment_effect=sentiment_effect)
    aggregate_sentiment_df.to_csv(llm_sentiment_output_weighted_filepath)
