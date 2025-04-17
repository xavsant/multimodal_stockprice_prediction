# Imports
# from numpy import where, ceil, floor
from dotenv import load_dotenv
from os import getenv
import pandas as pd
from pandas import read_csv, to_datetime, Timedelta


def aggregate_features(sent_df):

    # Step 1: Drop the 'headline' column and 'index' if it exists (if you have it as a column)
    sent_df = sent_df.drop(columns=['index','headline'], errors='ignore')

    # Step 2: Group by 'pub_date' and calculate the mean of 'sentiment_score' and 'impact_days'
    # Ensure 'pub_date' is a datetime object
    sent_df['pub_date'] = pd.to_datetime(sent_df['pub_date'])

    # Group by 'pub_date' and calculate the mean
    aggregated_sent_df = sent_df.groupby('pub_date')[['sentiment_score', 'impact_days']].mean().reset_index()
    
    return aggregated_sent_df

# Function to apply temporal impact and generate cumulative sentiment for each calendar day (not just trading days)
def apply_temporal_impact_all_days(aggregated_sent_df, price_df):
    # Ensure the index of both dataframes is datetime type
    aggregated_sent_df['pub_date'] = pd.to_datetime(aggregated_sent_df['pub_date'])
    aggregated_sent_df.set_index('pub_date', inplace=True)
 
    price_df.index = pd.to_datetime(price_df.index)
    
    # Create a DataFrame with all calendar days (not just trading days)
    all_days_index = pd.date_range(start=price_df.index.min(), end=price_df.index.max(), freq='D')
    result_df = pd.DataFrame(index=all_days_index)
    result_df['cumulative_sentiment'] = 0.0  # Initialize cumulative sentiment for all days
 
    # For each news publication date, calculate its decaying impact on subsequent days (including non-trading days)
    for pub_date, row in aggregated_sent_df.iterrows():
        sentiment = row['sentiment_score']
        impact_period = int(row['impact_days'])  # Duration of impact
        
        # Apply sentiment impact over the specified duration, including non-trading days
        for i in range(impact_period):
            target_date = pub_date + pd.Timedelta(days=i)
            
            # Ensure the target date is within the range of result_df index (which includes all days)
            if target_date not in result_df.index:
                continue
            
            # Calculate decay factor (linear decay from 1.0 to 0.0 over impact_period)
            decay_factor = 1.0 - (i / impact_period)
            
            # Apply decaying sentiment without using impact days as weights
            weighted_impact = sentiment * decay_factor
            result_df.loc[target_date, 'cumulative_sentiment'] += weighted_impact

    # Now merge the cumulative_sentiment with the price_df by index (dates)
    result_df = result_df[result_df.index.isin(price_df.index)]
    
    return result_df

# Function to combine the previous two functions and handle non-trading days
def process_sentiment_data_all_days(sent_df, price_df):
    # First aggregate sentiment data by publication date
    agg_sent_df = aggregate_features(sent_df)
    
    # Then calculate the impact of each publication on every calendar day
    result_df = apply_temporal_impact_all_days(agg_sent_df, price_df)
    
    return result_df

if __name__ == '__main__':
    load_dotenv('../modelling/.llm.env')
    target_stock = getenv('target_stock')
    stock_data_filepath = getenv('stock_data_filepath') + target_stock + '.csv'
    text_type = getenv('text_type')
    sentiment_input_filepath = '../../../data/clean/sentiment_analysis_results/gemini_' + text_type + '_features_predictions_' + target_stock + '.csv'
    llm_feature_output_filepath = getenv('llm_feature_output_filepath') + text_type + '_' + target_stock + '.csv'
    
    lstm_df = read_csv(stock_data_filepath, index_col='Date')
    sentiment_df = read_csv(sentiment_input_filepath)

    # Aggregate first, then adjust dates
    aggregated_sentiment_df = aggregate_features(sentiment_df)
    process_sentiment_data =process_sentiment_data_all_days(aggregated_sentiment_df, lstm_df)

    # Reindex to lstm_df
    aggregate_sentiment_df_reindex = process_sentiment_data.reindex(lstm_df.index, method=None)
    aggregate_sentiment_df_reindex['cumulative_sentiment'] = aggregate_sentiment_df_reindex['cumulative_sentiment'].fillna(0).astype(float)
    
    aggregate_sentiment_df_reindex.to_csv(llm_feature_output_filepath)