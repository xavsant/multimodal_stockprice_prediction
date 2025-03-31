# To Do: Account for aggregate redundancies

# Imports
from pandas import read_csv, to_datetime, Timedelta
from numpy import where, ceil, floor
from dotenv import load_dotenv
from os import getenv

# Function to aggregate the features into a single score for each pub_date
def aggregate_features(sent_df):
    # Remove unwanted columns (drop 'gemini_sentiment', 'headline', and index if necessary)
    sent_df = sent_df.drop(columns=['index', 'gemini_sentiment', 'headline'], errors='ignore')
    
    # Calculate the mean of the relevant columns for each 'pub_date'
    # Ensure the relevant columns are float type to avoid integer division
    sent_df[['sentiment_score', 'trend_strength', 'impact_days']] = sent_df[['sentiment_score', 'trend_strength', 'impact_days']].astype(float)

    # This assumes that 'sentiment_score', 'trend_strength', and 'impact_days' are numeric
    aggregated_sent_df = sent_df.groupby('pub_date')[['sentiment_score', 'trend_strength', 'impact_days']].mean()

    # Calculate the aggregated feature score (you can change the aggregation logic here if needed)
    aggregated_sent_df['aggregated_feature_score'] = aggregated_sent_df[['sentiment_score', 'trend_strength', 'impact_days']].mean(axis=1)

    # Ensure 'aggregated_feature_score' is float to preserve decimal points
    aggregated_sent_df['aggregated_feature_score'] = aggregated_sent_df['aggregated_feature_score'].astype(float)
    
    # Drop individual feature columns for the aggregated feature score
    aggregated_sent_df = aggregated_sent_df[['aggregated_feature_score']]

    return aggregated_sent_df

#Function to reindex the sentiment dataframe for valid trading days
def shift_to_next_working_day(aggregated_sent_df, price_df):
    # Ensure the index of both dataframes is datetime type
    aggregated_sent_df.index = to_datetime(aggregated_sent_df.index)  
    price_df.index = to_datetime(price_df.index)

    # Set of valid trading dates
    valid_dates = set(price_df.index)

    # Adjust the dates in aggregated_sent_df to match the trading days
    adjusted_dates = []
    for date in aggregated_sent_df.index:
        while date not in valid_dates:  # If the date is not valid, move to the next day
            date += Timedelta(days=1)
        adjusted_dates.append(date)

    # Update the index of aggregated_sent_df with adjusted dates
    aggregated_sent_df.index = adjusted_dates

    # Optional: Aggregate duplicate dates (if any) after shifting
    aggregated_sent_df = aggregated_sent_df.groupby(aggregated_sent_df.index).mean()

    return aggregated_sent_df

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
    aggregated_sentiment_df_adjusted = shift_to_next_working_day(aggregated_sentiment_df, lstm_df)

    # Reindex to lstm_df
    aggregate_sentiment_df_reindex = aggregated_sentiment_df_adjusted.reindex(lstm_df.index, method=None)
    aggregate_sentiment_df_reindex['aggregated_feature_score'] = aggregate_sentiment_df_reindex['aggregated_feature_score'].fillna(0).astype(float).round(2)
    
    aggregate_sentiment_df_reindex.to_csv(llm_feature_output_filepath)