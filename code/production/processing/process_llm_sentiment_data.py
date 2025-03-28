# To Do: Account for aggregate redundancies

# Imports
from pandas import read_csv, to_datetime, Timedelta
from numpy import where, ceil, floor
from dotenv import load_dotenv
from os import getenv

# Function to aggregate sentiment using a voting system
def aggregate_sentiment(row):
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    
    sentiments = [row['gemini_sentiment']]
    
    sentiment_counts = {'positive': sentiments.count('positive'), 
                        'neutral': sentiments.count('neutral'), 
                        'negative': sentiments.count('negative')}
    
    majority_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    return sentiment_map[majority_sentiment]

# Function to aggregate sentiments by pub_date with rounding
def aggregate_by_pub_date(sent_df):
    sent_df['aggregated_sentiment'] = sent_df.apply(aggregate_sentiment, axis=1)
    
    # Group by pub_date and calculate the average sentiment per day
    aggregated_sent_df = sent_df.groupby('pub_date')['aggregated_sentiment'].mean().to_frame()
    
    # Round the aggregated sentiment and convert to int (-1, 0, or 1)
    aggregated_sent_df['aggregated_sentiment'] = custom_round(aggregated_sent_df['aggregated_sentiment'])
    
    return aggregated_sent_df

# Function to adjust dates based on available working days
def shift_to_next_working_day(aggregated_sent_df, price_df):
    aggregated_sent_df = aggregated_sent_df.copy()
    aggregated_sent_df.index = to_datetime(aggregated_sent_df.index) # Ensure correct datatype
    price_df.index = to_datetime(price_df.index)

    valid_dates = set(price_df.index)
    adjusted_dates = []

    for date in aggregated_sent_df.index:
        while date not in valid_dates:
            date += Timedelta(days=1)  # Move to the next day
        adjusted_dates.append(date)

    aggregated_sent_df.index = adjusted_dates  

    # Aggregate duplicate dates (if any) after shifting
    aggregated_sent_df = aggregated_sent_df.groupby(aggregated_sent_df.index).mean()

    # Round the aggregated sentiment and convert to int (-1, 0, or 1)
    aggregated_sent_df['aggregated_sentiment'] = custom_round(aggregated_sent_df['aggregated_sentiment'])
    
    return aggregated_sent_df

def custom_round(series):
    """
    Rounds values in the series such that:
    - Values 0.5 and above are rounded up to 1.
    - Values -0.5 and below are rounded down to -1.
    - Values between -0.5 and 0.5 are rounded to 0.
    
    Parameters:
    - series: Pandas Series containing numerical sentiment values.

    Returns:
    - A Pandas Series with the custom-rounded values.
    """
    return where(
        series >= 0.5,
        ceil(series),  
        where(
            series <= -0.5,
            floor(series),
            0 # Otherwise, round to 0
        )
    ).astype(int)


if __name__ == '__main__':
    load_dotenv('../modelling/.llm.env')
    target_stock = getenv('target_stock')
    stock_data_filepath = getenv('stock_data_filepath') + target_stock + '.csv'
    text_type = getenv('text_type')
    sentiment_input_filepath = '../../../data/clean/sentiment_analysis_results/gemini_sentiment_analysis_results_' + text_type + '_' + target_stock + '.csv'
    llm_sentiment_output_filepath = getenv('llm_sentiment_output_filepath') + text_type + '_' + target_stock + '.csv'
    
    lstm_df = read_csv(stock_data_filepath, index_col='Date')
    sentiment_df = read_csv(sentiment_input_filepath)

    # Aggregate first, then adjust dates
    aggregated_sentiment_df = aggregate_by_pub_date(sentiment_df)
    aggregated_sentiment_df_adjusted = shift_to_next_working_day(aggregated_sentiment_df, lstm_df)

    # Reindex to lstm_df
    aggregate_sentiment_df_reindex = aggregated_sentiment_df_adjusted.reindex(lstm_df.index, method=None)
    aggregate_sentiment_df_reindex['aggregated_sentiment'] = aggregate_sentiment_df_reindex['aggregated_sentiment'].fillna(0).astype(int)
    
    aggregate_sentiment_df_reindex.to_csv(llm_sentiment_output_filepath)
