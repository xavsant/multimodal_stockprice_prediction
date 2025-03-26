# To Do: Account for aggregate redundancies

# Imports
from pandas import read_csv, to_datetime, Timedelta
from numpy import where, ceil, floor
from dotenv import load_dotenv
from os import getenv

# Function to aggregate sentiment using a voting system
def aggregate_sentiment(row):
    sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    
    sentiments = [row['vader_label'], row['drob_label'], row['deb_label']]
    
    sentiment_counts = {'Positive': sentiments.count('Positive'), 
                        'Neutral': sentiments.count('Neutral'), 
                        'Negative': sentiments.count('Negative')}
    
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
def shift_to_next_working_day(aggregated_sent_df, lstm_df):
    aggregated_sent_df = aggregated_sent_df.copy()
    aggregated_sent_df.index = to_datetime(aggregated_sent_df.index)
    lstm_df.index = to_datetime(lstm_df.index)

    valid_dates = set(lstm_df.index)
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
    load_dotenv('.concat.env')
    target_stock = getenv('target_stock')
    stock_data_filepath = getenv('stock_data_filepath') + '_' + target_stock + '.csv'
    sentiment_analysis_filepath = getenv('sentiment_analysis_filepath') + '_' + target_stock + '.csv'
    input_filepath = '../../data/clean/sentiment_analysis/sentiment_analysis_headlines_AAPL.csv'

    lstm_df = read_csv(stock_data_filepath, index_col='Date')
    sentiment_df = read_csv(input_filepath)
    sentiment_df.drop('ticker', axis=1, inplace=True)

    # Aggregate first, then adjust dates
    aggregated_sentiment_df = aggregate_by_pub_date(sentiment_df)
    aggregated_sentiment_df_adjusted = shift_to_next_working_day(aggregated_sentiment_df, lstm_df)

    # Reindex to lstm_df
    aggregate_sentiment_df_reindex = aggregated_sentiment_df_adjusted.reindex(lstm_df.index, method=None)
    aggregate_sentiment_df_reindex['aggregated_sentiment'] = aggregate_sentiment_df_reindex['aggregated_sentiment'].fillna(0).astype(int)

    
    aggregate_sentiment_df_reindex.to_csv(sentiment_analysis_filepath)
