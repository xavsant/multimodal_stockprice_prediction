# Imports
from pandas import read_csv
from dotenv import load_dotenv
from os import getenv

# Function to aggregate sentiment using a voting system
def aggregate_sentiment(row):
    sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    
    # Get the list of sentiment labels
    sentiments = [row['vader_label'], row['drob_label'], row['deb_label']]
    
    # Count occurrences of each sentiment
    sentiment_counts = {'Positive': sentiments.count('Positive'), 
                        'Neutral': sentiments.count('Neutral'), 
                        'Negative': sentiments.count('Negative')}
    
    # Choose the sentiment with the highest count, resolving ties by choosing the first
    majority_sentiment = max(sentiment_counts, key=sentiment_counts.get)

    return sentiment_map[majority_sentiment]

# Function to aggregate sentiments by pub_date with rounding
def aggregate_by_pub_date(sent_df):
    # Apply the aggregation function to the entire dataframe
    sent_df['aggregated_sentiment'] = sent_df.apply(aggregate_sentiment, axis=1)
    
    # Group by pub_date and calculate the average sentiment per day
    aggregated_sent_df = sent_df.groupby('pub_date')['aggregated_sentiment'].mean().to_frame()
    
    # Round the aggregated sentiment and convert to int (-1, 0, or 1)
    aggregated_sent_df['aggregated_sentiment'] = aggregated_sent_df['aggregated_sentiment'].round().astype(int)
    
    # Set 'pub_date' as the index of the result
    aggregated_sent_df.reset_index(inplace=True)
    aggregated_sent_df.set_index('pub_date', inplace=True)
    
    return aggregated_sent_df

if __name__ == '__main__':
    # Initialise variables
    load_dotenv('.concat.env')
    target_stock = getenv('target_stock')
    stock_data_filepath = getenv('stock_data_filepath') + '_' + target_stock + '.csv'
    sentiment_analysis_filepath = getenv('sentiment_analysis_filepath') + '_' + target_stock + '.csv'
    input_filepath = '../../data/clean/sentiment_analysis/sentiment_analysis_headlines_AAPL.csv'

    lstm_df = read_csv(stock_data_filepath, index_col='Date')
    sentiment_df = read_csv(input_filepath)
    sentiment_df.drop('ticker', axis=1, inplace=True)
    aggregate_sentiment_df = aggregate_by_pub_date(sentiment_df) # Aggregate the sentiment by pub_date
    aggregate_sentiment_df_reindex = aggregate_sentiment_df.reindex(lstm_df.index, method=None)
    aggregate_sentiment_df_reindex['aggregated_sentiment'] = aggregate_sentiment_df_reindex['aggregated_sentiment'].fillna(0).astype(int)

    # Export transformed dataset
    aggregate_sentiment_df_reindex.to_csv(sentiment_analysis_filepath)
