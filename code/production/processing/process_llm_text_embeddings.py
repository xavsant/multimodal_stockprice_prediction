# Imports
import pandas as pd
from pandas import read_json, read_csv
from dotenv import load_dotenv
from os import getenv

if __name__ == '__main__':
    # Load environment variables
    load_dotenv('../modelling/.llm.env')
    text_type = getenv('text_type')
    target_stock = getenv('target_stock')

    # Define file paths
    stock_data_filepath = getenv('stock_data_filepath') + target_stock + '.csv'
    llm_output_filepath = getenv('llm_text_embedding_output_filepath') + text_type + '_' + target_stock + '.csv'

    # Read in JSON and transform
    df_embeddings = read_json('../../../data/raw/gemini_headline_embeddings.json', orient='index')
    df_embeddings.index = pd.to_datetime(df_embeddings.index) # Ensure index is in datetime format
    df_embeddings = df_embeddings.sort_index()
    df_embeddings.columns = [f"emb_{i}" for i in range(df_embeddings.shape[1])] # Rename columns

    # Read in baseline stock data
    lstm_df = read_csv(stock_data_filepath, index_col='Date', parse_dates=True)

    # Reindex embeddings to match stock data
    df_reindex = df_embeddings.reindex(lstm_df.index, method=None)
    mean_embedding = df_embeddings.mean()
    df_processed = df_reindex.fillna(mean_embedding)  # Fill missing embeddings with mean

    # Save processed embeddings
    df_processed.to_csv(llm_output_filepath)

