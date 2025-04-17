# Imports
from pandas import read_json, read_csv
from dotenv import load_dotenv
from os import getenv

if __name__ == '__main__':
    load_dotenv('../modelling/.llm.env')
    text_type = getenv('text_type')
    target_stock = getenv('target_stock')
    llm_output_filepath = getenv('llm_price_prediction_output_filepath') + text_type + '_' + target_stock + '.csv' # Output path

    # Read in .json and transform
    df = read_json('../../../data/raw/gemini_price_predictions_' + target_stock + '.json', orient='records')
    df.columns = ['Date', 'gemini_predicted_price']
    df.set_index('Date', inplace=True)

    # Read in baseline data
    stock_data_filepath = getenv('stock_data_filepath') + target_stock + '.csv'
    lstm_df = read_csv(stock_data_filepath, index_col='Date')

    # Reindex
    df_reindex = df.reindex(lstm_df.index, method=None)
    df_processed = df_reindex['gemini_predicted_price'].fillna(lstm_df[f'{target_stock}(t-1)']) # Fill with stock t-1 value

    df_processed.to_csv(llm_output_filepath)
