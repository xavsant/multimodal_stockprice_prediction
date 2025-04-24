# Imports
from pandas import read_csv
from dotenv import load_dotenv
from os import getenv, path
import sys

from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append(path.abspath(path.join(path.dirname(__file__), '..', 'utility'))) # Quick-fix to access utility functions
from model_utility_functions import train_test_split, update_best_results, get_validation_plot

if __name__ == '__main__':
    # Initialise variables
    load_dotenv('.llm.env')

    # Training
    train_pct = float(getenv('train_pct'))

    # Name Variables
    model_name = getenv('model_name')
    target_stock = [getenv('target_stock')]
    text_type = getenv('text_type')
    detailed_model_name = model_name + '_' + text_type

    target_stock = ['AAPL', 'AMZN', 'CRM', 'IBM', 'MSFT', 'NVDA']

    for t in target_stock:

        # Filepaths
        stock_data_filepath = getenv('stock_data_filepath') + t + '.csv'
        llm_price_prediction_filepath = getenv('llm_price_prediction_output_filepath') + text_type + '_' + t + '.csv'
        
        # Preprocessing Stock Price Data
        lstm_df = read_csv(stock_data_filepath, index_col='Date')
        lstm_train, lstm_test = train_test_split(lstm_df, train_pct)
        y_test = lstm_test.iloc[:, -1]

        # Preprocessing Text Analysis Data
        llm_df = read_csv(llm_price_prediction_filepath, index_col='Date')
        llm_train, llm_test = train_test_split(llm_df, train_pct)
        yhat = llm_test.iloc[:, -1]

        # Results
        mae = mean_absolute_error(y_test, yhat)
        mse = mean_squared_error(y_test, yhat)
        update_best_results(mae, mse, t, detailed_model_name)
        get_validation_plot(y_test, y_test, yhat, t, detailed_model_name)
