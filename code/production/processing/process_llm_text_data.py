# Imports
from pandas import DataFrame, read_csv, to_datetime, Timedelta
from json import dump
from dotenv import load_dotenv
from os import getenv

# Function to preprocess text data
def preprocess_text_data(text_df, variable='headline'):
    text_df_processed = DataFrame(text_df['pub_date'])
    text_df_processed[variable] = text_df[variable]

    text_df_processed['pub_date'] = to_datetime(text_df['pub_date']).dt.date # Remove time element
    text_df_processed = text_df_processed.sort_values(by='pub_date').reset_index(drop=True)

    return text_df_processed

# Function to adjust dates based on available working days
def shift_to_next_working_day(text_df, price_df):
    text_df = text_df.copy()

    # Ensure correct datatype 
    text_df_dates = to_datetime(text_df['pub_date'])
    price_df.index = to_datetime(price_df.index) 

    valid_dates = set(price_df.index)
    adjusted_dates = []

    for date in text_df_dates:
        while date not in valid_dates:
            date += Timedelta(days=1)  # Move to the next day
        adjusted_dates.append(date)

    text_df['pub_date'] = adjusted_dates  

    return text_df

# Function to dictionary that contains all articles for a given date, across all dates
def create_text_dict(text_df_shifted, price_df):
    text_dict = {d.strftime('%Y-%m-%d'):[] for d in price_df.index}

    for date in text_dict.keys():
        content = list(text_df_shifted[text_df_shifted['pub_date'] == date]['headline'])
        for c in content:
            text_dict[date].append(c)
        
    return text_dict

# Function to dump dictionary content into .json
def to_json(text_dict, path):
    with open(path, "w") as outfile:
        dump(text_dict, outfile)


if __name__ == '__main__':
    load_dotenv('../modelling/.llm.env')
    text_type = getenv('text_type')
    target_stock = getenv('target_stock')
    stock_data_filepath = getenv('stock_data_filepath') + target_stock + '.csv'
    llm_input_filepath = getenv('llm_input_filepath')
    llm_output_filepath = getenv('llm_output_filepath') + text_type + '_' + target_stock + '.json' # Output path
    
    price_df = read_csv(stock_data_filepath, index_col='Date')
    text_df = read_csv(llm_input_filepath)

    text_df_processed = preprocess_text_data(text_df, variable=text_type)
    text_df_shifted = shift_to_next_working_day(text_df_processed, price_df)
    text_dict = create_text_dict(text_df_shifted, price_df)

    to_json(text_dict, llm_output_filepath)