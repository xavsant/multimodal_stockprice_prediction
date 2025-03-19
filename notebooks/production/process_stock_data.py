# Imports
from pandas import read_csv, DataFrame, concat, to_datetime

def time_series_to_lstm(df, target_variable, lag_steps=1, dropna=True, fill='ffill'):
    """
    Transforms time-series data into a supervised learning format compatible with LSTMs.
    
    Args:
        df (pd.DataFrame): The input time-series dataset.
        target_variable (str): The column to predict.
        lag_steps (int): Number of past time steps to include.
        dropna (bool): Whether to drop rows with NaN values.
        fill (str): What to fill NaN values with ('ffill', 'bfill', 'mean', 'median', None)
        
    Returns:
        df_transformed (pd.DataFrame): DataFrame to be used as input for LSTM model
    """
    # Ensure dataframe format
    if isinstance(df, list):
        df = DataFrame(df)
    
    cols = []
    feature_names = []
    
    # Create Lag Steps
    for i in range(lag_steps, 0, -1):
        cols.append(df.shift(i))
        feature_names += [f"{col}(t-{i})" for col in df.columns]
    
    # Current time step (t) for target variable
    cols.append(df[[target_variable]])
    feature_names += [f"{target_variable}(t)"]
    
    # Combine and assign column names
    df_transformed = concat(cols, axis=1)
    df_transformed.columns = feature_names

    # Drop NaN rows if required
    if dropna:
        df_transformed.dropna(inplace=True)
    else:
        if not None:
            df_transformed.fillna(method=fill, inplace=True)
    
    return df_transformed

if __name__ == '__main__':

    # Initialise variables
    target_stock = 'AAPL'
    selected_stocks = [target_stock]
    lag_steps = 1

    df = read_csv('../../data/raw/djia_stock_data.csv', header=[0,1],index_col=0)
    df.dropna(inplace=True)
    print('Original Shape:', df.shape)

    # Isolate Adj Close and target stock
    df_adjclose = df['Adj Close']
    df_adjclose_target = df_adjclose[selected_stocks]
    df_adjclose_target.index = to_datetime(df_adjclose_target.index)

    df_transformed = time_series_to_lstm(df_adjclose_target, target_stock, lag_steps)
    print('Transformed Shape:', df_transformed.shape)

    # Export transformed dataset
    df_transformed.to_csv('../../data/clean/baseline_transformed_dataset_AAPL.csv')
