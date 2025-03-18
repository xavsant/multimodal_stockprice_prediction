# Imports
import numpy as np
from numpy import concatenate
from pandas import read_csv, to_datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from os import getenv
from dotenv import load_dotenv

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM


def train_test_split(df, train_pct):
    split = int(df.shape[0]*train_pct)
    train = df[:split]
    test = df[split:]
    # print('Split Shape:', train.shape, test.shape)

    return train, test

def minmax_scale(train, test):
    '''Scale to avoid distance calculation bias'''
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test) # Avoid data leakage

    return train_scaled, test_scaled, scaler

def separate_features_from_target(train_scaled, test_scaled):
    '''Separate into features and target (last column)'''
    X_train, y_train = train_scaled[:, :-1], train_scaled[:, -1]
    X_test, y_test = test_scaled[:, :-1], test_scaled[:, -1]

    return X_train, y_train, X_test, y_test

def reshape_X(X_train, X_test, lag_steps):
    X_train_reshaped = X_train.reshape((X_train.shape[0], lag_steps, X_train.shape[1]))
    X_test_reshaped = X_test.reshape((X_test.shape[0], lag_steps, X_test.shape[1]))

    return X_train_reshaped, X_test_reshaped

def create_lstm(X_train_reshaped, y_train, X_test_reshaped, epochs, batch_size, retrain_model):
    '''Trains LSTM model and gets prediction for test data'''
    # Initialise variables
    history = ''
    path = '../weights/' + model_name + '_' + target_stock + '.weights.h5'

    # Create LSTM model
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    if retrain_model:
        # Model fitting
        history = model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=False)

        # Save weights
        model.save_weights(path)

    else:
        # Load weights
        model.load_weights(path)

    # Get predicted values
    yhat = model.predict(X_test_reshaped)

    return history, yhat

def transform_yhat(X_test_reshaped, y_test, yhat, scaler, num_features, lag_steps):
    # Reshaping back into 2D for inverse scaling
    X_test_inv = X_test_reshaped.reshape((X_test_reshaped.shape[0], X_test_reshaped.shape[2])) 

    # Concatenate and Inverse Scaling
    # Validation
    y_test_inv = y_test.reshape((len(y_test), 1))
    inv_y = concatenate((X_test_inv, y_test_inv), axis=1) # Both arrays must have same dimensions
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, num_features*lag_steps]

    # Prediction
    inv_yhat = concatenate((X_test_inv, yhat), axis=1) # Required to get back original scale
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, num_features*lag_steps] # Extract target_variable

    return inv_y, inv_yhat

def get_results(inv_y, inv_yhat):
    # Get Test Errors
    mae = np.sqrt(mean_absolute_error(inv_y, inv_yhat))
    print('LSTM Test MAE: %.3f' % mae)

    mse = mean_squared_error(inv_y, inv_yhat)
    print('LSTM Test MSE: %.3f' % mse)

def get_training_plot(history):
    # Plot training progression
    plt.plot(history.history['loss'])
    plt.title(f'Training Loss for LSTM Model for {target_stock} stock')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('../../plots/final/history_' + model_name + '_' + target_stock)

def get_validation_plot(test, inv_y, inv_yhat):
    # Convert index to datetime
    test.index = to_datetime(test.index)

    # Validation plot
    plt.figure(figsize=(10, 6))
    plt.plot(test.index, inv_y, label='Actual')
    plt.plot(test.index, inv_yhat, label='Predicted')

    # Format the x-axis to show quarterly ticks
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator((1,4,7,10)))  # Quarterly ticks
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format as Year-Quarter

    # Labels and title
    plt.title(f'Actual vs Predicted using LSTM Model for {target_stock} Stock Price')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('../../plots/final/validation_' + model_name + '_' + target_stock)


if __name__ == '__main__':
    # Initialise variables
    load_dotenv('.lstm.env')
    lag_steps = int(getenv('lag_steps'))
    num_features = int(getenv('num_features'))
    train_pct = float(getenv('train_pct'))
    epochs = int(getenv('epochs'))
    batch_size = int(getenv('batch_size'))
    stock_data_filepath = getenv('stock_data_filepath')
    model_name = getenv('model_name')
    target_stock = getenv('target_stock')
    retrain_model = bool(int(getenv('retrain_model')))

    # Preprocessing
    df = read_csv(stock_data_filepath, index_col='Date')
    train, test = train_test_split(df, train_pct)
    train_scaled, test_scaled, scaler = minmax_scale(train, test)
    X_train, y_train, X_test, y_test = separate_features_from_target(train_scaled, test_scaled)
    X_train_reshaped, X_test_reshaped = reshape_X(X_train, X_test, lag_steps)

    # Modelling
    history, yhat = create_lstm(X_train_reshaped, y_train, X_test_reshaped, epochs, batch_size, retrain_model)
    inv_y, inv_yhat = transform_yhat(X_test_reshaped, y_test, yhat, scaler, num_features, lag_steps)

    # Results
    get_results(inv_y, inv_yhat)
    if retrain_model:
        get_training_plot(history) 
    get_validation_plot(test, inv_y, inv_yhat)
