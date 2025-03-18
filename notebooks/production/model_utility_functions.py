# Imports
from numpy import concatenate
from pandas import to_datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from save_model_errors import model_errors

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

def transform_y(X_test_reshaped, y_test, yhat, scaler, num_features, lag_steps):
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

def get_results(inv_y, inv_yhat, target_stock, model_name):
    # Get Test Errors
    mae = mean_absolute_error(inv_y, inv_yhat)
    print('LSTM Test MAE: %.3f' % mae)

    mse = mean_squared_error(inv_y, inv_yhat)
    print('LSTM Test MSE: %.3f' % mse)

    # Export Test Errors
    model_errors_instance = model_errors(target_stock)
    model_errors_instance.update(model_name, mae, mse)

def get_training_plot(history, target_stock, model_name):
    # Plot training progression
    plt.plot(history.history['loss'])
    plt.title(f'Training Loss for {model_name} for {target_stock} Stock Price')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('../../plots/final/history_' + model_name + '_' + target_stock)

def get_validation_plot(test, inv_y, inv_yhat, target_stock, model_name):
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
    plt.title(f'Actual vs Predicted using {model_name} for {target_stock} Stock Price')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('../../plots/final/validation_' + model_name + '_' + target_stock)
