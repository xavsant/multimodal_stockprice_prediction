# Imports
from pandas import read_csv
from dotenv import load_dotenv
from os import getenv

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping

from model_utility_functions import train_test_split, minmax_scale, separate_features_from_target, reshape_X, transform_y, get_results, get_training_plot, get_validation_plot

def create_lstm(X_train_reshaped, y_train, X_test_reshaped, epochs, batch_size, model_name, target_stock, train_model):
    '''Trains LSTM model and gets prediction for test data'''
    # Initialise variables
    history = ''
    path = '../../data/weights/' + model_name + '_' + target_stock + '.weights.h5'

    # Create LSTM model
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')

    if train_model:
        # Model fitting
        callback = EarlyStopping(monitor='loss', mode='min', patience=5, min_delta=1e-4, restore_best_weights=True)
        history = model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=False, callbacks=[callback])

        # Save weights
        model.save_weights(path)

    else:
        # Load weights
        model.load_weights(path)

    # Get predicted values
    yhat = model.predict(X_test_reshaped)

    return history, yhat


if __name__ == '__main__':
    # Initialise variables
    load_dotenv('.lstm.env')

    # Data
    lag_steps = int(getenv('lag_steps'))
    num_features = int(getenv('num_features'))

    # Training
    train_pct = float(getenv('train_pct'))
    epochs = int(getenv('epochs'))
    batch_size = int(getenv('batch_size'))

    # Misc
    model_name = getenv('model_name')
    target_stock = getenv('target_stock')
    stock_data_filepath = getenv('stock_data_filepath') + target_stock + '.csv'
    train_model = bool(int(getenv('train_model')))

    # Preprocessing
    df = read_csv(stock_data_filepath, index_col='Date')
    train, test = train_test_split(df, train_pct)
    train_scaled, test_scaled, scaler = minmax_scale(train, test)
    X_train, y_train, X_test, y_test = separate_features_from_target(train_scaled, test_scaled)
    X_train_reshaped, X_test_reshaped = reshape_X(X_train, X_test, lag_steps)

    # Modelling
    history, yhat = create_lstm(X_train_reshaped, y_train, X_test_reshaped, epochs, batch_size, model_name, target_stock, train_model)
    inv_y, inv_yhat = transform_y(X_test_reshaped, y_test, yhat, scaler, num_features, lag_steps)

    # Results
    get_results(inv_y, inv_yhat, target_stock, model_name)
    if train_model:
        get_training_plot(history, target_stock, model_name) 
    get_validation_plot(test, inv_y, inv_yhat, target_stock, model_name)
