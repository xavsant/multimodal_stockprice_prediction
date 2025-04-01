# Imports
from pandas import read_csv
from dotenv import load_dotenv
from os import getenv, path
import sys

from keras.models import Model
from keras.layers import Dense, LSTM, Input, LeakyReLU
from keras.callbacks import EarlyStopping

sys.path.append(path.abspath(path.join(path.dirname(__file__), '..', 'utility'))) # Quick-fix to access utility functions
from model_utility_functions import train_test_split, minmax_scale, separate_features_from_target, reshape_X, transform_y, iterator_results, iterator_average_results, update_best_results, get_training_plot, get_validation_plot

def create_lstm_model(X_train_reshaped):
    lstm_input = Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), name="stock_input")
    lstm_hidden = LSTM(96, activation="tanh", name="lstm_layer")(lstm_input)
    lstm_dense = Dense(32, activation=LeakyReLU(negative_slope=0.01), name="lstm_dense")(lstm_hidden)
    output = Dense(1, name="output_layer")(lstm_dense)
    
    model = Model(inputs=lstm_input, outputs=output)
    model.compile(optimizer='adam', loss='mae')

    return model

def train_test_lstm_model(test, X_train_reshaped, y_train, X_test_reshaped, epochs, batch_size, model_name, target_stock, train_model, y_test, scaler, iterations):
    # Initialise variables
    history = ''
    mae = float('inf')
    mse = float('inf')
    best_index = 0
    best_mse = float('inf')
    path = '../../../data/weights/' + model_name + '_' + target_stock + '.weights.h5'
    model_dict = {'model':[], 'history':[], 'inv_yhat': [], 'mae': [], 'mse': []}

    if train_model:
        for i in range(iterations):
            print(f'Model Training Iteration {str(i)}')
            # Create model
            model = create_lstm_model(X_train_reshaped)

            # Model fitting
            callback = EarlyStopping(monitor='loss', mode='min', patience=5, min_delta=1e-3, restore_best_weights=True)
            history = model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=False, callbacks=[callback])
            
            # Get predicted values
            yhat = model.predict(X_test_reshaped)
            inv_y, inv_yhat = transform_y(X_test_reshaped, y_test, yhat, scaler)
            mae, mse = iterator_results(inv_y, inv_yhat, target_stock, model_name, iteration=i)

            if mse < best_mse:
                best_index = i
                best_mse = mse

            # Add to dictionary
            model_dict['model'].append(model)
            model_dict['history'].append(history)
            model_dict['inv_yhat'].append(inv_yhat)
            model_dict['mae'].append(mae)
            model_dict['mse'].append(mse)

        best_model = model_dict['model'][best_index]
        history = model_dict['history'][best_index]
        inv_yhat = model_dict['inv_yhat'][best_index]
        mae = model_dict['mae'][best_index]
        mse = model_dict['mse'][best_index]
        iterator_average_results(model_dict['mae'], model_dict['mse'], target_stock, model_name)

        # Export key outputs
        best_model.save_weights(path)
        update_best_results(mae, mse, target_stock, model_name)
        get_training_plot(history, target_stock, model_name) 
        get_validation_plot(test, inv_y, inv_yhat, target_stock, model_name)

    else:
        # Load weights
        # model.load_weights(path)
        print("Variable 'train_model' set to 0, to re-train, set to 1 in .lstm.env")

    return history, mae, mse


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
    iterations = int(getenv('iterations'))

    # Misc
    model_name = getenv('model_name')
    target_stock = getenv('target_stock')
    stock_data_filepath = getenv('stock_data_filepath') + target_stock + '.csv'
    train_model = bool(int(getenv('train_model')))

    # Preprocessing
    df = read_csv(stock_data_filepath, index_col='Date')
    train, test = train_test_split(df, train_pct)
    train_array = train.to_numpy()
    test_array = test.to_numpy()

    train_scaled, test_scaled, scaler = minmax_scale(train_array, test_array)
    X_train, y_train, X_test, y_test = separate_features_from_target(train_scaled, test_scaled)

    X_train_reshaped, X_test_reshaped = reshape_X(X_train, X_test, num_features)

    # Modelling
    train_test_lstm_model(test, X_train_reshaped, y_train, X_test_reshaped, epochs, batch_size, model_name, target_stock, train_model, y_test, scaler, iterations)

    

