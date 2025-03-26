# Imports
from pandas import read_csv
from dotenv import load_dotenv
from os import getenv

from keras.models import Model
from keras.layers import Dense, LSTM, Input, Concatenate
from keras.callbacks import EarlyStopping

from model_utility_functions import train_test_split, minmax_scale, separate_features_from_target, reshape_X, transform_y, get_results, get_training_plot, get_validation_plot

def create_concat_model(X_train_reshaped, y_train, X_test_reshaped, sentiment_train, sentiment_test, epochs, batch_size, train_model, target_stock, model_name):
    '''Trains Concat model and gets prediction for test data'''
    # Initialise variables
    history = ''
    lstm_path = f'../../data/weights/baseline_lstm_{target_stock}.weights.h5'
    concat_path = '../../data/weights/' + model_name + '_' + target_stock + '.weights.h5'

    # Create LSTM modality
    lstm_input = Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), name="stock_input")
    lstm_output = LSTM(100, name="lstm_layer")(lstm_input)  # Keep structure the same

    ## Load weights
    lstm_model = Model(inputs=lstm_input, outputs=lstm_output)
    lstm_model.load_weights(lstm_path)

    ## Add density layer
    lstm_dense = Dense(32, activation="relu")(lstm_model.output)

    # Create Sentiment modality
    sentiment_input = Input(shape=(sentiment_train.shape[1],))

    ## Add density layer
    sentiment_dense = Dense(8, activation='relu')(sentiment_input)

    # Concatenate modalities
    concatenated = Concatenate(name="concatenation_layer")([lstm_dense, sentiment_dense])

    ## Add final density layers
    dense_layer = Dense(16, activation='relu', name="dense_final")(concatenated)
    output = Dense(1, name="output_layer")(dense_layer)
    model = Model(inputs=[lstm_input, sentiment_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')

    if train_model:
        callback = EarlyStopping(monitor='loss', mode='min', patience=5, min_delta=1e-4, restore_best_weights=True)
        history = model.fit([X_train, sentiment_train], y_train, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=False, callbacks=[callback])
        
        # Save weights
        model.save_weights(concat_path)

    else:
        # Load weights
        model.load_weights(concat_path)

    # Get predicted values
    yhat = model.predict([X_test_reshaped, sentiment_test])

    return history, yhat


if __name__ == '__main__':
    # Initialise variables
    load_dotenv('.concat.env')

    # Data
    lag_steps = int(getenv('lag_steps'))
    num_features = int(getenv('num_features'))

    # Training
    train_pct = float(getenv('train_pct'))
    epochs = int(getenv('epochs'))
    batch_size = int(getenv('batch_size'))

    # Name Variables
    model_name = getenv('model_name')
    target_stock = getenv('target_stock')
    text_type = getenv('text_type')
    detailed_model_name = model_name + '_' + text_type

    # Filepaths
    stock_data_filepath = getenv('stock_data_filepath') + target_stock + '.csv'
    sentiment_analysis_filepath = getenv('sentiment_analysis_filepath') + text_type + '_' + target_stock + '.csv'

    # Train model?
    train_model = bool(int(getenv('train_model')))

    # Preprocessing Stock Price Data
    lstm_df = read_csv(stock_data_filepath, index_col='Date')
    train, test = train_test_split(lstm_df, train_pct)
    train_scaled, test_scaled, scaler = minmax_scale(train, test)
    X_train, y_train, X_test, y_test = separate_features_from_target(train_scaled, test_scaled)
    X_train_reshaped, X_test_reshaped = reshape_X(X_train, X_test, lag_steps)

    # Preprocessing Sentiment Analysis Data
    sentiment_df = read_csv(sentiment_analysis_filepath, index_col='Date')
    sentiment_train, sentiment_test = train_test_split(sentiment_df, train_pct)

    # Modelling
    history, yhat = create_concat_model(X_train_reshaped, y_train, X_test_reshaped, sentiment_train, 
                                        sentiment_test, epochs, batch_size, train_model, target_stock, detailed_model_name)
    inv_y, inv_yhat = transform_y(X_test_reshaped, y_test, yhat, scaler, num_features, lag_steps)

    # Results
    get_results(inv_y, inv_yhat, target_stock, detailed_model_name)
    if train_model:
        get_training_plot(history, target_stock, detailed_model_name) 
    get_validation_plot(test, inv_y, inv_yhat, target_stock, detailed_model_name)
