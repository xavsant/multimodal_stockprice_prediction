# Imports
from pandas import read_csv
from dotenv import load_dotenv
from os import getenv, path
import sys

from keras.models import Model
from keras.layers import Dense, LSTM, Input, Concatenate, RepeatVector
from keras.callbacks import EarlyStopping

sys.path.append(path.abspath(path.join(path.dirname(__file__), '..', 'utility'))) # Quick-fix to access utility functions
from model_utility_functions import train_test_split, minmax_scale, separate_features_from_target, reshape_X, transform_y, iterator_results, iterator_average_results, update_best_results, get_training_plot, get_validation_plot

def create_concat_model(X_train_reshaped, text_train, target_stock):
    lstm_path = f'../../../data/weights/baseline_lstm_{target_stock}.weights.h5'

    # Create LSTM modality
    lstm_input = Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), name="stock_input")
    lstm_hidden = LSTM(128, name="lstm_layer")(lstm_input)
    lstm_dense = Dense(32, activation="relu", name="lstm_dense")(lstm_hidden)

    ## Load weights
    lstm_model = Model(inputs=lstm_input, outputs=lstm_dense)
    lstm_model.load_weights(lstm_path)

    for layer in lstm_model.layers:
        layer.trainable = False

    # Create text modality
    text_input = Input(shape=(text_train.shape[1],))
    text_dense = Dense(16, activation='relu')(text_input)

    # Concatenate modalities
    concatenated = Concatenate(name="concatenation_layer")([lstm_dense, text_dense])

    ## Add final density layers
    dense_layer = Dense(32, activation='relu', name="dense_final")(concatenated)
    output = Dense(1, name="output_layer")(dense_layer)
    
    model = Model(inputs=[lstm_input, text_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def create_concat_model(X_train_reshaped, text_train, target_stock):
    # Create LSTM modality (stock input)
    lstm_input = Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), name="stock_input")
    
    # Create text modality (text input)
    text_input = Input(shape=(text_train.shape[1],), name="text_input")
    
    # Reshape the text input using RepeatVector to match the sequence length of stock input
    repeated_text = RepeatVector(X_train_reshaped.shape[1])(text_input)  # Repeat the text input to match stock sequence length
    
    # Concatenate the stock input and reshaped text input along the feature axis (axis=-1)
    merged_input = Concatenate(axis=-1, name="merged_input")([lstm_input, repeated_text])

    # Pass the concatenated features through an LSTM layer
    lstm_hidden = LSTM(128, return_sequences=False, name="lstm_layer")(merged_input)
    lstm_dense = Dense(32, activation="relu", name="lstm_dense")(lstm_hidden)

    # Final dense layer to make the prediction
    output = Dense(1, name="output_layer")(lstm_dense)

    # Define the model
    model = Model(inputs=[lstm_input, text_input], outputs=output)
    model.compile(optimizer="adam", loss="mean_squared_error")

    return model

# from tensorflow.keras.layers import Input, LSTM, RepeatVector, Concatenate, Dense, Multiply
# from tensorflow.keras.models import Model
# import tensorflow as tf

# # Define the Gated Attention Layer
# class GatedAttention(tf.keras.layers.Layer):
#     def __init__(self, units):
#         super(GatedAttention, self).__init__()
#         self.units = units

#     def build(self, input_shape):
#         lstm_shape, text_shape = input_shape
#         self.W = self.add_weight(shape=(lstm_shape[-1], self.units), initializer="random_normal", name="attention_weight")
#         self.b = self.add_weight(shape=(self.units,), initializer="zeros", name="attention_bias")
#         self.gate_weights = self.add_weight(shape=(text_shape[-1], self.units), initializer="random_normal", name="gate_weight")

#     def call(self, inputs):
#         lstm_output, text_input = inputs
#         # Compute attention scores
#         gate = tf.nn.sigmoid(tf.matmul(text_input, self.gate_weights))  # Gate value between 0 and 1 (adjusted for text input)
#         attention_scores = tf.matmul(lstm_output, self.W) + self.b
#         attention_scores = tf.nn.tanh(attention_scores)  # Apply tanh activation to the scores
#         attention_weights = tf.nn.softmax(attention_scores, axis=1)  # Compute attention weights

#         # Apply the gate to attention weights
#         gated_attention_weights = attention_weights * gate  # Element-wise multiplication

#         # Weighted sum of inputs based on attention
#         gated_output = Multiply()([lstm_output, gated_attention_weights])  # Apply attention to the LSTM output

#         return gated_output

# # Create the model with gated attention
# def create_concat_model(X_train_reshaped, text_train, target_stock):
#     # Create LSTM modality (stock input)
#     lstm_input = Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), name="stock_input")
    
#     # Create text modality (text input)
#     text_input = Input(shape=(text_train.shape[1],), name="text_input")
    
#     # Reshape the text input using RepeatVector to match the sequence length of stock input
#     repeated_text = RepeatVector(X_train_reshaped.shape[1])(text_input)  # Repeat the text input to match stock sequence length
    
#     # Concatenate the stock input and reshaped text input along the feature axis (axis=-1)
#     merged_input = Concatenate(axis=-1, name="merged_input")([lstm_input, repeated_text])
    
#     # Pass the concatenated features through an LSTM layer
#     lstm_hidden = LSTM(128, return_sequences=True, name="lstm_layer")(merged_input)  # Keep return_sequences=True for attention
    
#     # Apply Gated Attention Mechanism
#     gated_attention_output = GatedAttention(128)([lstm_hidden, repeated_text])  # Use 128 units to match LSTM output size
    
#     # # Dense layer after attention mechanism
#     lstm_dense = Dense(32, activation="relu", name="lstm_dense")(gated_attention_output)
    
#     # Final output layer
#     output = Dense(1, name="output_layer")(lstm_dense)

#     # Define the model
#     model = Model(inputs=[lstm_input, text_input], outputs=output)
#     model.compile(optimizer="adam", loss="mean_squared_error")

#     return model




def train_test_concat_model(X_train_reshaped, y_train, X_test_reshaped, text_train, text_test, epochs, batch_size, train_model, target_stock, model_name, iterations):
    # Initialise variables
    history = ''
    mae = float('inf')
    mse = float('inf')
    best_index = 0
    best_mse = float('inf')
    concat_path = '../../../data/weights/' + model_name + '_' + target_stock + '.weights.h5'
    model_dict = {'model':[], 'history':[], 'inv_yhat': [], 'mae': [], 'mse': []}
    
    if train_model:
        for i in range(iterations):
            print(f'Model Training Iteration {str(i)}')
            # Create model
            model = create_concat_model(X_train_reshaped, text_train, target_stock)

            # Model fitting
            callback = EarlyStopping(monitor='loss', mode='min', patience=5, min_delta=1e-4, restore_best_weights=True)
            history = model.fit([X_train, text_train], y_train, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=False, callbacks=[callback])
            
            # Get predicted values
            yhat = model.predict([X_test_reshaped, text_test])
            inv_y, inv_yhat = transform_y(X_test_reshaped, y_test, yhat, scaler, num_features, lag_steps)
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
        best_model.save_weights(concat_path)
        update_best_results(mae, mse, target_stock, model_name)
        get_training_plot(history, target_stock, model_name) 
        get_validation_plot(test, inv_y, inv_yhat, target_stock, model_name)

    else:
        # Load weights
        # model.load_weights(path)
        print("Variable 'train_model' set to 0, to re-train, set to 1 in .concat.env")

    return history, mae, mse


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
    iterations = int(getenv('iterations'))

    # Name Variables
    model_name = getenv('model_name')
    target_stock = getenv('target_stock')
    text_type = getenv('text_type')
    detailed_model_name = model_name + '_' + text_type

    # Filepaths
    stock_data_filepath = getenv('stock_data_filepath') + target_stock + '.csv'
    text_analysis_filepath = getenv('text_analysis_filepath') + text_type + '_' + target_stock + '.csv'

    # Train model?
    train_model = bool(int(getenv('train_model')))

    # Preprocessing Stock Price Data
    lstm_df = read_csv(stock_data_filepath, index_col='Date')
    train, test = train_test_split(lstm_df, train_pct)
    train_scaled, test_scaled, scaler = minmax_scale(train, test)
    X_train, y_train, X_test, y_test = separate_features_from_target(train_scaled, test_scaled)
    X_train_reshaped, X_test_reshaped = reshape_X(X_train, X_test, lag_steps)

    # Preprocessing Text Analysis Data
    text_df = read_csv(text_analysis_filepath, index_col='Date')
    text_train, text_test = train_test_split(text_df, train_pct)

    # Modelling
    train_test_concat_model(X_train_reshaped, y_train, X_test_reshaped, text_train, text_test, epochs, 
                            batch_size, train_model, target_stock, detailed_model_name, iterations)
