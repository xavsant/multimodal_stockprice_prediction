# Imports
from pandas import read_csv, DataFrame

class model_errors:
    def __init__(self, target_stock):
        self.filepath = f'../../../data/model_results/model_errors_{target_stock}.csv'
    
    def initialise(self):
        dummy_data = {'model': [], 'mae': [], 'mse': []}
        self.model_errors = DataFrame(dummy_data)
        self._export()

    def update(self, model: str, mae: float, mse: float ):
        self._read()
        new_data = {'model': model, 'mae': mae, 'mse': mse}

        if model in list(self.model_errors.loc[:, 'model']):
            self.model_errors.set_index('model', inplace=True) # Set index to update easily
            del new_data['model']
            self.model_errors.loc[model] = new_data
            self.model_errors.reset_index(inplace=True) 

        else:    
            self.model_errors.loc[len(self.model_errors.index)] = new_data

        self._export()

    def _read(self):
        self.model_errors = read_csv(self.filepath)

    def _export(self):
        self.model_errors.to_csv(self.filepath, index=False)

if __name__ == '__main__':
    # For experimentation, be careful as it will replace the existing file
    model_errors = model_errors('AAPL')

    # Initialise .csv for the first time
    # model_errors.initialise()

    # Update .csv
    # model_errors.update('LSTM', 1.2, 1.44)
