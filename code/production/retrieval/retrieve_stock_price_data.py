# Imports
import yfinance as yf
import datetime as dt

if __name__ == '__main__':
    stock_ticks_list = [
    'MMM', 'AXP', 'AMGN', 'AMZN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO',
    'DIS', 'GS', 'HD', 'HON', 'IBM', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT',
    'NKE', 'NVDA', 'PG', 'CRM', 'SHW', 'TRV', 'UNH', 'VZ', 'V', 'WMT'
    ]
    
    djia_begin_date = dt.date(2015, 1, 1)
    djia_end_date = dt.date(2024, 12, 31)

    output_path = '../../../data/raw/djia_stock_data.csv'

    df = yf.download(stock_ticks_list, djia_begin_date, djia_end_date, auto_adjust=False)
    df.to_csv(output_path)