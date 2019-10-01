import pandas as pd
from pandas_datareader import data
import datetime as dt

# Purpose to extract stock data from yahoo
def extract_stock_data(stock_symbols, start_date, end_date=dt.datetime.today().date()):
    all_stock_data = pd.DataFrame()
    for stock in stock_symbols:
        print('Extracting data for:', stock)
        try:
            stock_data = data.DataReader(stock, 'yahoo', start_date, end_date)
            stock_data['ticker'] = stock
            all_stock_data = pd.concat([all_stock_data, stock_data], axis=0)
        except:
            print(stock, 'didnt work')
            continue
            
    return all_stock_data


# Purpose is to get moving average of 'window_size' of 'ma_column' based on the 'split_column'
def multiple_moving_averages(df, ma_column, split_column, window_size):
    all_stock_data = pd.DataFrame()
    for stock in df[split_column].unique():
        stock_data = df[df[split_column]==stock].sort_values(by='Date')
        all_stock_data = pd.concat(
            [all_stock_data.reset_index(drop=True), 
             moving_average(stock_data, ma_column, window_size)],
            axis=0)
    return all_stock_data


# Purpose is to get moving average of 'window_size' of 'ma_column'
def moving_average(df, ma_column, window_size):
    ma_df = pd.DataFrame(
                (df[ma_column]
                       .rolling(window=window_size)
                       .mean()
                       .reset_index()[ma_column]))
    ma_df.columns = [ma_column + '_ma' + str(window_size)]
    
    df = pd.concat([df.reset_index(drop=True), ma_df], axis=1)
    return df