import csv
import numpy as np
# import datetime as dt
import datetime as dt
from datetime import timedelta
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import time

def read_csv(file_name, symbol_pos, name_pos):
    read_file = open(file_name + '.csv', 'r', encoding='latin1')
    csv_read = csv.reader(read_file)

    symbols = []
    firstline = True
    for row in read_file:
        if firstline:
            firstline = False
            continue
        symbols.append(row.strip())

    return symbols

def moving_average(x, y, window_size):

    window = np.ones(int(window_size)) / float(window_size)
    rel_y = y[window_size:]
    rel_x = x[window_size:]
    moving_avg = np.convolve(y, window, 'valid')
    moving_avg = moving_avg[:len(moving_avg) - 1]
    # print('length of rel_y, rel_x and moving average arrays are ', len(rel_y), len(rel_x), len(moving_avg))
    return rel_x, rel_y, moving_avg

def get_roll_anomaly(stock_name, rel_x, rel_y, avg, window_size, sigma):

    rel_rel_x = rel_x[window_size:]
    rel_rel_y = rel_y[window_size:]
    rel_avg = avg[window_size:]

    residual_np = rel_y - avg
    residual_np = np.absolute(residual_np)
    residual = pd.DataFrame(residual_np)
    roll_std = residual.rolling(window=window_size,center=False).std()
    roll_std = roll_std[window_size-1:len(roll_std)-1]
    roll_std = np.array(roll_std)

    # print('The length of rel_rel_x, rel_rel_y, roll_std and rel_avg are', len(rel_rel_x), len(rel_rel_y), len(roll_std), len(rel_avg))

    roll_anomaly_list = []
    roll_all_data = []
    for i in range(0, len(rel_rel_y) - 1, 1):
        if (rel_rel_y[i] > rel_avg[i] + roll_std[i]*sigma):
            roll_anomaly_list.append([stock_name, rel_rel_x[i].date(), rel_rel_y[i], float(roll_std[i]), float(rel_avg[i] + roll_std[i]*sigma), float(rel_avg[i] - roll_std[i]*sigma),'High'])
            roll_all_data.append([stock_name, rel_rel_x[i], rel_rel_y[i], float(roll_std[i]),float(rel_avg[i] + roll_std[i]*sigma), float(rel_avg[i] - roll_std[i]*sigma),'High'])
        elif (rel_rel_y[i] < rel_avg[i] - roll_std[i]*sigma):
            roll_anomaly_list.append([stock_name, rel_rel_x[i].date(), rel_rel_y[i], float(roll_std[i]),float(rel_avg[i] + roll_std[i]*sigma), float(rel_avg[i] - roll_std[i]*sigma),'Low'])
            roll_all_data.append([stock_name, rel_rel_x[i], rel_rel_y[i], float(roll_std[i]), float(rel_avg[i] + roll_std[i]*sigma), float(rel_avg[i] - roll_std[i]*sigma),'Low'])
        else:
            roll_all_data.append([stock_name, rel_rel_x[i], rel_rel_y[i], float(roll_std[i]), float(rel_avg[i] + roll_std[i]*sigma), float(rel_avg[i] - roll_std[i]*sigma), ''])

    # for a,b,c,d,e in roll_all_data:
    #     print(a,b,c,d,e)

    return rel_rel_x, rel_rel_x, rel_avg, roll_anomaly_list

def plot_stuff(x,y,roll_anomaly):

    roll_anomaly = np.array(roll_anomaly)
    plt.figure(figsize=(15, 8))
    plt.plot(x, y)
    plt.plot(roll_anomaly[:,0], roll_anomaly[:,1],'r*')
    plt.show()


def write_csv(w_file, roll_anomaly, header):

    w_file.writerow(header)
    for line in roll_anomaly:
        # print(line)
        w_file.writerow(line)


def get_anomaly(stock_name, df, window_size, sigma):

    close_df = df['Close']
    close_df = close_df.reset_index()
    print('This is close_df', close_df['Close'].count())
    close_df = close_df.dropna()
    print('This is close_df', close_df['Close'].count())

    data = np.array(close_df)
    # print(data)

    rel_x, rel_y, avg = moving_average(data[:, 0], data[:, 1], window_size)

    rel_rel_x, rel_rel_y, rel_avg, roll_anomaly = get_roll_anomaly(stock_name, rel_x, rel_y, avg, window_size, sigma)

    # plot_stuff(data[:,0], data[:,1], roll_anomaly)

    return roll_anomaly

if __name__ == '__main__':

    t0 = time.time()
    stock_symbol = read_csv('C:\\Users\\Joash\\Desktop\\University Stuff\\Personal Projects\\Stock Anomaly Detection\\stock_anomaly\\Data\\test', 0, 1)

    window_size = 10
    sigma = 5
    years_of_data = 5
    roll_anomaly_output = []

    # Ensures that only Mon-Fri dates are used
    weekday = dt.datetime.today().weekday()
    if (weekday == 5):
        end = dt.datetime.now()- timedelta(days=1)
    elif (weekday == 6):
        end = dt.datetime.now() - timedelta(days=2)
    else:
        end = dt.datetime.today().date()

    end = dt.datetime.today().date()
    start = dt.datetime(int(dt.datetime.today().year - years_of_data),int(dt.datetime.today().month), int(dt.datetime.today().day)).date()

    didnt_work = []
    worked = []
    # print(web.DataReader(stock_symbol[0], 'yahoo', start, end))
    for stock in stock_symbol:

        try:
            df = web.DataReader(stock, 'yahoo', start, end)
            worked.append(stock)

        except:
            didnt_work.append(stock)
            print(stock, 'didnt work')
            continue

        roll_anomaly_output = roll_anomaly_output + get_anomaly(stock, df, window_size, sigma)
        # get_anomaly(stock, df, window_size, sigma)

    print('Writing Output')
    w_file = open('C:\\Users\\Joash\\Desktop\\University Stuff\\Personal Projects\\Stock Anomaly Detection\\stock_anomaly\\Data\\Results' + str(dt.datetime.today().date()) + '.csv', 'w', newline='', encoding="latin1")
    write_csv(csv.writer(w_file), roll_anomaly_output,['Stock Symbol', 'Date', 'Price', 'Residual Std Dev', 'Upper Bound', 'Lower Bound', 'Type'])
    # print(didnt_work)
    t1 = time.time()
    print('Time to run code:', t1-t0)