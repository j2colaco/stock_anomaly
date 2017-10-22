#code source: https://www.datascience.com/blog/python-anomaly-detection
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import collections
from random import randint

from matplotlib import style


def read_csv(file_name, date_pos, price_pos):
    #Reading the file and putting everything into lists
    read_file = open(file_name + '.csv', 'r', encoding='latin1')
    csv_read = csv.reader(read_file)
    Date = []
    Price = []
    for row in csv_read:
        Date.append(row[date_pos])
        Price.append(row[price_pos])
        # print(row[0], " ", row[1])
    Date = Date[1:]
    Price = Price[1:]
    #Changes the data type from string to float for the Price list
    float_incidents = [float(x) for x in Price]
    x = np.array(Date, dtype='datetime64')
    y = np.array(float_incidents)
    read_file.close()
    return x, y


def moving_average(y,x, window_size=3):
    window = np.ones(int(window_size))/float(window_size)
    # print(window_size)
    rel_y = y[window_size:]
    rel_x = x[window_size:]
    moving_avg = np.convolve(y, window, 'valid')
    moving_avg = moving_avg[:len(moving_avg)-1]
    print('length of rel_y, rel_x and moving average arrays are ', len(rel_y), len(rel_x),len(moving_avg))
    return rel_x, rel_y, moving_avg

def get_stationary_anomaly(rel_x, rel_y, avg, sigma):
    residual = rel_y - avg
    residual_std = np.std(residual)
    print('The std dev is ', residual_std)
    rel_y_avg = zip(rel_y, avg)

    count = 0
    anomaly_list = []
    for i in range(0,len(rel_y)-1,1):
        if (rel_y[i] > avg[i] + residual_std*sigma) | (rel_y[i] < avg[i] - residual_std*sigma):
            anomaly_list.append([rel_x[i], rel_y[i], rel_y[i+1], rel_y[i+5]])

    return np.array(anomaly_list)

def plot_stuff(x,y, rel_x, rel_y, avg, anomaly_list):

    plt.figure(figsize=(15, 8))
    plt.plot(x, y)
    plt.plot(rel_x, avg, color='green')
    plt.plot(anomaly_list[:,0], anomaly_list[:,1], "r*", markersize=12)
    plt.show()

def get_roll_anomaly(rel_x, rel_y, avg, window_size, sigma):

    residual = pd.DataFrame(rel_y - avg)
    roll_std = residual.rolling(window=window_size,center=False).std()
    print(roll_std)
    roll_std = roll_std[window_size-1:len(roll_std)-1]
    print(roll_std)


if __name__ == '__main__':

    sigma = 2.5
    window_size = 10

    # Read all the stock data from the csv file
    x, y = read_csv('C:\\Users\\Joash\\Desktop\\University Stuff\\Personal Projects\\Stock Anomaly Detection\\stock_anomaly\\MFC.TO2', 0, 1)

    rel_x, rel_y, avg = moving_average(y, x, window_size)

    stationary_anomaly = get_stationary_anomaly(rel_x, rel_y, avg, sigma)
    print(len(stationary_anomaly))

    roll_anomaly = get_roll_anomaly(rel_x, rel_y, avg, window_size, sigma)

    # plot_stuff(x,y, rel_x, rel_y, avg, stationary_anomaly)

