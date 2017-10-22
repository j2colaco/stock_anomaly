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


def moving_average(y, window_size=3):
    window = np.ones(int(window_size))/float(window_size)
    moving_avg = np.convolve(y, window, 'same')
    return moving_avg.tolist()

x, y = read_csv('C:\\Users\\Joash\\Desktop\\University Stuff\\Personal Projects\\Stock Anomaly Detection\\MFC.TO2', 0, 1)

c = moving_average(y,10)
# print(c)


residue = y - c
residue_std = np.std(residue)
# print(residue_std)

izip = zip
zipped = zip(y, c)
# for a,b in zipped:
#     print(a,b)

sigma = 1.5
count = 0
anomaly_list = []
anomaly_date = []
for a,b in zipped:
    if (a > b + residue_std*sigma):
        # print('High anomaly: ', a, ' @', x[count])
        anomaly_list.append(1)
        anomaly_date.append([x[count], a])
    elif (a < b - residue_std*sigma):
        # print('Low anomaly: ', a, ' @', x[count])
        anomaly_list.append(1)
        anomaly_date.append([x[count], a])
    else:
        anomaly_list.append(0)
    count += 1

# for i in anomaly_list:
#     print(i)


b = np.multiply(anomaly_list,y)
print(b)

an_date = []
an_price = []
for a in anomaly_date:
    an_date.append(a[0])
    an_price.append(a[1])

an_date2 = np.array(an_date, dtype='datetime64')
# print(an_date2[1])

plt.figure(figsize=(15, 8))
plt.plot(x, y)
y_moving_average = moving_average(y, 5)
plt.plot(x, y_moving_average, color='green')
plt.plot(an_date2, an_price, "r*", markersize=12)
plt.show()


#Test moving average using discrete convolution
# nn = np.convolve([4,4,4,5,6,7,8,10], [.33, 1,.33], 'same')
# print(nn)