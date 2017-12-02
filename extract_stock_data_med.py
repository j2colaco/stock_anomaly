import csv
import numpy as np
# import datetime as dt
import datetime as dt
from datetime import timedelta
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import time
import xlsxwriter

def read_csv(file_name):
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

def moving_median(x, y, window_size):

    window = np.ones(int(window_size)) / float(window_size)
    rel_y = y[window_size:]
    rel_x = x[window_size:]

    y_df = pd.DataFrame(y)
    moving_med = y_df.rolling(window=window_size).median()
    moving_med = moving_med[window_size-1:len(moving_med)-1]
    moving_med = np.array(moving_med)

    return rel_x, rel_y, moving_med

def get_roll_anomaly(stock_name, rel_x, rel_y, avg, window_size, sigma):

    rel_rel_x = rel_x[window_size:]
    rel_rel_y = rel_y[window_size:]
    rel_med = avg[window_size:]
    # print('this is', rel_med)
    # print(rel_med[2])

    residual_np = rel_y - avg
    residual_np = np.absolute(residual_np)
    residual = pd.DataFrame(residual_np)
    roll_std = residual.rolling(window=window_size,center=False).std()
    roll_std = roll_std[window_size-1:len(roll_std)-1]
    roll_std = np.array(roll_std)

    # print('The length of rel_rel_x, rel_rel_y, roll_std and rel_avg are', len(rel_rel_x), len(rel_rel_y), len(roll_std), len(rel_avg))

    roll_anomaly_list = []
    roll_all_data = []
    for i in range(0, len(rel_rel_y), 1):
        if (rel_rel_y[i] > rel_med[i][0] + roll_std[i][0]*sigma):
            type = "High"
            if (i + 30 <= len(rel_rel_y)-1):
                day_1 = rel_rel_y[i + 1] - rel_rel_y[i]
                day_2 = rel_rel_y[i + 2] - rel_rel_y[i]
                day_3 = rel_rel_y[i + 3] - rel_rel_y[i]
                day_5 = rel_rel_y[i + 5] - rel_rel_y[i]
                day_10 = rel_rel_y[i + 10] - rel_rel_y[i]
                day_30 = rel_rel_y[i + 30] - rel_rel_y[i]

            elif (i + 10 <= len(rel_rel_y)-1):
                day_1 = rel_rel_y[i + 1] - rel_rel_y[i]
                day_2 = rel_rel_y[i + 2] - rel_rel_y[i]
                day_3 = rel_rel_y[i + 3] - rel_rel_y[i]
                day_5 = rel_rel_y[i + 5] - rel_rel_y[i]
                day_10 = rel_rel_y[i + 10] - rel_rel_y[i]
                day_30 = None

            elif (i + 5 <= len(rel_rel_y)-1):
                day_1 = rel_rel_y[i + 1] - rel_rel_y[i]
                day_2 = rel_rel_y[i + 2] - rel_rel_y[i]
                day_3 = rel_rel_y[i + 3] - rel_rel_y[i]
                day_5 = rel_rel_y[i + 5] - rel_rel_y[i]
                day_10 = None; day_30 = None

            elif (i + 3 <= len(rel_rel_y)-1):
                day_1 = rel_rel_y[i + 1] - rel_rel_y[i]
                day_2 = rel_rel_y[i + 2] - rel_rel_y[i]
                day_3 = rel_rel_y[i + 3] - rel_rel_y[i]
                day_5 = None; day_10 = None; day_30 = None

            elif (i + 2 <= len(rel_rel_y) - 1):
                day_1 = rel_rel_y[i + 1] - rel_rel_y[i]
                day_2 = rel_rel_y[i + 2] - rel_rel_y[i]
                day_3 = None; day_5 = None; day_10 = None; day_30 = None

            elif (i + 1 <= len(rel_rel_y)-1):
                day_1 = rel_rel_y[i + 1] - rel_rel_y[i]
                day_2 = None; day_3 = None; day_5 = None; day_10 = None; day_30 = None

            else:
                day_1 = None; day_2 = None; day_3 = None; day_5 = None; day_10 = None; day_30 = None

            roll_anomaly_list.append(
                [stock_name, rel_rel_x[i].date(), rel_rel_y[i], day_1, day_2, day_3, day_5, day_10, day_30,
                 float(roll_std[i][0]), float(rel_med[i][0] + roll_std[i][0] * sigma),
                 float(rel_med[i][0] - roll_std[i][0] * sigma), type])

            roll_all_data.append([stock_name, rel_rel_x[i], rel_rel_y[i], float(roll_std[i][0]),
                                  float(rel_med[i][0] + roll_std[i][0] * sigma), float(rel_med[i][0] - roll_std[i][0] * sigma),
                                  type])

        elif (rel_rel_y[i] < rel_med[i][0] - roll_std[i][0]*sigma):

            type = "Low"
            if (i + 30 <= len(rel_rel_y)-1):
                day_1 = rel_rel_y[i + 1] - rel_rel_y[i]; day_2 = rel_rel_y[i + 2] - rel_rel_y[i]
                day_3 = rel_rel_y[i + 3] - rel_rel_y[i]; day_5 = rel_rel_y[i + 5] - rel_rel_y[i]
                day_10 = rel_rel_y[i + 10] - rel_rel_y[i]; day_30 = rel_rel_y[i + 30] - rel_rel_y[i]

            elif (i + 10 <= len(rel_rel_y)-1):
                day_1 = rel_rel_y[i + 1] - rel_rel_y[i]
                day_2 = rel_rel_y[i + 2] - rel_rel_y[i]
                day_3 = rel_rel_y[i + 3] - rel_rel_y[i]
                day_5 = rel_rel_y[i + 5] - rel_rel_y[i]
                day_10 = rel_rel_y[i + 10] - rel_rel_y[i]
                day_30 = None

            elif (i + 5 <= len(rel_rel_y)-1):
                day_1 = rel_rel_y[i + 1] - rel_rel_y[i]
                day_2 = rel_rel_y[i + 2] - rel_rel_y[i]
                day_3 = rel_rel_y[i + 3] - rel_rel_y[i]
                day_5 = rel_rel_y[i + 5] - rel_rel_y[i]
                day_10 = None; day_30 = None

            elif (i + 3 <= len(rel_rel_y)-1):
                day_1 = rel_rel_y[i + 1] - rel_rel_y[i]
                day_2 = rel_rel_y[i + 2] - rel_rel_y[i]
                day_3 = rel_rel_y[i + 3] - rel_rel_y[i]
                day_5 = None; day_10 = None; day_30 = None

            elif (i + 2 <= len(rel_rel_y) - 1):
                day_1 = rel_rel_y[i + 1] - rel_rel_y[i]
                day_2 = rel_rel_y[i + 2] - rel_rel_y[i]
                day_3 = None; day_5 = None; day_10 = None; day_30 = None

            elif (i + 1 <= len(rel_rel_y)-1):
                day_1 = rel_rel_y[i + 1] - rel_rel_y[i]
                day_2 = None; day_3 = None; day_5 = None; day_10 = None; day_30 = None
            else:
                day_1 = rel_rel_y[i + 1] - rel_rel_y[i]
                day_2 = None; day_3 = None; day_5 = None; day_10 = None; day_30 = None

            roll_anomaly_list.append(
                [stock_name, rel_rel_x[i].date(), rel_rel_y[i], day_1, day_2, day_3, day_5, day_10, day_30,
                 float(roll_std[i][0]), float(rel_med[i][0] + roll_std[i][0] * sigma),
                 float(rel_med[i][0] - roll_std[i][0] * sigma), type])

            roll_all_data.append([stock_name, rel_rel_x[i], rel_rel_y[i], float(roll_std[i][0]),
                                  float(rel_med[i][0] + roll_std[i][0] * sigma), float(rel_med[i][0] - roll_std[i][0] * sigma),
                                  type])
        else:
            roll_all_data.append([stock_name, rel_rel_x[i], rel_rel_y[i], float(roll_std[i][0]),
                                  float(rel_med[i][0] + roll_std[i][0]*sigma), float(rel_med[i][0] - roll_std[i][0]*sigma), ''])

    # for a,b,c,d,e in roll_all_data:
    #     print(a,b,c,d,e)

    return rel_rel_x, rel_rel_x, rel_med, roll_anomaly_list, roll_all_data

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

def manipulate_data(roll_anomally, wb):

    df = pd.DataFrame(roll_anomally, columns=['stock_name', 'date', 'price', '1day', '2day', '3day', '5day', '10day', '30day', 'std', 'upper_bound', 'lower_bound', 'type'])
    low_df = df[df['type']== 'Low']
    high_df = df[df['type']== 'High']
    # print(new_df)

    # low = low_df.groupby('stock_name').agg({'1day': np.nanmean, '1day': np.nanmedian, '2day': np.nanmean, '2day': np.nanmedian, '3day': np.nanmean, '3day': np.nanmedian, '5day': np.nanmean, '5day': np.nanmedian, '10day': np.nanmean, '10day': np.nanmedian, '30day': np.nanmean, '30day': np.nanmedian})
    low_mean = low_df.groupby('stock_name').agg({'1day': np.nanmean, '2day': np.nanmean, '3day': np.nanmean, '5day': np.nanmean, '10day': np.nanmean, '30day': np.nanmean}).reset_index()
    high_mean = high_df.groupby('stock_name').agg({'1day': np.nanmean, '2day': np.nanmean, '3day': np.nanmean, '5day': np.nanmean, '10day': np.nanmean, '30day': np.nanmean}).reset_index()
    low_median = low_df.groupby('stock_name').agg({'1day': np.nanmedian, '2day': np.nanmedian, '3day': np.nanmedian, '5day': np.nanmedian, '10day': np.nanmedian, '30day': np.nanmedian}).reset_index()
    high_median = high_df.groupby('stock_name').agg({'1day': np.nanmedian, '2day': np.nanmedian, '3day': np.nanmedian, '5day': np.nanmedian, '10day': np.nanmedian, '30day': np.nanmedian}).reset_index()

    low_mean_ws = wb.add_worksheet('Low Average')
    low_median_ws = wb.add_worksheet('Low Median')
    high_mean_ws = wb.add_worksheet('High Average')
    high_median_ws = wb.add_worksheet('High Median')

    write_xlsx(low_mean, low_mean_ws, ['Symbol', '1day_avg', '2day_avg', '3day_avg', '5day_avg', '10day_avg', '30day_avg'])
    write_xlsx(low_median, low_median_ws,
               ['Symbol', '1day_med', '2day_med', '3day_med', '5day_med', '10day_med', '30day_med'])
    write_xlsx(high_median, high_median_ws,
               ['Symbol', '1day_med', '2day_med', '3day_med', '5day_med', '10day_med', '30day_med'])
    write_xlsx(high_mean, high_mean_ws,
               ['Symbol', '1day_avg', '2day_avg', '3day_avg', '5day_avg', '10day_avg', '30day_avg'])
    # print(low_mean)

def write_xlsx(df, ws, header):
    row = 0
    col = 0

    for i in header:
        ws.write(row, col, i)
        col += 1
    col = 0
    row = 1

    for a in df.values:
        for i in range(0, len(a), 1):
            ws.write(row, col, a[i])
            col += 1
        col = 0
        row += 1


def get_anomaly(stock_name, df, window_size, sigma):

    close_df = df['Close']
    close_df = close_df.reset_index()
    close_df = close_df.dropna()
    # print(close_df)

    data = np.array(close_df)

    rel_x, rel_y, med = moving_median(data[:, 0], data[:, 1], window_size)


    rel_rel_x, rel_rel_y, rel_med, roll_anomaly, roll_all_data = get_roll_anomaly(stock_name, rel_x, rel_y, med, window_size, sigma)

    # roll_anomaly_avg = manipulate_data(roll_anomaly)

    # plot_stuff(data[:,0], data[:,1], roll_anomaly)

    return roll_anomaly, roll_all_data

if __name__ == '__main__':


    t0 = time.time()
    # stock_symbol = read_csv('C:\\Users\\Joash\\Desktop\\University Stuff\\Personal Projects\\Stock Anomaly Detection\\stock_anomaly\\Data\\test')
    stock_symbol = read_csv('C:\\Users\\Joash\\Desktop\\University Stuff\\Personal Projects\\Stock Anomaly Detection\\stock_anomaly\\Data\\S&P.TSX 1 col')

    window_size = 10
    sigma = 5.5
    years_of_data = 3
    roll_anomaly_output = []
    roll_all_data = []

    # Ensures that only Mon-Fri dates are used
    weekday = dt.datetime.today().weekday()
    if (weekday == 5):
        end = dt.datetime.now()- timedelta(days=1)
    elif (weekday == 6):
        end = dt.datetime.now() - timedelta(days=2)
    else:
        end = dt.datetime.today().date()

    start = dt.datetime(int(dt.datetime.today().year - years_of_data),int(dt.datetime.today().month), int(dt.datetime.today().day)).date()

    didnt_work = []
    didnt_work2 = []
    not_enff_data = []
    worked = []
    wb = xlsxwriter.Workbook(
        'C:\\Users\\Joash\\Desktop\\University Stuff\\Personal Projects\\Stock Anomaly Detection\\stock_anomaly\\Data\\Med_Manipulated_Results_' + str(
            dt.datetime.today().date()) + '.xlsx')

    # df = web.DataReader('SW.TO', 'yahoo', start, end)
    # temp_roll_anomaly_output, temp_roll_all_data = get_anomaly('SW.TO', df, window_size, sigma)
    # roll_anomaly_output += temp_roll_anomaly_output
    # roll_all_data += temp_roll_all_data
    # roll_anomaly_output = roll_anomaly_output + get_anomaly('SW.TO', df, window_size, sigma)

    # # print(web.DataReader(stock_symbol[0], 'yahoo', start, end))
    for stock in stock_symbol:

        try:
            df = web.DataReader(stock, 'yahoo', start, end)
            if len(df) < 21:
                not_enff_data.append(stock)
                continue
            print(stock)
            # roll_anomaly_output = roll_anomaly_output + get_anomaly(stock, df, window_size, sigma)

            temp_roll_anomaly_output, temp_roll_all_data = get_anomaly(stock, df, window_size, sigma)
            # print(temp_roll_all_data)
            roll_anomaly_output.append(temp_roll_anomaly_output)
            roll_all_data.append(temp_roll_all_data)

            worked.append(stock)

        except:
            didnt_work.append(stock)
            print(stock, 'didnt work')
            continue


    for stock in didnt_work:

        try:
            df = web.DataReader(stock, 'yahoo', start, end)
            if len(df) < 21:
                not_enff_data.append(stock)
                continue
            print(stock)
            roll_anomaly_output = roll_anomaly_output + get_anomaly(stock, df, window_size, sigma)

            # temp_roll_anomaly_output, temp_roll_all_data = get_anomaly(stock, df, window_size, sigma)
            # roll_anomaly_output += temp_roll_anomaly_output
            # roll_all_data += temp_roll_all_data
            worked.append(stock)

        except:
            didnt_work2.append(stock)
            print(stock, 'didnt work again')
            continue

    roll_anomaly_avg = manipulate_data(roll_anomaly_output, wb)

    print('Writing Output')
    w_file = open('C:\\Users\\Joash\\Desktop\\University Stuff\\Personal Projects\\Stock Anomaly Detection\\stock_anomaly\\Data\\Med_Results_' + str(dt.datetime.today().date()) + '.csv', 'w', newline='', encoding="latin1")
    write_csv(csv.writer(w_file), roll_anomaly_output,['Stock Symbol', 'Date', 'Price', '1day', '2day', '3day', '5day', '10day', '30day', 'Residual Std Dev', 'Upper Bound', 'Lower Bound', 'Type'])
    # all_anomalies_ws = wb.add_worksheet('All_Anomalies')
    # write_xlsx(pd.DataFrame(roll_anomaly_output), all_anomalies_ws,
               # ['Symbol', 'Date', 'Price', '1day', '2day', '3day', '5day', '10day', '30day', 'Residual Std Dev', 'Upper Bound', 'Lower Bound', 'Type'])
    wb.close()
    print('These didnt work', didnt_work2)
    print('Not enough data', not_enff_data)
    t1 = time.time()
    print('Time to run code:', (t1-t0)/60, 'minutes')