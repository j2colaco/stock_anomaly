import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader as web
import numpy as np

# style.use('ggplot')
#
# start = dt.datetime(2000,1,1)
# end = dt.datetime(2016, 12, 31)
#
# df = web.DataReader('MFC.TO', 'yahoo', start, end)
# print(df.tail(6))
# print('hi')
#
# N = 30
#
# if (today == 5):
#     New = dt.datetime.now()- timedelta(days=N)
#     New = New.date()
# elif (today == 6):
#     New = dt.datetime.now() - timedelta(days=N)
#     New = New.date()
#
# # print(New)
# today = dt.datetime.today().weekday()
# print(today)

a = [['a', 5, 1], ['b', 1, 5], ['a', 3, 1], ['b', 2, None]]
# print(a)
df = pd.DataFrame(a, columns=['alpha', 'num', 'num2'])
print(df)

new_df = df.groupby('alpha').agg({'num': np.average, 'num2': np.average})
print(new_df)