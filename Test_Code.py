#code source: https://www.datascience.com/blog/python-anomaly-detection
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import collections
from random import randint

from matplotlib import style


a = [1,2,3,4,5,6]
b = [1,1,1]

c = np.convolve(a,b, 'valid')
c = c[:len(c)-1]
print(c)

print(len(a))
