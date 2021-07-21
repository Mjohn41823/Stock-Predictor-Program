import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('TataBeverageLimited.csv')
print(df.head())

plt.figure(figsize=(16,8))
print(plt.plot(df['Close'],label="Close Price History"))

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
