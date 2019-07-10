import pandas as pd
import imageio
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GRU
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

train_data = pd.read_csv(r'C:\Users\krish\Downloads\titanic.csv')
train_data = train_data.replace('?', np.nan)
df1 = train_data.iloc[:,[1, 2, 4, 5, 6, 7, 9, 11]]

df1['Age'] = pd.to_numeric(df1['Age'])

df2 = df1.dropna()

age_mean = df2['Age'].mean()

df1['Age'] = df1['Age'].fillna(df1['Age'].mean())

# df1['Age'] = df1["Age"].fillna(age_mean, inplace = True)
#
# df_no_na = df1.dropna()

print (df1)

# x_train, x_validation, y_train, y_validation = train_test_split(data_full, labels_full, test_size = 0.1)


