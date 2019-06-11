import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers.core import Dropout
from keras import optimizers
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


payment_data = pd.read_csv(r'C:\Users\krish\Desktop\AI Lab - Technical Interview\AI Lab - Technical Interview\Anomaly Detection\payment_data_ratio20.csv')

payment_data.info()
payment_data.describe()
payment_data.head(10)

print (payment_data.groupby('id'))