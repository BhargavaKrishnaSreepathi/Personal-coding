import pandas as pd
import imageio
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GRU
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


train_data = pd.read_csv(r'C:\Users\krish\Desktop\machine learning kaggle\titanic\train.csv')

print(train_data.head())  # checking the first 5 rows
print(train_data.tail())  # checking the last 5 rows

print(train_data.count())  # checking if there are any empty cells
print(train_data.info())  # information about the dataframe
print(train_data.describe())


#  checking the correlation of various variables with the survival rate
print(train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# Pclass 1 has the highest survival rate (62.96%), Pclass 2 has (47.28%) and Pclass 3 has (24.23%)

print(train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# Sex class female has survival rate of 74.20% and Sex class male has a survival rate of 18.89%

print(train_data[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# has a lot of columns with varying ages, so difficult to deduct information without other constraints

print(train_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# Sibsp 1 has the highest survival rate of 53.58%, Sibsp 2 has a survival rate of 46.42%, Sibsp 0 has a survival rate of 34.53%

print(train_data[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train_data[['Ticket', 'Survived']].groupby(['Ticket'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train_data[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train_data[['Cabin', 'Survived']].groupby(['Cabin'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))



train_data['Title'] = train_data.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())

# normalize the titles
normalized_titles = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
}

train_data.Title = train_data.Title.map(normalized_titles)

print(train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False))
