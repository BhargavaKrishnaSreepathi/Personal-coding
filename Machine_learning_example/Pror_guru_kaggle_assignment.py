
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from tensorflow.python.keras.preprocessing import image as kp_image


import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops


class_labels = ['text', 'floorplan', 'map', 'face', 'collage', 'property', 'siteplan']

train = pd.read_csv(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\imgs_train.csv')    # reading the csv file
train.head()      # printing first five rows of the file

train['class_text'] = np.zeros(len(train))
train['class_floorplan'] = np.zeros(len(train))
train['class_map'] = np.zeros(len(train))
train['class_face'] = np.zeros(len(train))
train['class_collage'] = np.zeros(len(train))
train['class_property'] = np.zeros(len(train))
train['class_siteplan'] = np.zeros(len(train))

train_data = []
train_labels = []
IMG_SIZE = 300

for i in range(len(train)):
    if 'text' in train.loc[i]['labels']:
        train.loc[i, 'class_text'] = 1

    if 'floorplan' in train.loc[i]['labels']:
        train.loc[i, 'class_floorplan'] = 1

    if 'map' in train.loc[i]['labels']:
        train.loc[i, 'class_map'] = 1

    if 'face' in train.loc[i]['labels']:
        train.loc[i, 'class_face'] = 1

    if 'collage' in train.loc[i]['labels']:
        train.loc[i, 'class_collage'] = 1

    if 'property' in train.loc[i]['labels']:
        train.loc[i, 'class_property'] = 1

    if 'siteplan' in train.loc[i]['labels']:
        train.loc[i, 'class_siteplan'] = 1

    img = Image.open(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/' + str(train.loc[i, 'images_id']))
    img = img.convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    train_data.append([np.array(img), np.array([train.loc[i, 'class_text'], train.loc[i, 'class_text']])])
    # train_labels.append([train.loc[i, 'class_text'], train.loc[i, 'class_floorplan'],
    #                    train.loc[i, 'class_map'], train.loc[i, 'class_face'], train.loc[i, 'class_collage'],
    #                    train.loc[i, 'class_property'], train.loc[i, 'class_face']])


trainImages = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
trainLabels = np.array([i[1] for i in train_data])

# trainLabels = np.array([train_labels[i] for i in range(len(train_labels))]).reshape(len(class_labels), 10000)




model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(2, activation = 'softmax'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
model.fit(trainImages, trainLabels, batch_size = 50, epochs = 5, verbose = 1)


# max_dim = 512
# img = Image.open(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\all_images\image_moderation_images\57445597.jpg')
# long = max(img.size)
# scale = max_dim / long
# img = img.resize((round(img.size[0] * scale), round(img.size[1] * scale)), Image.ANTIALIAS)
#
# img = kp_image.img_to_array(img)
#
# # We need to broadcast the image array such that it has a batch dimension
# img = np.expand_dims(img, axis=0)