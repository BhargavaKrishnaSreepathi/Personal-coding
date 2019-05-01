import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from keras.models import model_from_json
import imageio
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization
import numpy as np

def further_model(input):
    json_file = open(
        r'C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning_example\model_further_' + input + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(
        r"C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning_example\model_further_" + input + ".h5")
    print("Loaded model from disk")

    train = pd.read_csv(
        r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\imgs_train.csv')  # reading the csv file
    train.head()  # printing first five rows of the file

    train[input] = np.zeros(len(train))

    train_data_original = []
    train_labels_original = []
    IMG_SIZE = 299

    for i in range(len(train)):
        if input in train.loc[i]['labels']:
            label = 1
        else:
            label = 0

        img = imageio.imread(
            r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/' + str(
                train.loc[i, 'images_id']))
        arr = np.array(img)

        if len(arr.shape) > 2:
            train_data_original.append([np.array(img), label])

    trainImages = np.array([i[0] for i in train_data_original]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    trainLabels = np.array([i[1] for i in train_data_original])
    print('Data Processed')

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
    model.fit(trainImages, trainLabels, batch_size = 100, epochs = 1, verbose = 1)
    loss, acc = model.evaluate(trainImages, trainLabels, verbose = 0)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model_further_" + input + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_further_" + input + ".h5")
    print("Saved model to disk")

# further_model('floorplan')
# further_model('map')
# further_model('face')
# further_model('collage')
# further_model('property')
# further_model('siteplan')
further_model('text')