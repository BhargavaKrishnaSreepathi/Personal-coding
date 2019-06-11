
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import imageio
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization
import numpy as np
from keras.models import model_from_json

def custom_classification(input):

    train = pd.read_csv(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\imgs_train.csv')    # reading the csv file
    train.head()      # printing first five rows of the file

    train[input] = np.zeros(len(train))

    train_data_original = []
    train_labels_original = []
    IMG_SIZE = 299

    for i in range(len(train)):
        if input in train.loc[i]['labels']:
            label = 1
        else:
            label = 0

        img = imageio.imread(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/' + str(train.loc[i, 'images_id']))
        arr = np.array(img)

        if len(arr.shape) > 2:

            train_data_original.append([np.array(img), label])


    trainImages = np.array([i[0] for i in train_data_original]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    trainLabels = np.array([i[1] for i in train_data_original])
    print ('Data Processed')

    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
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
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
    model.fit(trainImages, trainLabels, batch_size = 100, epochs = 1, verbose = 1)
    loss, acc = model.evaluate(trainImages, trainLabels, verbose = 0)

    scores = model.evaluate(trainImages, trainLabels, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # serialize model to JSON
    model_json = model.to_json()
    with open("model_gpu_" + input + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_gpu_" + input + ".h5")
    print("Saved model to disk")

    #
    # test = pd.read_csv(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\imgs_train.csv')    # reading the csv file
    #
    # test_data_original = []
    # IMG_SIZE = 299
    #
    # for i in range(len(test)):
    #     # img = Image.open(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/' + str(train.loc[i, 'images_id']))
    #     # img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    #     img = imageio.imread(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/' + str(test.loc[i, 'images_id']))
    #     arr = np.array(img)
    #
    #     if len(arr.shape) > 2:
    #
    #         test_data_original.append([np.array(img)])
    #
    # testImages = np.array([i[0] for i in test_data_original]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    #
    # ytest = model.predict_classes(testImages)
    #
    # # show the inputs and predicted outputs
    # for i in range(len(testImages)):
    #     print("X=%s, Predicted=%s" % (testImages[i], ytest[i]))

# class_labels = ['text', 'floorplan', 'map', 'face', 'collage', 'property', 'siteplan']

# custom_classification('text')
# custom_classification('floorplan')
# custom_classification('map')
# custom_classification('face')
# custom_classification('collage')
# custom_classification('property')
# custom_classification('siteplan')
custom_classification('text')


# # load json and create model
# input = 'text'
# json_file = open(r'C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning\model_' + input + '.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights(r"C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning\model_" + input + ".h5")
# print("Loaded model from disk")
#
# test = pd.read_csv(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\imgs_train.csv')    # reading the csv file
#
# test_data_original = []
# IMG_SIZE = 299
#
# for i in range(len(test)):
#     # img = Image.open(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/' + str(train.loc[i, 'images_id']))
#     # img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
#     img = imageio.imread(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/' + str(test.loc[i, 'images_id']))
#     arr = np.array(img)
#     if len(arr.shape) > 2:
#         test_data_original.append([np.array(img)])
#
# testImages = np.array([i[0] for i in test_data_original]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
#
# ynew = loaded_model.predict_classes(testImages)
