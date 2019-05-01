
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
from keras.preprocessing.image import ImageDataGenerator


def custom_classification(input):

    train = pd.read_csv(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\imgs_train.csv')    # reading the csv file
    train.head()      # printing first five rows of the file

    train[input] = np.zeros(len(train))

    x_train_original = []
    IMG_SIZE = 299

    for i in range(len(train)):
        if input in train.loc[i]['labels']:
            label = 1
        else:
            label = 0
        img = imageio.imread(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/' + str(train.loc[i, 'images_id']))
        arr = np.array(img)
        if len(arr.shape) > 2:
            x_train_original.append([np.array(img), label])

    x_train = np.array([i[0] for i in x_train_original]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    y_train = np.array([i[1] for i in x_train_original])
    print ('Data Processed')

    predictive_model = Sequential()
    predictive_model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    predictive_model.add(MaxPooling2D(pool_size=(2,2)))
    predictive_model.add(BatchNormalization())
    predictive_model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    predictive_model.add(MaxPooling2D(pool_size=(2,2)))
    predictive_model.add(BatchNormalization())
    predictive_model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
    predictive_model.add(MaxPooling2D(pool_size=(2,2)))
    predictive_model.add(BatchNormalization())
    predictive_model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
    predictive_model.add(MaxPooling2D(pool_size=(2,2)))
    predictive_model.add(BatchNormalization())
    predictive_model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    predictive_model.add(MaxPooling2D(pool_size=(2,2)))
    predictive_model.add(BatchNormalization())
    predictive_model.add(Dropout(0.2))
    predictive_model.add(Flatten())
    predictive_model.add(Dense(256, activation='relu'))
    predictive_model.add(Dropout(0.2))
    predictive_model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.3))
    predictive_model.add(Dense(1, activation = 'sigmoid'))

    predictive_model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=0,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True)

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(x_train)

    epochs = 10
    batch_size = 100

    # fits the model on batches with real-time data augmentation:
    predictive_model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(x_train) / batch_size, epochs=epochs, verbose = 1)



    # model.fit(trainImages, trainLabels, batch_size = 100, epochs = 1, verbose = 1)
    loss, acc = predictive_model.evaluate(x_train, y_train, verbose = 1)

    scores = predictive_model.evaluate(x_train, y_train, verbose=1)
    print("%s: %.2f%%" % (predictive_model.metrics_names[1], scores[1] * 100))

    # serialize model to JSON
    model_json = predictive_model.to_json()
    with open("model_data_augmented_1_" + input + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    predictive_model.save_weights("model_data_augmented_1_" + input + ".h5")
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
custom_classification('property')
# custom_classification('siteplan')
# custom_classification('text')



