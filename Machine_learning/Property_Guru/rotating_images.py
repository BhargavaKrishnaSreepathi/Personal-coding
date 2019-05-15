import pandas as pd
import matplotlib.pyplot as plt
import imageio
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

__author__ = "Sreepathi Bhargava Krishna"
__credits__ = ["Sreepathi Bhargava Krishna"]
__email__ = "s.bhargava.krishna@gmail.com"
__status__ = "Made for the Assessment"

def custom_classification(input):

    train = pd.read_csv(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\imgs_train.csv')    # reading the csv file
    train.head()      # printing first five rows of the file

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

    data_full = np.array([i[0] for i in x_train_original]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    labels_full = np.array([i[1] for i in x_train_original])

    x_train, x_validation, y_train, y_validation = train_test_split(data_full, labels_full, test_size = 0.1)
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
    predictive_model.add(Dense(128, activation='relu'))
    predictive_model.add(Dense(1, activation = 'sigmoid'))

    predictive_model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
    epochs = 50
    batch_size = 100

    history = predictive_model.fit(data_full, labels_full, batch_size = batch_size, epochs = epochs, verbose = 1)
    loss, acc = predictive_model.evaluate(data_full, labels_full, verbose = 1)
    print (loss, acc)
    print (input + ' done')

    scores = predictive_model.evaluate(data_full, labels_full, verbose=1)
    print("%s: %.2f%%" % (predictive_model.metrics_names[1], scores[1] * 100))

    # serialize model to JSON
    model_json = predictive_model.to_json()
    with open("model_data_validation_final_" + input + "_4.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    predictive_model.save_weights("model_data_validation_final_" + input + "_4.h5")
    print("Saved model to disk")

    # print(history.history.keys())
    # #  "Accuracy"
    # plt.plot(history.history['acc'])
    # # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.savefig('accuracy_plot_' + input + '_4.png')
    # # "Loss"
    # plt.plot(history.history['loss'])
    # # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.savefig('loss_plot_' + input + '_4.png')


# class_labels = ['text', 'floorplan', 'map', 'face', 'collage', 'property', 'siteplan']

# custom_classification('text')
# custom_classification('floorplan')
custom_classification('map')
custom_classification('face')
custom_classification('collage')
custom_classification('property')
custom_classification('siteplan')
# custom_classification('text')



