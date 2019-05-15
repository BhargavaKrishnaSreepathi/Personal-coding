import pandas as pd

from keras.models import model_from_json
import imageio

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def further_model(input):
    json_file = open(
        r'C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning_example\model_data_validation_final_' + input + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    predictive_model = model_from_json(loaded_model_json)
    # load weights into new model
    predictive_model.load_weights(
        r"C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning_example\model_data_validation_final_" + input + ".h5")
    print("Loaded model from disk")

    train = pd.read_csv(
        r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\imgs_train.csv')  # reading the csv file
    train.head()  # printing first five rows of the file

    train[input] = np.zeros(len(train))

    x_train_original = []
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
            x_train_original.append([np.array(img), label])


    data = np.array([i[0] for i in x_train_original]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    labels = np.array([i[1] for i in x_train_original])

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.2)
    print('Data Processed')

    datagen = ImageDataGenerator(shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 vertical_flip=True)
    #
    # # compute quantities required for featurewise normalization
    # # (std, mean, and principal components if ZCA whitening is applied)
    # datagen.fit(x_train)

    epochs = 60
    batch_size = 100
    epoch_step = len(x_train) / batch_size

    train_generator = datagen.flow(x_train, y_train, batch_size=batch_size)
    predictive_model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])


    # fits the model on batches with real-time data augmentation:
    predictive_model.fit_generator(train_generator, steps_per_epoch=epoch_step, validation_data=(x_test, y_test), epochs=epochs, verbose = 1)
    # predictive_model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_split = 0.2)
    loss, acc = predictive_model.evaluate(x_train, y_train, verbose = 1)
    print (loss, acc)
    print (input + ' done')

    scores = predictive_model.evaluate(data, labels, verbose=1)
    print("%s: %.2f%%" % (predictive_model.metrics_names[1], scores[1] * 100))

    # serialize model to JSON
    model_json = predictive_model.to_json()
    with open("model_further_data_validation_final_" + input + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    predictive_model.save_weights("model_further_data_validation_final_" + input + ".h5")
    print("Saved model to disk")

# further_model('floorplan')
# further_model('map')
# further_model('face')
# further_model('collage')
further_model('property')
# further_model('siteplan')
# further_model('text')