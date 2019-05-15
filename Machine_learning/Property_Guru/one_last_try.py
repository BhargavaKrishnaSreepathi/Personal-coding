from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.optimizers import Adam
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from imutils import paths
import imutils
import numpy as np
import imageio
import cv2
import os
from keras import preprocessing
import random
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

train = pd.read_csv(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\imgs_train.csv')    # reading the csv file

IMG_SIZE = 299
train_image = []
labels = []

for i in range(len(train)):
    if 'text' in train.loc[i]['labels']:
        label_text = 1.0
    else:
        label_text = 0.0

    if 'floorplan' in train.loc[i]['labels']:
        label_floorplan = 1.0
    else:
        label_floorplan = 0.0

    if 'map' in train.loc[i]['labels']:
        label_map = 1.0
    else:
        label_map = 0.0

    if 'face' in train.loc[i]['labels']:
        label_face = 1.0
    else:
        label_face = 0.0

    if 'collage' in train.loc[i]['labels']:
        label_collage = 1.0
    else:
        label_collage = 0.0

    if 'property' in train.loc[i]['labels']:
        label_property = 1.0
    else:
        label_property = 0.0

    if 'siteplan' in train.loc[i]['labels']:
        label_siteplan = 1.0
    else:
        label_siteplan = 0.0

    img = image.load_img(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/' + str(train.loc[i, 'images_id']), target_size=(IMG_SIZE, IMG_SIZE, 3))
    arr = np.array(img) / 255.0
    if len(arr.shape) > 2:
        train_image.append([np.array(img)])
        labels.append((label_text, label_floorplan, label_map, label_face, label_collage, label_property, label_siteplan))
    else:
        print (str(train.loc[i, 'images_id']))

data_full = np.array([i[0] for i in train_image]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
labels_full = np.array(labels)

x_train, x_validation, y_train, y_validation = train_test_split(data_full, labels_full, test_size = 0.2)
print ('Data Processed')

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=(3, 3), activation="relu", input_shape=(IMG_SIZE,IMG_SIZE,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

epochs = 300
batch_size = 50
epoch_step = len(x_train) / batch_size

# fits the model on batches with real-time data augmentation:
history = model.fit(x_train, y_train, epochs=epochs, verbose = 1, validation_data=(x_validation, y_validation))
loss, acc = model.evaluate(data_full, labels_full, verbose = 1)
print (loss, acc)
print ('done')

# serialize model to JSON
model_json = model.to_json()
with open("model_data_validation_final_all_last_try.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_data_validation_final_all_last_try.h5")
print("Saved model to disk")

scores = model.evaluate(data_full, labels_full, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('accuracy_plot_last_try.png')
# "Loss"
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('loss_plot_last_try.png')

img = image.load_img(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/7681593.jpg',target_size=(IMG_SIZE,IMG_SIZE,3))
img = image.img_to_array(img)
img = img/255.0

classes = np.array(train.columns[2:])
proba = model.predict(img.reshape(1,IMG_SIZE,IMG_SIZE,3))
print (proba)
