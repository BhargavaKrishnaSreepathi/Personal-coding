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

class SmallerVGGNet:
    @staticmethod
    def build(width, height, depth, classes, finalAct="sigmoid"):
        model = Sequential()
        inputShape = (height, width, depth)

        # CONV => RELU => POOL
        model.add(Conv2D(32, (3, 3), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))

        # CONV => RELU => CONV => RELU => POOL
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # CONV => RELU => CONV => RELU => POOL
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # FC => RELU
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Output
        model.add(Dense(classes))
        model.add(Activation(finalAct))

        return model

train = pd.read_csv(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\imgs_train.csv')    # reading the csv file

data = []
labels = []

IMG_SIZE = 299

for i in range(len(train)):
    img = imageio.imread(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/' + str(train.loc[i, 'images_id']))
    arr = np.array(img) / 255.0
    if len(arr.shape) > 2:
        data.append([np.array(img)])
        l = (train.loc[i]['labels']).split(" ")
        labels.append(l)

data_full = np.array([i[0] for i in data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
labels_full = np.array(labels)

image_dims = [IMG_SIZE, IMG_SIZE, 3]
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels_full)

# total 6 labels
print("class labels:")
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i + 1, label))

(trainX, testX, trainY, testY) = train_test_split(data_full, labels, test_size=0.2, random_state=42)

aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
model = SmallerVGGNet.build(width=image_dims[1], height=image_dims[0], depth=image_dims[2], classes=len(mlb.classes_), finalAct="sigmoid")
model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])

batch_size = 20
epochs = 1
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=batch_size), validation_data=(testX, testY), steps_per_epoch=len(trainX) // batch_size,
    epochs=epochs, verbose=1)

# serialize model to JSON
model_json = model.to_json()
with open("model_data_validation_final_all_vgg_t.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_data_validation_final_all_vgg_t.h5")
print("Saved model to disk")

plt.figure(figsize=(10,10))
plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, epochs), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="best")
plt.savefig('accuracy_plot_vgg_t.png')

img = cv2.imread(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\all_images\image_moderation_images\3853906.jpg')
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img.astype("float") / 255.0
img = img_to_array(img)
img = np.expand_dims(img, axis=0)
model.predict(img)

proba = model.predict(img)[0]
idxs = np.argsort(proba)[::-1][:2]

# show the probabilities for each of the individual labels
for (label, p) in zip(mlb.classes_, proba):
    print("{}: {:.2f}%".format(label, p * 100))

# plot image and label
plt.figure(figsize=(10, 10))
for (i, j) in enumerate(idxs):
    # build the label and draw the label on the image
    label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
    plt.text(10, (i * 30) + 25, label, fontsize=16, color='y')

output = load_img(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\all_images\image_moderation_images\3853906.jpg')
plt.xticks([])
plt.yticks([])
plt.imshow(output)