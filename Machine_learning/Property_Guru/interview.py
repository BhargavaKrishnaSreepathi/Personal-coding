import pandas as pd
import imageio
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image


__author__ = "Sreepathi Bhargava Krishna"
__credits__ = ["Sreepathi Bhargava Krishna"]
__email__ = "s.bhargava.krishna@gmail.com"
__status__ = "Made for the Assessment"

train = pd.read_csv(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/imgs_train.csv')    # reading the csv file

x_train_original = []
IMG_SIZE = 299

for i in range(len(train)):
    if 'text' in train.loc[i]['labels']:
        label_text = 1
    else:
        label_text = 0

    if 'floorplan' in train.loc[i]['labels']:
        label_floorplan = 1
    else:
        label_floorplan = 0

    if 'map' in train.loc[i]['labels']:
        label_map = 1
    else:
        label_map = 0

    if 'face' in train.loc[i]['labels']:
        label_face = 1
    else:
        label_face = 0

    if 'collage' in train.loc[i]['labels']:
        label_collage = 1
    else:
        label_collage = 0

    if 'property' in train.loc[i]['labels']:
        label_property = 1
    else:
        label_property = 0

    if 'siteplan' in train.loc[i]['labels']:
        label_siteplan = 1
    else:
        label_siteplan = 0

    img = imageio.imread(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/' + str(train.loc[i, 'images_id']))
    arr = np.array(img)/255.0
    if len(arr.shape) > 2:
        x_train_original.append([np.array(img), (label_text, label_floorplan, label_map, label_face, label_collage, label_property, label_siteplan)])

data_full = np.array([i[0] for i in x_train_original]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
labels_full = np.array([i[1] for i in x_train_original])

x_train, x_validation, y_train, y_validation = train_test_split(data_full, labels_full, test_size = 0.1)
print ('Data Processed')

# Convolutional Model for Image Classification
predictive_model = Sequential()
predictive_model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
predictive_model.add(MaxPooling2D(pool_size=(2,2)))
# predictive_model.add(BatchNormalization())

predictive_model.add(Conv2D(64, kernel_size=(5,5), activation='relu'))
predictive_model.add(MaxPooling2D(pool_size=(2,2)))
# predictive_model.add(BatchNormalization())

predictive_model.add(Conv2D(96, kernel_size=(7,7), activation='relu'))
predictive_model.add(MaxPooling2D(pool_size=(2,2)))
# predictive_model.add(BatchNormalization())

predictive_model.add(Conv2D(64, kernel_size=(5,5), activation='relu'))
predictive_model.add(MaxPooling2D(pool_size=(2,2)))

predictive_model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
predictive_model.add(MaxPooling2D(pool_size=(2,2)))
# predictive_model.add(BatchNormalization())
predictive_model.add(Dropout(0.2))
predictive_model.add(Flatten())

predictive_model.add(Dense(128, activation='relu'))
predictive_model.add(Dense(64, activation='relu'))
predictive_model.add(Dense(32, activation='relu'))
predictive_model.add(Dense(7, activation = 'sigmoid'))

predictive_model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

epochs = 100
batch_size = 100
epoch_step = len(x_train) / batch_size

# fits the model on batches
history = predictive_model.fit(x_train, y_train, epochs=epochs, verbose = 1, validation_data=(x_validation, y_validation))
loss, acc = predictive_model.evaluate(data_full, labels_full, verbose = 1)
print (loss, acc)
print ('done')

# serialize model to JSON
model_json = predictive_model.to_json()
with open("model_data_validation_final_all_interview_bw.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
predictive_model.save_weights("model_data_validation_final_all_interview_bw.h5")
print("Saved model to disk")

scores = predictive_model.evaluate(data_full, labels_full, verbose=1)
print("%s: %.2f%%" % (predictive_model.metrics_names[1], scores[1] * 100))

print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('accuracy_plot_bw.png')
# "Loss"
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('loss_plot_bw.png')







