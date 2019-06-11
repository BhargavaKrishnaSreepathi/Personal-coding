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
from keras.optimizers import SGD
from keras.models import model_from_json
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
from keras.optimizers import SGD
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping


__author__ = "Sreepathi Bhargava Krishna"
__credits__ = ["Sreepathi Bhargava Krishna"]
__email__ = "s.bhargava.krishna@gmail.com"
__status__ = "Made for the Assessment"


train = pd.read_csv(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\imgs_train.csv')    # reading the csv file

x_train_original = []
IMG_SIZE = 299

for i in range(8000, 10000):
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
    arr = np.array(img) / 255.0
    if len(arr.shape) > 2:
        x_train_original.append([arr, (label_text, label_floorplan, label_map, label_face, label_collage, label_property, label_siteplan)])

data_full = np.array([i[0] for i in x_train_original]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
labels_full = np.array([i[1] for i in x_train_original])


x_train, x_validation, y_train, y_validation = train_test_split(data_full, labels_full, test_size = 0.1)
print ('Data Processed')

json_file = open(r"C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning\Property_Guru\model_data_validation_final_all_interview_bw_inception_3.json", "r")
loaded_model_json = json_file.read()
json_file.close()
predictive_model = model_from_json(loaded_model_json)
# load weights into new model
predictive_model.load_weights(r"C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning\Property_Guru\vgg16_3.h5")
print("Loaded model from disk")

predictive_model.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

checkpoint = ModelCheckpoint("vgg19_4.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=100, verbose=1, mode='auto')

epochs = 200
batch_size = 100
epoch_step = len(x_train) / batch_size

# fits the model on batches

history = predictive_model.fit(x_train, y_train, epochs=epochs, verbose = 1, validation_data=(x_validation, y_validation), callbacks = [checkpoint, early])
loss, acc = predictive_model.evaluate(data_full, labels_full, verbose = 1)
print (loss, acc)
print ('done')

# serialize model to JSON
model_json = predictive_model.to_json()
with open("model_data_validation_final_all_interview_bw_inception_3.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
predictive_model.save_weights("model_data_validation_final_all_interview_bw_inception_3.h5")
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
plt.savefig('accuracy_plot_bw_fruther.png')
# "Loss"
plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('loss_plot_bw_further.png')







