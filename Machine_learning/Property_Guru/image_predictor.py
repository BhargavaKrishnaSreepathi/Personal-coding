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

# text classifier
json_file = open(r'C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning_example\model_further_text.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
text_classifier = model_from_json(loaded_model_json)
# load weights into new model
text_classifier.load_weights(r"C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning_example\model_further_text.h5")

# map classifier
json_file = open(r'C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning_example\model_further_map.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
map_classifier = model_from_json(loaded_model_json)
# load weights into new model
map_classifier.load_weights(r"C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning_example\model_further_map.h5")

# siteplan classifier
json_file = open(r'C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning_example\model_further_siteplan.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
siteplan_classifier = model_from_json(loaded_model_json)
# load weights into new model
siteplan_classifier.load_weights(r"C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning_example\model_further_siteplan.h5")

# floorplan classifier
json_file = open(r'C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning_example\model_further_floorplan.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
floorplan_classifier = model_from_json(loaded_model_json)
# load weights into new model
floorplan_classifier.load_weights(r"C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning_example\model_further_floorplan.h5")

# face classifier
json_file = open(r'C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning_example\model_further_face.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
face_classifier = model_from_json(loaded_model_json)
# load weights into new model
face_classifier.load_weights(r"C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning_example\model_further_face.h5")

# collage classifier
json_file = open(r'C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning_example\model_further_collage.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
collage_classifier = model_from_json(loaded_model_json)
# load weights into new model
collage_classifier.load_weights(r"C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning_example\model_further_collage.h5")

# property classifier
json_file = open(r'C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning_example\model_further_property.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
property_classifier = model_from_json(loaded_model_json)
# load weights into new model
property_classifier.load_weights(r"C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning_example\model_further_property.h5")

def predictor(input):

    test = pd.read_csv(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\test_set.csv')    # reading the csv file

    test_data_original = []
    IMG_SIZE = 299
    image_id = []
    black_and_white = []

    # img = Image.open(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/' + str(train.loc[i, 'images_id']))
    # img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    img = imageio.imread(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/' + input)
    arr = np.array(img)
    if len(arr.shape) > 2:
        test_data_original.append([np.array(img)])


    testImages = np.array([i[0] for i in test_data_original]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    image_ids = np.array(image_id)

    ynew = loaded_model.predict_classes(testImages)
    print ('predictions done')

    a = []
    for i in range(len(ynew)):
        a.append(ynew[i][0])

    df = pd.DataFrame({'images_id': image_ids,'labels': a})
    bh = []
    bh1 = np.zeros(len(test))

    for i in range(len(test)):
        bh.append(test.loc[i, 'images_id'])

        for j in range(len(df)):
            if test.loc[i, 'images_id'] == df.loc[j, 'images_id']:
                bh1[i] = df.loc[j, 'labels']


    df1 = pd.DataFrame({'images_id': bh,'labels': bh1})
    df1.to_csv(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\Trained_models\labelled_further_' + input + '.csv')
    print ('saved the predictions')

predictor('floorplan')
predictor('face')
predictor('map')
predictor('siteplan')
predictor('property')
predictor('collage')
predictor('text')