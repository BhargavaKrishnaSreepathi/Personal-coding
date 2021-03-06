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
from keras.preprocessing.image import ImageDataGenerator

__author__ = "Sreepathi Bhargava Krishna"
__credits__ = ["Sreepathi Bhargava Krishna"]
__email__ = "s.bhargava.krishna@gmail.com"
__status__ = "Made for the Assessment"

# load json and create model
# class_labels = ['text', 'floorplan', 'map', 'face', 'collage', 'property', 'siteplan']

def predictor(input):
    json_file = open(r'C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning_example\model_data_validation_final_' + input + '_4.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(r"C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning_example\model_data_validation_final_" + input + "_4.h5")
    print("Loaded model from disk")

    test = pd.read_csv(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\test_set.csv')    # reading the csv file
    # test = pd.read_csv(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\imgs_train.csv')    # reading the csv file

    test_data_original = []
    IMG_SIZE = 299
    image_id = []
    black_and_white = []


    for i in range(len(test)):
        # img = Image.open(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/' + str(train.loc[i, 'images_id']))
        # img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
        img = imageio.imread(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/' + str(test.loc[i, 'images_id']))
        arr = np.array(img)
        if len(arr.shape) > 2:
            test_data_original.append([np.array(img)])
            image_id.append(str(test.loc[i, 'images_id']))


    testImages = np.array([i[0] for i in test_data_original]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    image_ids = np.array(image_id)


    datagen = ImageDataGenerator()

    epochs = 100
    batch_size = 100
    epoch_step = len(testImages) / batch_size

    test_generator = datagen.flow(testImages, batch_size=batch_size)

    test_generator.reset()
    pred=loaded_model.predict_generator(test_generator,steps=epoch_step,verbose=1)

    pred_bool = (pred >0.5)
    predictions = pred_bool.astype(int)


    # ynew = loaded_model.predict_classes(testImages)
    print ('predictions done')

    input_df = []


    for i in range(len(predictions)):
        input_df.append(predictions[i][0])


    df = pd.DataFrame({'images_id': image_ids,'text': input_df})
    bh = []
    bh1 = np.zeros(len(test))


    for i in range(len(test)):
        bh.append(test.loc[i, 'images_id'])

        for j in range(len(df)):
            if test.loc[i, 'images_id'] == df.loc[j, 'images_id']:
                bh1[i] = df.loc[j, 'text']


    df1 = pd.DataFrame({'images_id': bh,'text': bh1})
    df1.to_csv(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\Trained_models\labelled_model_data_validation_final_' + input + '_4.csv')
    print ('saved the predictions')
#
predictor('floorplan')
predictor('face')
predictor('map')
predictor('siteplan')
predictor('property')
predictor('collage')
predictor('text')
# labelled_model_data_validation_final_all
combined_labels = pd.read_excel(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\Trained_models\Combined labelling separate_4.xlsx')

combined_text = []
bhar = 0

for i in range(len(combined_labels)):
    c = ''
    if combined_labels.loc[i,'text'] == 1:
        if len(c) == 0:
            c = c + 'text'
        else:
            c = c + ' text'

    if combined_labels.loc[i,'floorplan'] == 1:
        if len(c) == 0:
            c = c + 'floorplan'
        else:
            c = c + ' floorplan'

    if combined_labels.loc[i,'map'] == 1:
        if len(c) == 0:
            c = c + 'map'
        else:
            c = c + ' map'

    if combined_labels.loc[i,'face'] == 1:
        if len(c) == 0:
            c = c + 'face'
        else:
            c = c + ' face'

    if combined_labels.loc[i,'collage'] == 1:
        if len(c) == 0:
            c = c + 'collaged'
        else:
            c = c + ' collaged'

    if combined_labels.loc[i,'property'] == 1:
        if len(c) == 0:
            c = c + 'property'
        else:
            c = c + ' property'

    if combined_labels.loc[i,'siteplan'] == 1:
        if len(c) == 0:
            c = c + 'siteplan'
        else:
            c = c + ' siteplan'

    if len(c) == 0:
        c = 'text'
        bhar = bhar + 1
    combined_text.append(c)

z = np.array(combined_text)

df = pd.DataFrame({'images_id': combined_labels['images_id'], 'labels': z})
df.to_csv(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\Trained_models\submission_separate_4.csv')
print (bhar)