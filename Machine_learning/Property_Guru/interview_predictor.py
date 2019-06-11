import pandas as pd
from keras.models import model_from_json
import imageio
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

__author__ = "Sreepathi Bhargava Krishna"
__credits__ = ["Sreepathi Bhargava Krishna"]
__email__ = "s.bhargava.krishna@gmail.com"
__status__ = "Made for the Assessment"

# load json and create model
# class_labels = ['text', 'floorplan', 'map', 'face', 'collage', 'property', 'siteplan']


json_file = open(r"C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning\Property_Guru\model_data_validation_final_all_interview_bw_inception_3.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(r"C:\Users\krish\Documents\GitHub\Personal-coding\Machine_learning\Property_Guru\vgg19_3.h5")
print("Loaded model from disk")

test = pd.read_csv(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\test_set.csv')    # reading the csv file
# test = pd.read_csv(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\imgs_train.csv')    # reading the csv file


data = []
image_id = []
predictions_df = []

IMG_SIZE = 299

for i in range(8000, 10000):
    img_original = Image.open(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/' + str(test.loc[i, 'images_id']))
    arr_original = np.array(img_original)/255.0
    # img_180 = Image.open(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/r_180_' + str(test.loc[i, 'images_id']))
    # arr_180 = np.array(img_180)
    # img_90 = Image.open(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/r_90_' + str(test.loc[i, 'images_id']))
    # arr_90 = np.array(img_90)
    # img_45 = Image.open(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/r_45_' + str(test.loc[i, 'images_id']))
    # arr_45 = np.array(img_45)
    # img_135 = Image.open(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/r_135_' + str(test.loc[i, 'images_id']))
    # arr_135 = np.array(img_135)
    # img_270 = Image.open(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/r_270_' + str(test.loc[i, 'images_id']))
    # arr_270 = np.array(img_270)
    # img_blur = Image.open(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation/all_images\image_moderation_images/blur_' + str(test.loc[i, 'images_id']))
    # arr_blur = np.array(img_blur)
    if len(arr_original.shape) > 2:
        if arr_original.shape[2] == 3:
            data.append([arr_original])
            # data.append([arr_180])
            # data.append([arr_90])
            # data.append([arr_45])
            # data.append([arr_135])
            # data.append([arr_270])
            # data.append([arr_blur])


            image_id.append(str(test.loc[i, 'images_id']))
            # data_full = np.array([i[0] for i in data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
            # predictions = loaded_model.predict(data_full)
            # pred_bool = (predictions > 0.4)
            # predictions_df.append(pred_bool.astype(int))

data_full = np.array([i[0] for i in data]).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
image_ids = np.array(image_id)

predictions = loaded_model.predict(data_full)
pred_bool = (predictions >0.1)
predictions = pred_bool.astype(int)

# ynew = loaded_model.predict_classes(testImages)
print ('predictions done')

text = []
floorplan = []
siteplan = []
face = []
map  = []
collage = []
property = []

for i in range(len(predictions)):
    text.append(predictions[i][0])
    floorplan.append(predictions[i][1])
    map.append(predictions[i][2])
    face.append(predictions[i][3])
    collage.append(predictions[i][4])
    property.append(predictions[i][5])
    siteplan.append(predictions[i][6])

df = pd.DataFrame({'images_id': image_ids,'text': text, 'floorplan': floorplan, 'map': map, 'face': face, 'collage': collage, 'property': property, 'siteplan': siteplan})
bh = []
bh1 = np.zeros(len(test))
text1 = np.zeros(len(test))
floorplan1 = np.zeros(len(test))
siteplan1 = np.zeros(len(test))
face1 = np.zeros(len(test))
map1  = np.zeros(len(test))
collage1 = np.zeros(len(test))
property1 = np.zeros(len(test))

for i in range(len(test)):
    bh.append(test.loc[i, 'images_id'])

    for j in range(len(df)):
        if test.loc[i, 'images_id'] == df.loc[j, 'images_id']:
            text1[i] = df.loc[j, 'text']
            floorplan1[i] = df.loc[j, 'floorplan']
            map1[i] = df.loc[j, 'map']
            face1[i] = df.loc[j, 'face']
            collage1[i] = df.loc[j, 'collage']
            property1[i] = df.loc[j, 'property']
            siteplan1[i] = df.loc[j, 'siteplan']


df1 = pd.DataFrame({'images_id': bh,'text': text1, 'floorplan': floorplan1, 'map': map1, 'face': face1, 'collage': collage1, 'property': property1, 'siteplan': siteplan1})
df1.to_csv(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\Trained_models\vgg_F.csv')
print ('saved the predictions')

combined_labels = pd.read_csv(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\Trained_models\vgg_F.csv')

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
        c = 'property text'
        bhar = bhar + 1
    combined_text.append(c)

z = np.array(combined_text)

df = pd.DataFrame({'images_id': combined_labels['images_id'], 'labels': z})
df.to_csv(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\Trained_models\vgg19_submission_F.csv')
print (bhar)
