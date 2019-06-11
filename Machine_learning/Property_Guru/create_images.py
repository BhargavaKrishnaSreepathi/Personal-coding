from PIL import Image, ImageFilter
import pandas as pd


path_to_image_location = r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\all_images\image_moderation_images/'
train = pd.read_csv(r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\test_set.csv')  # reading the csv file

df = pd.DataFrame(columns=['labels', 'images_id'])

for i in range(len(train)):

    img = Image.open(path_to_image_location + str(train.loc[i, 'images_id']))
    r_180 = 'r_180_' + str(train.loc[i, 'images_id'])
    r_90 = 'r_90_' + str(train.loc[i, 'images_id'])
    r_45 = 'r_45_' + str(train.loc[i, 'images_id'])
    r_135 = 'r_135_' + str(train.loc[i, 'images_id'])
    r_270 = 'r_270_' + str(train.loc[i, 'images_id'])
    blur = 'blur_' + str(train.loc[i, 'images_id'])
    img.rotate(180).save(path_to_image_location + r_180)
    img.rotate(90).save(path_to_image_location + r_90)
    img.rotate(45).save(path_to_image_location + r_45)
    img.rotate(135).save(path_to_image_location + r_135)
    img.rotate(270).save(path_to_image_location + r_270)
    img.filter(ImageFilter.BLUR).save(path_to_image_location + blur)

    df1 = pd.DataFrame({'images_id':[r_180, r_90, r_45, r_135, r_270, blur]})
    df = df.append(df1)

df.to_csv(path_to_image_location + 'updated_testing.csv')