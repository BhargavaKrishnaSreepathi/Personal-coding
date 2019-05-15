from keras.models import model_from_json
import imageio
import numpy as np

__author__ = "Sreepathi Bhargava Krishna"
__credits__ = ["Sreepathi Bhargava Krishna"]
__email__ = "s.bhargava.krishna@gmail.com"
__status__ = "Made for the Assessment"

json_file = open('model_data_validation_final_all_7.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_data_validation_final_all_7.h5")
print("Loaded model from disk")

path_to_image_location = r'C:\Users\krish\Desktop\Property Guru\pg-image-moderation\all_images\image_moderation_images/'

IMG_SIZE = 299

def predict(list_of_images):

    for i in range(len(list_of_images)):
        output_list = []

        img = imageio.imread(path_to_image_location + list_of_images[i])
        arr = np.array(img)
        if len(arr.shape) > 2:
            testImages = np.array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
            pred = loaded_model.predict(testImages)

            pred_bool = (pred > 0.4)
            predictions = pred_bool.astype(int)
            predictions = predictions[0]
            if predictions[0] == 1:
                output_list.append('text')

            if predictions[1] == 1:
                output_list.append('floorplan')

            if predictions[2] == 1:
                output_list.append('map')

            if predictions[3] == 1:
                output_list.append('face')

            if predictions[4] == 1:
                output_list.append('collaged')

            if predictions[5] == 1:
                output_list.append('property')

            if predictions[6] == 1:
                output_list.append('siteplan')


            if len(output_list) == 0:
                print ('couldnt recognize any class')
            else:
                print (output_list)
        else:
            print ('It is not a color image, this classifier works only for color images')


if __name__ == '__main__':

    list_of_images = ['3853906.jpg', '3869890.jpg', '18016960.jpg', '35914048.jpg', '52722208.jpg', '55468423.jpg', '38262437.jpg']
    predict(list_of_images)



