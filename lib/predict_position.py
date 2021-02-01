from tensorflow.keras.models import load_model
import cv2
import sys
import os
import numpy as np
import multiprocessing

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
category_names = ['Empty', 'Straight', 'Tilted']
IMG_HEIGHT = 150
IMG_WIDTH = 150


def prepare_image(img_name):
    input_image = cv2.imread(img_name)
    image_resize = cv2.resize(
        input_image, (IMG_HEIGHT, IMG_WIDTH))
    image_resize = image_resize / 255
    image_reshape = image_resize.reshape(
        (1, IMG_HEIGHT, IMG_WIDTH, 3))
    return image_reshape


def predict_image(img):
    new_model = load_model(
        sys.argv[1] + '/puck_visualization_model_25Sep20.h5')
    prediction = np.argmax(new_model.predict(
        [prepare_image(img)]), axis=-1)
    print(category_names[prediction[0]])


for i in range(1, 17):
    img = sys.argv[2] + '/' + sys.argv[3] + '_' + str(i) + '.jpg'
    print('Postion: ' + str(i))
    p1 = multiprocessing.Process(target=predict_image(img))
    p1.start()
# print(p1)
