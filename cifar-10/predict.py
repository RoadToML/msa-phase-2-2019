from tensorflow.keras.models import load_model
from scipy import misc
from io import BytesIO

import requests
import numpy as np
import cv2

def open_image(url):

    img = requests.get(url)
    img_arr = misc.imread(BytesIO(img.content), mode='RGB')
    img_arr = img_process(img_arr)

    return img_arr

def img_process(img_array):
    return np.reshape(np.array(cv2.resize(img_array, (32, 32)))/255., (-1, 32, 32, 3))

def predict(url):

    model = load_model('model.h5')

    img_arr = open_image(url)

    hist = model.predict(img_arr)

    # list object with classes
    label_word = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    return label_word[np.argmax(hist)]


if __name__ == '__main__':

    predict('https://cdn-prod.medicalnewstoday.com/content/images/articles/322/322868/golden-retriever-puppy.jpg')
