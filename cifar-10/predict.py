from tensorflow.keras.models import load_model, Model
from scipy import misc
from io import BytesIO

import matplotlib.pyplot as plt
import requests
import numpy as np
import cv2
import base64

def open_image(url):

    img = requests.get(url)
    img_arr = misc.imread(BytesIO(img.content), mode='RGB')
    img_arr = img_process(img_arr)

    return img_arr

def img_process(img_array):
    return np.reshape(np.array(cv2.resize(img_array, (32, 32)))/255., (-1, 32, 32, 3))

def fig_to_base64(fig):

    img = BytesIO()
    fig.savefig(img, format='png',
                bbox_inches='tight')
    img.seek(0)

    return base64.b64encode(img.getvalue())

def feature_map(img_array, model):
    feature_model = Model(inputs=model.inputs, outputs=model.layers[0].output)

    feature_maps = feature_model.predict(img_array)

    for _ in range(1):
        for _ in range(1):

            fig, ax = plt.subplots()

            ax.set_xticks([])
            ax.set_yticks([])

            plt.imshow(feature_maps[0, :, :, 1], cmap='rainbow')

    encoded = fig_to_base64(fig)
    my_html = "data:image/png;base64,{}".format(encoded.decode('ascii'))

    return my_html

def my_predict(url, visualise = False):

    model = load_model('model.h5')

    img_arr = open_image(url)

    if visualise:

        return feature_map(img_arr, model)

    elif not visualise:
        hist = model.predict(img_arr)

        sorted_vals = np.argsort(hist)[0][-3:][::-1]

        # list object with classes
        all_labels = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer',
                      5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

        classify_list = [[all_labels[i], str(hist[0][i]*100)[:5]+'%'] for i in sorted_vals]

        return classify_list

if __name__ == '__main__':

    my_predict('https://cdn-prod.medicalnewstoday.com/content/images/articles/322/322868/golden-retriever-puppy.jpg')
