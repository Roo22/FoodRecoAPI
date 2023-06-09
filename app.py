from flask import Flask, request, render_template, url_for, jsonify
from tensorflow.python.keras.models import load_model
from keras.layers import BatchNormalization, Layer
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import preprocess_input
from PIL import Image
import numpy as np
from tensorflow.python.keras import backend as K

import tensorflow as tf

class CustomBatchNormalization(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomBatchNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zeros',
            name='beta'
        )
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='ones',
            name='gamma'
        )

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=-1, keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + tf.keras.backend.epsilon())
        return self.gamma * normalized + self.beta
app = Flask(__name__)

#def preprossing(image):
    #image=Image.open(image)
    #image = image.resize((240, 240))
    #image_arr = np.array(image.convert('RGB'))
    #image_arr.shape = (1, 240, 240, 3)
    #return image_arr
data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

def preprossing(image):
    img = Image.open(image)
    img = img.resize((240, 240))
    img_arr = np.array(img).astype(np.float32)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = data_gen.standardize(img_arr)
    return img_arr
classes = ['Coriander','FreshApple','FreshBanana','FreshBittergourd','FreshCapsicum','FreshOrange','FreshTomato','Parsley',
           'StaleApple','StaleBanana','StaleBittergourd','StaleCapsicum','StaleOrange','StaleTomato']
#classes=['StaleBanana','FreshTomato','FreshBanana','Parsley','Coriander','StaleTomato']
model = tf.keras.models.load_model('recoo.h5', custom_objects={'CustomBatchNormalization': CustomBatchNormalization})
@app.route('/')
def index():

    return render_template('index.html', appName="Food Classification")


@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        image_arr = preprossing(image)
        print("Model predicting ...")
        result = model.predict(image_arr)
        print("Model predicted")
        ind = np.argmax(result)
        prediction = classes[ind]
        print(prediction)
        return jsonify({'prediction': prediction})
    except:
        return jsonify({'Error': 'Error occur'})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        image = request.files['fileup']
        print("image loaded....")
        image_arr= preprossing(image)
        print("predicting ...")
        result = model.predict(image_arr)
        print("predicted ...")
        ind = np.argmax(result)
        prediction = classes[ind]

        print(prediction)

        return render_template('index.html', prediction=prediction, image='static/IMG/', appName="Food Classification")
    else:
        return render_template('index.html',appName="Food Classification")


if __name__ == '__main__':
    app.run(debug=True)
