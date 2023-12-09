import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()

ap.add_argument("-d", "--directory", required = True)
args = vars(ap.parse_args())

images = []

model = tf.keras.models.load_model("Models")

for filename in os.listdir(args["directory"]):
    if filename [0] != ".":
        image_path = os.path.join(args["directory"], filename)
        img = image.load_img(image_path, target_size = (224, 224))
        image_array = image.img_to_array(img)
        image_array = np.expand_dims(image_array, axis = 0)
        images.append(tf.keras.applications.vgg16.preprocess_input(image_array))

images = np.vstack(images)

predictions = model.predict(images)

class_names = ['cloudy', 'foggy', 'rainy', 'shine', 'sunrise']
for i, predicted_class in enumerate(predictions):
    predicted_class_name = class_names[np.argmax(predicted_class)]
    print (f"image {i+1}: Class - {predicted_class_name}")