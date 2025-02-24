import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", required=True, help="Directory containing images to classify")
args = vars(parser.parse_args())

# Load the pre-trained model
model = tf.keras.models.load_model("Models")

# Prepare list to hold preprocessed images
image_list = []

# Load and preprocess images from the specified directory
for filename in os.listdir(args["directory"]):
    if not filename.startswith("."):  # Skip hidden files
        image_path = os.path.join(args["directory"], filename)
        img = image.load_img(image_path, target_size=(224, 224))
        image_array = image.img_to_array(img)
        image_array = np.expand_dims(image_array, axis=0)
        preprocessed_image = tf.keras.applications.vgg16.preprocess_input(image_array)
        image_list.append(preprocessed_image)

# Stack images into a single array for batch prediction
images = np.vstack(image_list)

# Make predictions
predictions = model.predict(images)

# Define class names for the weather categories
class_names = ['cloudy', 'foggy', 'rainy', 'shine', 'sunrise']

# Print the predicted class for each image
for i, prediction in enumerate(predictions):
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    print(f"Image {i+1}: Predicted class - {predicted_class_name}")
