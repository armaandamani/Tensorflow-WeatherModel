import tensorflow as tf
import tensorflow.keras.utils as utils
import tensorflow.keras.applications as applications
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras import losses, optimizers
import matplotlib.pyplot as plt

# Load the training and validation datasets
train = utils.image_dataset_from_directory(
    'Weather',
    label_mode='categorical',
    image_size=(224, 224),
    seed=8080,
    validation_split=0.30,
    subset='training',
    batch_size=32
)

validation = utils.image_dataset_from_directory(
    'Weather',
    label_mode='categorical',
    image_size=(224, 224),
    seed=8080,
    validation_split=0.30,
    subset='validation',
    batch_size=32
)

# Preprocess datasets for VGG16
train = train.map(lambda x, y: (applications.vgg16.preprocess_input(x), y))
validation = validation.map(lambda x, y: (applications.vgg16.preprocess_input(x), y))

# Load the VGG16 base model
base_model = applications.vgg16.VGG16(
    include_top=False,
    weights='vgg16_weights.h5',
    input_shape=(224, 224, 3)
)

base_model.trainable = False

# Define the custom model
inputs = layers.Input(shape=(224, 224, 3))
outputs = base_model(inputs, training=False)
outputs = layers.Flatten()(outputs)
outputs = layers.Dense(5, activation='softmax')(outputs)
WeatherClassifier = keras.Model(inputs, outputs)

# Define optimizer and loss function
optimizer = optimizers.legacy.Adam(learning_rate=0.00001)
loss_fn = losses.CategoricalCrossentropy()

# Compile the model
WeatherClassifier.compile(
    optimizer=optimizer,
    loss=loss_fn,
    metrics=['accuracy']
)

# Train the model
history = WeatherClassifier.fit(
    train,
    epochs=10,
    verbose=1,
    validation_data=validation
)

# Save the model
WeatherClassifier.save("weather_classifier")

# Plot training results
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Weather Classifier Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Weather Classifier Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
