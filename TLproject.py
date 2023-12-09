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
    label_mode = 'categorical',
    image_size = (224, 224),
    seed = 8080,
    validation_split = 0.30,
    subset = 'training',
    batch_size = 32
)

validation = utils.image_dataset_from_directory(
    'Weather',
    label_mode = 'categorical',
    image_size = (224, 224),
    seed = 8080,
    validation_split = 0.30,
    subset = 'validation',
    batch_size = 32
)

# Organizes dataset for processing
train = train.map(lambda x, y: (applications.vgg16.preprocess_input(x), y))
validation = validation.map(lambda x, y: (applications.vgg16.preprocess_input(x), y))

# Load the VGG16 model
vgg16Model = applications.vgg16.VGG16(
    include_top = False,
    weights = 'vgg16_weights.h5',
    input_shape = (224, 224, 3),
    classifier_activation = 'softmax'
)

vgg16Model.trainable = False

# Create a model

inputs = layers.Input(shape=(224, 224, 3))
outputs = vgg16Model(inputs, training = False)
outputs = layers.Flatten()(outputs)
outputs = layers.Dense(5, activation = 'relu')(outputs)
WeatherModel = keras.Model(inputs, outputs)

# Optimizers and loss functions
optimizer_function = (optimizers.legacy.Adam(learning_rate=0.00001))

loss_function = (losses.CategoricalCrossentropy())

# Compile the model with the optimizer and loss function
WeatherModel.compile(
    optimizer = optimizer_function,
    loss = loss_function, 
    metrics = ['accuracy']
)

# Train the model with the custom callback to print loss and optimizer values
history = WeatherModel.fit(
    train,
    epochs = 10,
    verbose = 1,
    validation_data = validation,
)

WeatherModel.save("Models")

plt.figure(figsize = (12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title(f'Weather Model (Accuracy)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc = 'upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(f'Weather Model (loss)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc = 'upper left')

# Show the plot
plt.show()