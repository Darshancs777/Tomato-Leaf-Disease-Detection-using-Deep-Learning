from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# Basic CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=10, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define batch sizes
batch_size_train = 32
batch_size_valid = 16
batch_size_test = 16

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('D:\\test\\Dataset\\train',
                                                 target_size=(128, 128),
                                                 batch_size=batch_size_train,
                                                 class_mode='categorical')
valid_set = test_datagen.flow_from_directory('D:\\test\\Dataset\\valid',
                                             target_size=(128, 128),
                                             batch_size=batch_size_valid,
                                             class_mode='categorical')

# Prepare the test set
test_set = test_datagen.flow_from_directory('D:\\test\\Dataset\\test',
                                            target_size=(128, 128),
                                            batch_size=batch_size_test,
                                            class_mode='categorical')

labels = training_set.class_indices
print(labels)

# Calculate steps_per_epoch and validation_steps based on the dataset size
steps_per_epoch = int(np.ceil(training_set.samples / training_set.batch_size))
validation_steps = int(np.ceil(valid_set.samples / valid_set.batch_size))
test_steps = int(np.ceil(test_set.samples / test_set.batch_size))

# Fit the model
classifier.fit(training_set,
               steps_per_epoch=steps_per_epoch,
               epochs=50,
               validation_data=valid_set,
               validation_steps=validation_steps)

# Evaluate the model on the validation set
val_loss, val_accuracy = classifier.evaluate(valid_set, steps=validation_steps)
print(f'Validation accuracy: {val_accuracy:.4f}')

# Evaluate the model on the test set

# Save the model
classifier_json = classifier.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(classifier_json)
classifier.save("model.h5")
print("Saved model to disk")
