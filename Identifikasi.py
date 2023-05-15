from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import csv
from keras.preprocessing import image
from tensorflow import keras
from tensorflow import lite

img_width, img_height = 224,224
nb_train_samples = 200
nb_validation_samples = 200
epochs = 2
batch_size = 10
if K.image_data_format()=='channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width,img_height,3)
    
#Membuat model
model = Sequential()
model.add(Conv2D(28, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(56, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(56, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(56, activation='relu'))
model.add(Dense(62))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
print('Compiling Model.......')
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

train_datagen = ImageDataGenerator(rescale=1.0/255.)
test_datagen = ImageDataGenerator(rescale=1.0/255.)

train_generator = train_datagen.flow_from_directory('dataset/train/',
                                                    batch_size=batch_size,
                                                    class_mode='binary',
                                                    target_size=(224,224))
validation_generator = test_datagen.flow_from_directory('dataset/validation/',
                                                    batch_size=batch_size,
                                                    class_mode='binary',
                                                    target_size=(224,224))

history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=10, 
                    epochs=2,
                    validation_steps=2,
                    verbose = 1)

from matplotlib import pyplot as plt
# Plot history: MAE
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.ylabel('value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()


# Plot history: MSE
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()

#Membuat file h5
MODEL_BASE_PATH = "model"
PROJECT_NAME = "batuan"
SAVE_MODEL_NAME = "model_batuan.h5"
save_model_path = os.path.join(MODEL_BASE_PATH, PROJECT_NAME, SAVE_MODEL_NAME)

if os.path.exists(os.path.join(MODEL_BASE_PATH, PROJECT_NAME)) == False:
    os.makedirs(os.path.join(MODEL_BASE_PATH, PROJECT_NAME))
    
print('Saving Model At {}...'.format(save_model_path))
model.save(save_model_path,include_optimizer=False)    
    