import tensorflow as tf
import pandas as pd
import numpy as np
import os
import cv2

from tensorflow import keras
from keras import layers
from glob import glob
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

import warnings
warnings.filterwarnings('ignore')


# Hyperparameters
IMG_SIZE = 256
SPLIT = 0.2
EPOCHS = 25
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 10


# Data Import
X = []
Y = []

images_dir = 'cell_images/cell_images/'

classes = os.listdir(images_dir)


# Data Preprocessing
for i, name in enumerate(classes):
    images = glob(f'{images_dir}/{name}/*.png')

    for image in images:
        img = cv2.imread(image)

        X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
        Y.append(i)

X = np.asarray(X)
one_hot_encoded_Y = pd.get_dummies(Y).values


# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, one_hot_encoded_Y, test_size= SPLIT, random_state= 42)


# Creating Modell Based On EfficientNet
base_model = keras.applications.efficientnet.EfficientNetB3(include_top= False, weights= 'imagenet', input_shape= IMG_SHAPE, pooling= 'max')

model = keras.Sequential([
    base_model,
    layers.BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
    layers.Dense(256, activation= 'relu'),
    layers.Dropout(rate= 0.45, seed= 123),
    layers.Dense(2, activation= 'softmax')
])

model.compile(optimizer= 'adam', loss= 'categorical_crossentropy', metrics= ['accuracy'])
print(model.summary())


# Callbacks
checkpoint = ModelCheckpoint('output/model_checkpoint.h5',
                             monitor= 'val_accuracy',
                             verbose= 1,
                             save_best_only= True,
                             save_weights_only= True,
                             )


# Model Training
model.fit(X_train, Y_train,
          batch_size= BATCH_SIZE,
          verbose= 1,
          epochs= EPOCHS,
          callbacks= checkpoint,
          validation_data=(X_test, Y_test)
          )