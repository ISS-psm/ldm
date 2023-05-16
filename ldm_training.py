# -*- coding: utf-8 -*-
"""
 Training LDM with Convolutional Neural Network (CNN)
 Classify with: 1. Sigmoid function and 2.SVM
 Trained on 2007 C1 and 2008 C1

@ author: Daniel 
"""



# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__
# Part 1 - Data Preprocessing

# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset2007/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255.)
test_set = test_datagen.flow_from_directory('dataset2007/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Step 2 - Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=1))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=1))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection
cnn.add(tf.keras.layers.Dense(units=64, activation='relu'))
# Step 5 - Output Layer
# Classification with
   # 1. logisitc function
# cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
   # 2. SVM
from tensorflow.keras.regularizers import l2
cnn.add(tf.keras.layers.Dense(units=1, kernel_regularizer=l2(0.01), activation = "linear"))
# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) ## sigmoid
# opt = tf.optimizers.Adam(learning_rate=1e-4)
# cnn.compile(optimizer = 'sgd' , loss = 'squared_hinge', metrics = ['accuracy'])  ## SVM
#cnn.summary()
# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 90)



# Part 4 - Making a single prediction

import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from os import listdir

images=listdir('dataset/prediction/')
for img in images:
    path_image='dataset2/prediction/'+ img
    test_image = load_img(path_image, target_size = (64, 64))
    test_image = load_img('img9.jpg', target_size = (64, 64))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = cnn.predict(test_image,verbose = 0)
    if result[0][0] > 0.7:
        prediction = '--- > simple'
    else:
        prediction = ' --- > complex'
    print(prediction )

# SAVE MODEL
cnn.save("ldm_2007_64n_40iter.h5")
# load model
cnn = tf.keras.models.load_model('ldm_2007c1_SVM.h5')
