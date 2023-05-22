# -*- coding: utf-8 -*-
"""
 Training LDM with Convolutional Neural Network (CNN)
 Classify with: 1. Sigmoid function and 2.SVM
 Trained on 2007 C1 and 2008 C1

@ author: Daniel Dumitru

   date               changelog
-------------      --------------
 22.05.2023         Original code
 
"""

# Importing the libraries
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator


# Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset2008/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255.)
test_set = test_datagen.flow_from_directory('dataset2008/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')



def build_model():

    cnn = tf.keras.models.Sequential()
    
    #  Convolution
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
    
    # Pooling
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=1))
    
    # Adding a second convolutional layer
    cnn.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu')) # filters = 16,32
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=1))
    
    # Flattening
    cnn.add(tf.keras.layers.Flatten())
    
    # Full Connection
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    
    # Output Layer    
    lambda_= 1 # set the regularizer, lambda > 0 
    cnn.add(tf.keras.layers.Dense(units=1, 
                                  kernel_regularizer=l2(lambda_), activation = "linear")) # SVM
    # Training the CNN
    opt = tf.optimizers.Adam(learning_rate=1e-4)
    
    # Classification with logistic function
    #cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Classification with SVM
    cnn.compile(optimizer = opt , loss = 'squared_hinge', metrics = ['accuracy'])
    
    # Train the model
    cnn.fit(x = training_set, validation_data = test_set, epochs = 40)

    return cnn

cnn = build_model()
#cnn.summary()

# SAVE MODEL
cnn.save("YOUR_MODEL_NAME.h5")
