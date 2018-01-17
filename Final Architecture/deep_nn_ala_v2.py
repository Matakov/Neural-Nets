# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:00:35 2018

@author: TOSHIBA
"""

from __future__ import print_function
from keras.utils.data_utils import get_file
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, AveragePooling3D
from keras.layers import Input
from keras import backend as K
from keras.utils.conv_utils import convert_kernel


from keras.utils.layer_utils import convert_all_kernels_in_model

TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

def VGG16():
    model = Sequential()
    # 1
    model.add(Conv2D(4, (3, 3), activation='relu', padding='same', name='block1_conv1',  input_shape = (128, 128, 1)))
    model.add(Conv2D(4, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(AveragePooling2D((2, 2), strides=(2, 2), name='block1_pool'))
    # 2
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(AveragePooling2D((2, 2), strides=(2, 2), name='block2_pool'))
    # 3
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(AveragePooling2D((2, 2), strides=(2, 2), name='block3_pool'))
    # FC Block
    model.add(Flatten(name='flatten'))
    model.add(Dense(512, activation='relu', name='fc1'))
    model.add(Dense(512, activation='relu', name='fc2'))
    model.add(Dense(1, activation='linear', name='fc3'))
    model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=['acc'])

    return model
