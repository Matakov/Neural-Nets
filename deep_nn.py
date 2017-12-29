from __future__ import print_function
from keras.utils.data_utils import get_file
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input
from keras import backend as K
from keras.utils.conv_utils import convert_kernel
import tensorflow as tf
import warnings
from keras.utils.layer_utils import convert_all_kernels_in_model

TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

def VGG16():
    model = Sequential()
    # 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1',  input_shape = (128, 128, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

#    #Load Weights
#    print('K.image_dim_ordering:', K.image_dim_ordering())
#    if K.image_dim_ordering() == 'th':
#        weights_path = get_file('vgg16_weights_th_dim_ordering_th_kernels_notop.h5',
#                                TH_WEIGHTS_PATH_NO_TOP,
#                                cache_subdir='models')
#        model.load_weights(weights_path)
#        if K.backend() == 'tensorflow':
#            warnings.warn('You are using the TensorFlow backend, yet you '
#                          'are using the Theano '
#                          'image dimension ordering convention '
#                          '(`image_dim_ordering="th"`). '
#                          'For best performance, set '
#                          '`image_dim_ordering="tf"` in '
#                          'your Keras config '
#                          'at ~/.keras/keras.json.')
#            convert_all_kernels_in_model(model)
#    else:
#        weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
#                                TF_WEIGHTS_PATH_NO_TOP,
#                                cache_subdir='models')
#        model.load_weights(weights_path)
#        if K.backend() == 'theano':
#            convert_all_kernels_in_model(model)

    # FC Block
    model.add(Flatten(name='flatten'))
    model.add(Dense(1024, activation='relu', name='fc1'))
    model.add(Dense(1024, activation='relu', name='fc2'))
    model.add(Dense(1, activation='linear', name='fc3'))

    model.compile(loss='mean_squared_error', optimizer=Adam())
    return model
