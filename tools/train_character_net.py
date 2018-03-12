from __future__ import print_function

import numpy as np

import sys

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras.backend as K

import dataset_reader

from keras.layers import Input, Reshape, Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Activation, Conv2D, MaxPooling2D, PReLU
from keras.layers.convolutional import Convolution2D

from keras.models import Model
from keras import regularizers

from keras import optimizers
from keras.models import Model, Sequential
from keras import losses
from keras import metrics
import keras

from spatial_transformer import SpatialTransformer

import scipy.misc

gpu_memory_usage = 0.5

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_usage
set_session(tf.Session(config=config))

trnData, trnChars, _, tstData, tstChars, _ = dataset_reader.read_character_position(directory="../OutputsPositions/")

def chars_to_int(chars):
    new_chars = []
    for value in chars:
        new_value = ord(value) - ord('a')
        new_chars.append(new_value)

    return new_chars


trnChars = chars_to_int(trnChars)
tstChars = chars_to_int(tstChars)


def mapLabelsOneHot(labels, classes):
    data = np.asarray(labels)
    out = np.zeros((data.shape[0], classes)).astype(np.float32)
    out[range(out.shape[0]), data.astype(int)] = 1
    return out

trnChars = mapLabelsOneHot(trnChars, 26)
tstChars = mapLabelsOneHot(tstChars, 26)

trnData = trnData.astype(np.float32) / 255.0 - 0.5
tstData = tstData.astype(np.float32) / 255.0 - 0.5

w_decay = 0.0001
w_reg = regularizers.l2(w_decay)

def build_model(shapes):
    input_shape = shapes[0]
    output_shape = shapes[1]

    # STN
    # b = np.zeros((2, 3), dtype='float32')
    # b[0, 0] = 1
    # b[1, 1] = 1
    b = np.array([1, 0, 0])
    W = np.zeros((128, 3), dtype='float32')
    weights = [W, b.flatten()]

    locnet = Sequential()
    
    locnet.add(MaxPooling2D(pool_size=(1,1), input_shape=input_shape))
    
    locnet.add(Convolution2D(8, (2, 2)))
    locnet.add(Convolution2D(8, (2, 2)))
    locnet.add(MaxPooling2D(pool_size=(2,2)))

    locnet.add(Convolution2D(16, (2, 2)))
    locnet.add(Convolution2D(16, (2, 2)))
    locnet.add(MaxPooling2D(pool_size=(2,2)))

    locnet.add(Flatten())
    
    locnet.add(Dense(128))
    locnet.add(Activation('relu'))
    locnet.add(Dropout(rate=0.3))
    
    locnet.add(Dense(128))
    locnet.add(Activation('relu'))
    locnet.add(Dropout(rate=0.3))

    # locnet.add(Dense(6, weights=weights))
    locnet.add(Dense(3, weights=weights))

    # MAIN MODEL
    model = Sequential()

    model.add(SpatialTransformer(localization_net=locnet, output_size=(32,32), input_shape=input_shape))

    model.add(Conv2D(16, (2, 2), padding='same'))#, input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (2, 2), padding='same'))
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (2, 2), padding='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (2, 2), padding='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))
    
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(rate=0.5))

    model.add(Dense(output_shape))
    model.add(Activation('softmax'))

    
    
    
    # image_input = Input(shape=shapes[0], name='image_input')
    
    # net = Conv2D(filters=16, kernel_size=2, activation='relu', padding='same')(image_input)
    # net = Conv2D(filters=16, kernel_size=2, activation='relu', padding='same')(net)
    # net = Conv2D(filters=16, kernel_size=2, activation='relu', padding='same')(net)
    # net = MaxPooling2D(pool_size=2)(net)
    # net = Conv2D(filters=32, kernel_size=2, activation='relu', padding='same')(net)
    # net = Conv2D(filters=32, kernel_size=2, activation='relu', padding='same')(net)
    # net = Conv2D(filters=32, kernel_size=2, activation='relu', padding='same')(net)
    # net = MaxPooling2D(pool_size=2)(net)
    # net = Conv2D(filters=64, kernel_size=2, activation='relu', padding='same')(net)
    # net = Conv2D(filters=64, kernel_size=2, activation='relu', padding='same')(net)
    # net = Conv2D(filters=64, kernel_size=2, activation='relu', padding='same')(net)
    # net = MaxPooling2D(pool_size=2)(net)

    # net = Flatten()(net)

    # net = Dense(1024, activation='relu')(net)
    # net = Dropout(rate=0.5)(net)

    # net = Dense(1024, activation='relu')(net)
    # net = Dropout(rate=0.5)(net)
    
    # net = Dense(shapes[1], name='out', activation='softmax')(net)

    # model = Model(inputs=[image_input], outputs=[net])

    return model, locnet

shapes = [trnData.shape[1:], trnChars.shape[1]]
model, locnet = build_model(shapes)

print('Model')
locnet.summary()
model.summary()

model.compile(
    loss=losses.categorical_crossentropy,
    optimizer=optimizers.Adam(lr=0.0001),
    metrics=[metrics.categorical_accuracy])

model.fit(
    x=trnData, y=trnChars,
    batch_size=48, epochs=100, verbose=1,
    validation_data=[tstData, tstChars], shuffle=True)

# predicts = model.predict(x=tstData)

# print("Predicted > correct")
# for index, probabilities in enumerate(predicts):
#     character = chr(np.argmax(probabilities) + ord('a'))
#     correct = chr(np.argmax(tstChars[index]) + ord('a'))
#     print(character + " > " + correct)


def write_image(img, path):
    # print(img.shape)
    scipy.misc.toimage(img, cmin=0.0, cmax=255.0).save(path)
    # im = Image.fromarray(img)
    # if im.mode != 'RGB':
    #     im = im.convert('RGB')

    # im.save(path)

inp = model.input
outputs = [model.layers[0].output]
functor = K.function([inp] + [K.learning_phase()], outputs)

np.set_printoptions(formatter={'float': '{: 0.5f}'.format})
for index, test_image in enumerate(tstData):
    tstImage = np.expand_dims(test_image, axis=0)
    layer_outs = functor([tstImage, 1.])

    print(str(index), locnet.predict(tstImage)[0])

    tstImage = (tstImage[0] + 0.5) * 255
    outImage = (layer_outs[0][0] + 0.5) * 255

    tstImage = tstImage.astype(int)
    outImage = outImage.astype(int)

    write_image(tstImage, str(index) + "stn_input.png")
    write_image(outImage, str(index) + "stn_output.png")

