import numpy as np

import sys

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras.backend as K

import dataset_reader

from keras.layers import Input, Reshape, Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Activation, Conv2D, MaxPooling2D, PReLU
from keras.models import Model
from keras import regularizers

from keras import optimizers
from keras.models import Model
from keras import losses
from keras import metrics
import keras

gpu_memory_usage = 0.5

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_usage
set_session(tf.Session(config=config))

trnData, trnChars, trnDeltas, tstData, tstChars, tstDeltas = dataset_reader.read_character_position(directory="../OutputsPositions/")


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


trnDeltas = mapLabelsOneHot(trnDeltas, 32)
tstDeltas = mapLabelsOneHot(tstDeltas, 32)

trnChars = mapLabelsOneHot(trnChars, 26)
tstChars = mapLabelsOneHot(tstChars, 26)

trnData = trnData.astype(np.float32) / 255.0 - 0.5
tstData = tstData.astype(np.float32) / 255.0 - 0.5

w_decay = 0.0001
w_reg = regularizers.l2(w_decay)

def build_model(shapes):
    image_input = Input(shape=shapes[0], name='image_input')
    character_input = Input(shape=shapes[1], name='character_input')

    net = Conv2D(filters=16, kernel_size=2, activation='relu', padding='same')(image_input)
    net = Conv2D(filters=16, kernel_size=2, activation='relu', padding='same')(net)
    net = Conv2D(filters=16, kernel_size=2, activation='relu', padding='same')(net)
    net = MaxPooling2D(pool_size=2)(net)
    net = Conv2D(filters=32, kernel_size=2, activation='relu', padding='same')(net)
    net = Conv2D(filters=32, kernel_size=2, activation='relu', padding='same')(net)
    net = Conv2D(filters=32, kernel_size=2, activation='relu', padding='same')(net)

    net = Flatten()(net)
    net = keras.layers.concatenate([net, character_input])

    net = Dense(1024, activation='relu')(net)
    net = Dropout(rate=0.5)(net)

    net = Dense(1024, activation='relu')(net)
    net = Dropout(rate=0.5)(net)
    
    net = Dense(shapes[2], name='out', activation='softmax')(net)

    model = Model(inputs=[image_input, character_input], outputs=[net])

    return model

shapes = [trnData.shape[1:], trnChars.shape[1:], trnDeltas.shape[1]]
model = build_model(shapes)

print('Model')
model.summary()

model.compile(
    loss=losses.categorical_crossentropy,
    optimizer=optimizers.Adam(lr=0.0001),
    metrics=[metrics.categorical_accuracy])

model.fit(
    x={'image_input': trnData, 'character_input': trnChars}, y=trnDeltas,
    batch_size=32, epochs=30, verbose=1,
    validation_data=[{'image_input': tstData, 'character_input': tstChars}, tstDeltas], shuffle=True)

predicts = model.predict(x={'image_input': tstData, 'character_input': tstChars})
err = 0
for index, probabilities in enumerate(predicts):
    err += abs(np.argmax(probabilities) - np.argmax(tstDeltas[index]))

print("Error:", err, err / 35.)

# classProb = model.predict(x=tstData[0:2])
# print('Class probabilities:', classProb, '\n')
# loss, acc = model.evaluate(x=tstData, y=tstLabels, batch_size=1024)
# print()
# print('loss', loss)
# print('acc', acc)
