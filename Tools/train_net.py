import numpy as np

import sys

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

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

trnData, trnLabels, tstData, tstLabels = dataset_reader.read(directory="../Outputs/", target_size=(128,128))


def mapLabelsOneHot(labels):
    data = np.asarray(labels)
    class_no = int(data.max() + 1)
    out = np.zeros((data.shape[0], class_no)).astype(np.float32)
    out[range(out.shape[0]), data.astype(int)] = 1
    return out


trnLabels = mapLabelsOneHot(trnLabels)
tstLabels = mapLabelsOneHot(tstLabels)

trnData = trnData.astype(np.float32) / 255.0 - 0.5
tstData = tstData.astype(np.float32) / 255.0 - 0.5

w_decay = 0.0001
w_reg = regularizers.l2(w_decay)


def build_VGG_block(net, channels, layers, prefix):
    for i in range(layers):
        net = Conv2D(channels, 3, activation='relu', padding='same', name='{}.{}'.format(prefix, i))(net)
    net = MaxPooling2D(2, 2, padding="same")(net)
    return net


def build_VGG(input_data, block_channels=[16, 32, 64], block_layers=[2, 2, 2], fcChannels=[256, 256], p_drop=0.4):
    net = input_data
    for i, (cCount, lCount) in enumerate(zip(block_channels, block_layers)):
        net = build_VGG_block(net, cCount, lCount, 'conv{}'.format(i))

    net = Flatten()(net)

    for i, cCount in enumerate(fcChannels):
        FC = Dense(cCount, activation='relu', name='fc{}'.format(i))
        net = Dropout(rate=p_drop)(FC(net))

    net = Dense(10, name='out', activation='softmax')(net)

    return net


def build_VGG_Bnorm_block(net, channels, layers, prefix):
    for i in range(layers):
        net = Conv2D(channels, 3, padding='same', name='{}.{}'.format(prefix, i))(net)
        net = BatchNormalization()(net)
        net = PReLU()(net)
    net = MaxPooling2D(2, 2, padding="same")(net)
    return net


def build_VGG_Bnorm(input_data, block_channels=[16, 32, 64], block_layers=[2, 2, 2], fcChannels=[256, 256], p_drop=0.4):
    net = input_data
    for i, (cCount, lCount) in enumerate(zip(block_channels, block_layers)):
        net = build_VGG_Bnorm_block(net, cCount, lCount, 'conv{}'.format(i))
        net = Dropout(rate=0.25)(net)

    net = Flatten()(net)

    for i, cCount in enumerate(fcChannels):
        net = Dense(cCount, name='fc{}'.format(i))(net)
        net = BatchNormalization()(net)
        net = PReLU()(net)

        net = Dropout(rate=p_drop)(net)

    net = Dense(186, name='out', activation='softmax')(net)

    return net


input_data = Input(shape=(trnData.shape[1:]), name='data')
net = build_VGG_Bnorm(input_data, block_channels=[64, 128, 256], block_layers=[3, 3, 3], fcChannels=[320, 320], p_drop=0.5)
model = Model(inputs=[input_data], outputs=[net])

print('Model')
model.summary()

model.compile(
    loss=losses.categorical_crossentropy,
    optimizer=optimizers.Adam(lr=0.001),
    metrics=[metrics.categorical_accuracy])

tbCallBack = keras.callbacks.TensorBoard(
    log_dir='./Graph',
    histogram_freq=1,
    write_graph=True, write_images=True)

model.fit(
    x=trnData, y=trnLabels,
    batch_size=48, epochs=20, verbose=1,
    validation_data=[tstData, tstLabels], shuffle=True)  # , callbacks=[tbCallBack])

classProb = model.predict(x=tstData[0:2])
print('Class probabilities:', classProb, '\n')
loss, acc = model.evaluate(x=tstData, y=tstLabels, batch_size=1024)
print()
print('loss', loss)
print('acc', acc)
