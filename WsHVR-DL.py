from __future__ import print_function
from tensorflow import keras
import numpy as np
from tensorflow.keras import backend as K
import os
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Model
#from keras.callbacks import ReduceLROnPlateau
import matplotlib as mpl
import itertools
import random
#from tensorflow.keras.backend.tensorflow_backend import set_session
from tensorflow.compat.v1.keras.backend import set_session
from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

seed = 7
np.random.seed(seed)
def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == 'channel_first':
        channel = input_feature.shape[1]
        ARCAM_feature = keras.layers.Permute((2, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        ARCAM_feature = input_feature

    avg_pool = keras.layers.Lambda(lambda x: K.mean(x, axis=2, keepdims=True))(ARCAM_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = keras.layers.Lambda(lambda x: K.max(x, axis=2, keepdims=True))(ARCAM_feature)
    assert max_pool.shape[-1] == 1
    concat = keras.layers.Concatenate(axis=2)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    ARCAM_feature = keras.layers.Conv1D(filters=1,
                          kernel_size=kernel_size,
                          activation='hard_sigmoid',
                          strides=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert ARCAM_feature.shape[-1] == 1

    if K.image_data_format() == 'channel_first':
        ARCAM_feature = keras.layers.Permute((2, 1))(ARCAM_feature)

    return keras.layers.multiply([input_feature, ARCAM_feature])

def self_attention(input_feature):
    channel_axis = 1 if K.image_data_format() == 'channel_first' else -1
    channel = input_feature.shape[channel_axis]

    query = keras.layers.Dense(channel // 4)(input_feature)
    key = keras.layers.Dense(channel // 4)(input_feature)
    value = keras.layers.Dense(channel)(input_feature)

    attention_weights = keras.layers.Dot(axes=(2, 2))([query, key])
    attention_weights = keras.layers.Activation('softmax')(attention_weights)
    output = keras.layers.Dot(axes=(2, 1))([attention_weights, value])

    return keras.layers.Add()([input_feature, output])

def ARCAM_block(ARCAM_feature, ratio=8):
    ARCAM_feature = self_attention(ARCAM_feature)
    ARCAM_feature = spatial_attention(ARCAM_feature, )
    return ARCAM_feature

dilations = [1, 2, 4]
nb_filters = 64
filter_length = 3
dropout = 0.3

def build_WsHVR(input_shape, n_feature_maps, nb_classes):
    x = keras.layers.Input(shape=(input_shape))

    conv1 = keras.layers.Conv1D(n_feature_maps , 1, 1, padding='same')(x)
    conv2 = keras.layers.Conv1D(n_feature_maps , 3, 1, padding='same')(x)
    conv0 = keras.layers.Conv1D(n_feature_maps, 5, 1, padding='same')(x)

    concat = keras.layers.Concatenate()([conv1, conv2, conv0])

    ARCAM = ARCAM_block(concat)
    ARCAM = keras.layers.add([concat, ARCAM])


    #conv4 = keras.layers.Conv1D(n_feature_maps * 1, 5, 1, padding='same')(ARCAM)
    #conv4 = keras.layers.BatchNormalization()(conv4)
    #conv4 = keras.layers.Activation('relu')(conv4)
    #conv4 = keras.layers.AveragePooling1D(pool_size=2, padding='same')(conv4)

    #conv5 = keras.layers.Conv1D(n_feature_maps * 2, 8, 1, padding='same')(conv4)
    #conv5 = keras.layers.BatchNormalization()(conv5)
    #conv5 = keras.layers.Activation('relu')(conv5)
    #conv5 = keras.layers.AveragePooling1D(pool_size=2, padding='same')(conv5)

    print(ARCAM.shape)

    full = keras.layers.GlobalAveragePooling1D()(ARCAM)
    print(full.shape)
    out = keras.layers.Dense(nb_classes, activation='softmax')(full)
    print(out.shape)
    return x, out

def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))


def readucr(filename):
    data = np.loadtxt(filename, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y

nb_epochs =245
flist = ['five_type']
for each in flist:
    fname = each
    x_data, y_data = readucr('UCRArchive_2018' + '/' + fname + '/' + fname + '.csv')
    nb_classes = len(np.unique(y_data))
    print(nb_classes)
    y_data = (y_data - y_data.min()) / (y_data.max() - y_data.min()) * (nb_classes - 1)
    Y_data = keras.utils.to_categorical(y_data, nb_classes)
    x_data_mean = x_data.mean()
    x_data_std = x_data.std()
    x_data = (x_data - x_data_mean) / (x_data_std)
    x_data = x_data.reshape(x_data.shape + (1,))

(trainX, testX, trainY, testY) = train_test_split(x_data, Y_data, test_size=0.2, random_state=42)

batch_size = min(trainX.shape[0] // 10, 16)

inputs = keras.Input(shape=trainX.shape[1:])
x, y = build_WsHVR(trainX.shape[1:], 32, nb_classes)
model = keras.models.Model(inputs=x, outputs=y)
optimizer = keras.optimizers.Adam()

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()])

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5,
                                              patience=50, min_lr=0.0001)
hist = model.fit(trainX, trainY, batch_size=batch_size, epochs=nb_epochs,
                 verbose=1, validation_split=0.1, callbacks=[reduce_lr])


sess = K.get_session()
graph = sess.graph
stats_graph(graph)

start_time = time.time()
scores = model.evaluate(testX, testY, verbose=0)
secs = time.time() - start_time
print(secs)
print(model.metrics_names)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
print("%s: %.2f%%" % (model.metrics_names[2], scores[2] * 100))
print("%s: %.2f%%" % (model.metrics_names[3], scores[3] * 100))
accuracy = scores[1]
precision = scores[2]
recall = scores[3]


acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range(len(acc))