# CHH Yang et al. 2021 (http://proceedings.mlr.press/v139/yang21j/yang21j.pdf)
# Apache Apache-2.0 License
import sys

import numpy as np

import tensorflow as tf
import kapre
from tensorflow.keras.models import Model, load_model
from kapre.time_frequency import Melspectrogram, Spectrogram
from tensorflow.keras.layers import ZeroPadding2D, Input, Layer, ZeroPadding1D, Reshape, Permute, Dense, Dropout
from tensorflow.keras import initializers,regularizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K
from kapre.utils import Normalization2D
from SpeechModels import AttRNNSpeechModel
from utils import multi_mapping
from tensorflow import keras

print("tensorflow vr. ", tf.__version__, "kapre vr. ",kapre.__version__)

def AttRNN_Model():

    nCategs=36
    sr=16000
    #iLen=16000

    model = AttRNNSpeechModel(nCategs, samplingrate = sr, inputLength = None)
    model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy'])

    model.load_weights('weight/pr_attRNN.h5')

    return model


# Adverserial Reprogramming layer
class ARTLayer(Layer):
    def __init__(self, W_regularizer=0.05, **kwargs):
        self.init = initializers.GlorotUniform()
        self.W_regularizer = regularizers.l2(W_regularizer)
        super(ARTLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='kernel', 
                                      shape=(16000,1),
                                      initializer=self.init,regularizer = self.W_regularizer,
                                      trainable=True)

        super(ARTLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, dropout=0.4, training=True):
        prog = Dropout(dropout)(self.W, training=training) # remove K.tanh
        out = x + prog
        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1], input_shape[2])


# White Adversairal Reprogramming Time Series (WART) Model 
def WARTmodel(input_shape, pr_model, source_classes, mapping_num, target_classes, dropout=0.5):
    x = Input(shape=input_shape)
    out = ARTLayer()(x,dropout)
    out = Reshape([16000,])(out)
    probs = pr_model(out) 
    
    map_probs = multi_mapping(probs, source_classes, mapping_num, target_classes)
    model = Model(inputs=x, outputs= map_probs)

    return model


def make_model(input_shape, num_classes):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)
