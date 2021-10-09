# Original CAM Code is modified from Yang et al. ICASSP 2021 (https://arxiv.org/pdf/2010.13309.pdf)
# Please consider to cite both de Andrade et al. 2018 and Yang et al. 2021, if you use the attention heads and CAM visualization.
import sys
sys.path.append("..")

import argparse
import time as ti

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import ZeroPadding2D, Input, Layer, ZeroPadding1D

import librosa
import pandas as pd
import soundfile as sound

from PIL import Image
import librosa.display
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils import layer_output, to_rgb
from ts_model import  AttRNN_Model, ARTLayer, WARTmodel

from ts_dataloader import load_data

rnd = np.random.RandomState(824)


parser = argparse.ArgumentParser()
parser.add_argument("--weight", type = str, default = "repr-tl/0.01-trials1-61-0.0102-0.9918.h5", help = "weights in weight/")
parser.add_argument("--mapping", type= int, default= 2, help = "number of multi-mapping")
parser.add_argument("--layer", type = str, default = "conv2d_1", help = "the layer for cam")
parser.add_argument("--baseline", type=bool, default=False, help="Train baseline system")
args = parser.parse_args()

base_model = AttRNN_Model(args.baseline)
base_model.summary()
model = base_model

attM = Model(inputs=model.input, outputs=[model.get_layer('attSoftmax').output,
                                          model.get_layer('mel_stft').output])


# load data and classes
train_csv = '/home/dodohow1011/Voice2Series/Datasets/AR-SpeechCommands/train_full.csv'
test_csv = '/home/dodohow1011/Voice2Series/Datasets/AR-SpeechCommands/test_full.csv'
x_train, y_train = load_data(train_csv)
x_test, y_test = load_data(test_csv)
tmp_xt = x_test
classes = np.unique(np.concatenate((y_train, y_test), axis=0))
num_classes = len(np.unique(y_train))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
target_shape = x_test[0].shape

art_model = WARTmodel(target_shape, model, args.baseline, 36, args.mapping, 16)
art_model.load_weights("AR-SpeechCommands/weight/".format(lang) + args.weight)


ReproM = Model(inputs=art_model.input, outputs=[art_model.get_layer('reshape_1').output])

repros = ReproM.predict(x_test)


def visual_sp(audios, idAudio, use='base', clayer = args.layer):

    attW, specs = attM.predict(audios)

    w_x, h_x = specs[idAudio,:,:,0].shape
    i_heatmap1, _ = layer_output(audios, base_model, 'conv2d', idAudio)
    i_heatmap2, _ = layer_output(audios, base_model, 'conv2d_1', idAudio)
    i_cam1 = to_rgb(i_heatmap1, w_x, h_x)
    i_cam2 = to_rgb(i_heatmap2, w_x, h_x)


    plt.figure()
    plt.style.use("seaborn-whitegrid")
    fig, (ax1, ax2,ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 20))

    ax1.set_title('Raw waveform', fontsize=18)
    ax1.set_ylabel('Amplitude', fontsize=18)
    ax1.set_xlabel('Sample index', fontsize=18)
    ax1.plot(audios[idAudio], 'b-',label = "Reprogrammed time series")
    if use != 'base':
        x_tmp = tmp_xt[idAudio].reshape((tmp_xt[idAudio].shape[0], 1))  
        x_tmp = tf.expand_dims(x_tmp, axis=0)
        print(x_tmp.shape)
        aug_tmp = SegZeroPadding1D(x_tmp, 1, tmp_xt[idAudio].shape[0])
        ax1.plot(tf.squeeze(tf.squeeze(aug_tmp, axis=0), axis=-1), 'k-', label="Target time series")
        print(aug_tmp.shape)
    ax1.legend(fancybox=True, framealpha=1,  borderpad=1, fontsize=16)

    ax2.set_title('Attention weights (log)', fontsize=18)
    ax2.set_ylabel('Log of attention weight', fontsize=18)
    ax2.set_xlabel('Mel-spectrogram index', fontsize=18)
    ax2.plot(np.log(attW[idAudio]), 'r-')

    # img3 = ax3.imshow(librosa.power_to_db(specs[idAudio,:,:,0], ref=np.max))
    img3 = ax3.pcolormesh(specs[idAudio,:,:,0])
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="1%", pad=0.2)
    fig.colorbar(img3,cax=cax)
    ax3.set_title('Spectrogram visualization', fontsize=18)
    ax3.set_ylabel('Frequency', fontsize=18)
    ax3.set_xlabel('Time', fontsize=18)

    img4 = ax4.imshow(i_cam1, aspect="auto",cmap="jet")
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="1%", pad=0.2)
    fig.colorbar(img4,cax=cax)
    ax4.set_title('Class Activation Mapping Conv2d', fontsize=18)
    ax4.set_xticks([])
    ax4.set_yticks([])
    
    img5 = ax5.imshow(i_cam2, aspect="auto",cmap='jet')
    divider = make_axes_locatable(ax5)
    cax = divider.append_axes("right", size="1%", pad=0.2)
    fig.colorbar(img5,cax=cax)
    ax5.set_title('Class Activation Mapping Conv2d_1', fontsize=18)
    ax5.set_xticks([])
    ax5.set_yticks([])

    plt.tight_layout()
    plt.savefig("results/" + y_test[idAudio] + ".png")
    

for i in [0, 6, 15, 18, 27, 33, 36, 45, 48, 63, 69, 72, 78, 81]:
    visual_sp(repros, i, "adv")

