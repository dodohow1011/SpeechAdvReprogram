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
from ts_dataloader import load_data
from ts_model import  AttRNN_Model, ARTLayer, WARTmodel


parser = argparse.ArgumentParser()
parser.add_argument("--weight", type = str, default = "repr-tl/0.01-0.005-trials4-100.h5", help = "weight in weight/")
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
train_csv = '/home/dodohow1011/Voice2Series/Datasets/ZH-SpeechCommands/train_full.csv'
test_csv = '/home/dodohow1011/Voice2Series/Datasets/ZH-SpeechCommands/test_full.csv'
x_train, y_train = load_data(train_csv)
x_test, y_test = load_data(test_csv)
tmp_xt = x_test
classes = np.unique(np.concatenate((y_train, y_test), axis=0))
num_classes = len(np.unique(y_train))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
target_shape = x_test[0].shape

art_model = WARTmodel(target_shape, model, args.baseline, 36, args.mapping, 15)
art_model.load_weights("weight/"+ args.weight)

art_model.summary()
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
    fig, (ax3, ax4, ax5,ax6,ax7) = plt.subplots(5, 1, figsize=(12, 20))
    x_tmp = tmp_xt[idAudio].reshape((tmp_xt[idAudio].shape[0], 1))  
    x_tmp = tf.expand_dims(x_tmp, axis=0)
    print(x_tmp.shape)
    aug_tmp = SegZeroPadding1D(x_tmp, 1, tmp_xt[idAudio].shape[0])
    print(aug_tmp.shape)
    """
    ax1.set_title('Raw waveform', fontsize=18)
    ax1.set_ylabel('Amplitude', fontsize=18)
    ax1.set_xlabel('Sample index', fontsize=18)
    ax1.plot(tf.squeeze(tf.squeeze(aug_tmp, axis=0), axis=-1), 'k-')
    ax1.legend(fancybox=True, framealpha=1,  borderpad=1, fontsize=16)
    
    ax2.set_title('Reprogramming noise', fontsize=18)
    ax2.set_ylabel('Amplitude', fontsize=18)
    ax2.set_xlabel('Sample index', fontsize=18)
    ax2.set_ylim(-1,1)
    noise = audios[idAudio]-tf.squeeze(tf.squeeze(aug_tmp, axis=0), axis=-1)
    ax2.plot(noise, 'r-')
    ax2.legend(fancybox=True, framealpha=1,  borderpad=1, fontsize=16)
    """
    # ax3.set_title('Raw waveform', fontsize=18)
    ax3.set_ylabel('Amplitude', fontsize=24)
    ax3.set_xlabel('Sample index', fontsize=24)
    ax3.plot(audios[idAudio], 'r-', label='Reprogrammed waveform')
    ax3.plot(tf.squeeze(tf.squeeze(aug_tmp, axis=0), axis=-1), 'k-', label='Target waveform')
    ax3.legend(fancybox=True, framealpha=1,  borderpad=1, fontsize=20)

    ax4.set_title('Attention weights (log)', fontsize=18)
    ax4.set_ylabel('Log of attention weight', fontsize=18)
    ax4.set_xlabel('Mel-spectrogram index', fontsize=18)
    ax4.plot(np.log(attW[idAudio]), 'g-')

    # img3 = ax3.imshow(librosa.power_to_db(specs[idAudio,:,:,0], ref=np.max))
    img5 = ax5.pcolormesh(specs[idAudio,:,:,0])
    divider = make_axes_locatable(ax5)
    cax = divider.append_axes("right", size="1%", pad=0.2)
    fig.colorbar(img5,cax=cax)
    # ax5.set_title('Spectrogram visualization', fontsize=18)
    ax5.set_ylabel('Frequency', fontsize=28)
    ax5.set_xlabel('Time', fontsize=28)

    img6 = ax6.imshow(i_cam1, aspect="auto",cmap="jet")
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes("right", size="1%", pad=0.2)
    fig.colorbar(img6,cax=cax)
    # ax6.set_title('Class Activation Mapping Conv2d', fontsize=18)
    ax6.set_xticks([])
    ax6.set_yticks([])
    
    img7 = ax7.imshow(i_cam2, aspect="auto",cmap='jet')
    divider = make_axes_locatable(ax7)
    cax = divider.append_axes("right", size="1%", pad=0.2)
    fig.colorbar(img7,cax=cax)
    # ax7.set_title('Class Activation Mapping Conv2d_1', fontsize=18)
    ax7.set_xticks([])
    ax7.set_yticks([])

    plt.tight_layout()
    plt.savefig("results/"+ str(y_test[idAudio]) + str(idAudio) + ".pdf")
    

def SegZeroPadding1D(orig_x, seg_num, orig_xlen):
    
    src_xlen = 16000
    all_seg = src_xlen//orig_xlen
    assert seg_num <= all_seg
    seg_len = np.int(np.floor(all_seg//seg_num))
    aug_x = tf.zeros([src_xlen,1])
    for s in range(seg_num):
        startidx = (s*seg_len)*orig_xlen
        endidx = (s*seg_len)*orig_xlen + orig_xlen
        # print('seg idx: {} --> start: {}, end: {}'.format(s, startidx, endidx))
        seg_x = ZeroPadding1D(padding=(startidx, src_xlen-endidx))(orig_x)
        aug_x += seg_x

    return aug_x

# for i in range(len(x_test)):
#     visual_sp(repros, i, "adv")
visual_sp(repros, 106, "adv")
