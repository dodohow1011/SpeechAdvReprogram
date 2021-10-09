import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from SpeechModels import AttRNNSpeechModel
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import models
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable

def multi_mapping(prob, source_num, mapping_num, target_num):
    
    mt = mapping_num * target_num ##mt must smaller than source_num
    label_map = np.zeros([source_num, mt]) ##[source_num, map_num*target_num]
    label_map[0:mt, 0:mt] = np.eye(mt) ##[source_num, map_num*target_num]
    map_prob = tf.matmul(prob, tf.constant(label_map, dtype=tf.float32)) ## [1, source_num] * [source_num, map_num*target_num] = [1, map_num*target_num]
    final_prob = tf.reduce_mean(tf.reshape(map_prob, shape=[tf.shape(map_prob)[0], target_num, mapping_num]), axis=-1) ##[target_num]
    # weight = np.zeros([source_num, target_num])
    # cluster_labels = [[4,7], [20,24], [16,26,32], [6,13,31], [1,3,30], [9,19], [0,8], [17,18], [10,25], [5,11], [2,23], [21,27,35], [15,22,33], [28,29,34], [12,14]]
    # cluster_labels = [[20,22,28], [4,8], [10,13], [14,23,27], [0,21], [12,18,29], [2,15,33], [6,7,31], [11,25,26], [1,5,30], [17,19,35], [9,16,32], [3,24,34]]
    # for i, ls in enumerate(cluster_labels):
    #     for num in ls:
    #         weight[num][i] = 1
    # final_prob = tf.matmul(prob, tf.constant(weight,dtype=tf.float32))
    return final_prob 

def layer_output(in_feats, model, ly_name = "batch_normalization_6 ", n = 7):
    conv_layer = model.get_layer(ly_name)
    heatmap_model = models.Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as gtape:
        conv_output, predictions = heatmap_model(in_feats[n:n+1])
        loss = predictions[:, np.argmax(predictions[0])]
        grads = gtape.gradient(loss, conv_output)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    return heatmap, conv_output

def vis_map(heatmap):
    heatmap = np.maximum(heatmap, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat
    return heatmap

def to_rgb(heatmap, h_x, w_x):
    heatmap = np.uint8(255 * vis_map((heatmap[0])))
    # We use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # We use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[np.flipud(np.transpose(heatmap))]

    # We create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)

    jet_heatmap = jet_heatmap.resize((  w_x, h_x))

    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Save the superimposed image
    superimposed_img = keras.preprocessing.image.array_to_img(jet_heatmap)

    return superimposed_img

def ts_CAM(model, x_test, y_test):
    get_last_conv = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-2].output])
    last_conv = get_last_conv([x_test[:100], 1])[0]
    get_softmax = keras.backend.function([model.layers[0].input, keras.backend.learning_phase()], [model.layers[-1].output])
    softmax = get_softmax(([x_test[:100], 1]))[0]
    softmax_weight = model.get_weights()[-2]
    CAM = np.dot(last_conv, softmax_weight)
    k = 0
    # for k in range(5):
    CAM = (CAM - CAM.min(axis=1, keepdims=True)) / (CAM.max(axis=1, keepdims=True) - CAM.min(axis=1, keepdims=True))
    c = np.exp(CAM) / np.sum(np.exp(CAM), axis=1, keepdims=True)
    plt.figure(figsize=(13, 7))
    plt.plot(x_test[k].squeeze())
    plt.scatter(np.arange(len(x_test[k])), x_test[k].squeeze(), cmap='hot_r', c=c[k, :, :, int(y_test[k])].squeeze(), s=100)
    plt.title('True label:' + str(y_test[k]) + '   likelihood of label ' + str(y_test[k]) + ': ' + str(softmax[k][int(y_test[k])]))
    plt.colorbar()
    plt.savefig("cam.pdf")


def plot_acc_loss(x_history, eps, map_num):

    plt.figure()
    plt.style.use("seaborn")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))

    ax1.plot(x_history.history["val_accuracy"], label="Val. acc")
    ax1.plot(x_history.history["accuracy"], label="Training acc")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([0, 1])
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(x_history.history["val_loss"], label="Val. loss")
    ax2.plot(x_history.history["loss"], label="Training loss")
    ax2.set_ylabel("Loss")
    #ax2.set_ylim(top=5.5)
    ax2.set_xlabel("Epoch")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("results/AR-SpeechCommands" + "_eps" + eps + "_map" + map_num + "_.png") #PadCenter/


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im,cax=cax)
    
    ax.set_title(title, fontsize='large')
    
    tick_marks = np.arange(len(classes))    
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)

    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()
