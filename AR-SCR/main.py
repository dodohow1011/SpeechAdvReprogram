import os
import sys
sys.path.append(os.getcwd())

import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

from funcs import EarlyStopping
from ts_dataloader import load_data
from ts_model import AttRNN_Model, ARTLayer, WARTmodel, make_model

# Learning phase is set to 0 since we want the network to use the pretrained moving mean/var
# K.clear_session()

def main(args):

    train_csv = 'Datasets/AR-SCR/train_full.csv'
    dev_csv = 'Datasets/AR-SCR/dev_full.csv'
    test_csv = 'Datasets/AR-SCR/test_full.csv'


    x_train, y_train = load_data(train_csv)
    x_dev, y_dev = load_data(dev_csv)
    x_test, y_test = load_data(test_csv)

    classes = np.unique(y_train)
    cls2label = {label: i for i, label in enumerate(classes.tolist())}
    num_classes = len(classes)
    
    y_train = [cls2label[y] for y in y_train]
    y_dev = [cls2label[y] for y in y_dev]
    y_test = [cls2label[y] for y in y_test]
    y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_dev = keras.utils.to_categorical(y_dev, num_classes=num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)


    print("--- X shape : ", x_train[0].shape, "--- Num of Classes : ", num_classes) ## target class


    ## Pre-trained Model for Adv Program  
    pr_model = AttRNN_Model()


    ## # of Source classes in Pre-trained Model
    source_classes = 36 ## Google Speech Commands

    target_shape = (x_train[0].shape[0], 1)

    ## Adv Program Time Series (ART)
    mapping_num = args.mapping
    try:
        assert mapping_num*num_classes <= source_classes
    except AssertionError:
        print("Error: The mapping num should be smaller than source_classes / num_classes: {}".format(source_classes//num_classes)) 
        exit(1)

    model = WARTmodel(target_shape, pr_model, source_classes, mapping_num, num_classes, args.dropout)

    ## Loss
    adam = tf.keras.optimizers.Adam(lr=args.lr)
    model.compile(loss='categorical_crossentropy', optimizer = adam, metrics=['accuracy'])
        
    ## Checkpoints
    save_path = "AR-SCR/weight/" + str(args.lr) + "-{epoch:02d}-{val_accuracy:.4f}.h5"
    if not os.path.exists('AR-SCR/weight'):
        os.makedirs('AR-SCR/weight')

    checkpoints = tf.keras.callbacks.ModelCheckpoint(save_path,save_weights_only=True)
    earlystop = EarlyStopping(monitor='val_loss', patience=10, start_epoch=50, restore_best_weights=True)
    exp_callback = [earlystop, checkpoints]


    model.summary()

    batch_size = 32
    epochs = args.eps

    exp_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                            validation_data=(x_dev, y_dev), callbacks= exp_callback)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('--- Test loss:', score[0])
    print('- Test accuracy:', score[1])

    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=int, default=1, help="Number of multi-mapping")
    parser.add_argument("--eps", type=int, default=50, help="Epochs") 
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
    args = parser.parse_args()
    
    main(args)
