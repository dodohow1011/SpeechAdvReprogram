import os
import sys
sys.path.append(os.getcwd())

import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

from funcs import EarlyStopping, LR_Warmup
from ts_dataloader import load_data, DataGenerator
from ts_model import AttRNN_Model, ARTLayer, WARTmodel, make_model


# Learning phase is set to 0 since we want the network to use the pretrained moving mean/var
# K.clear_session()

def main(args, rnd):

    train_csv = 'Datasets/LT-SCR/train_limit20.csv'
    dev_csv = 'Datasets/LT-SCR/dev_full.csv'
    test_csv = 'Datasets/LT-SCR/test_full.csv'

    noise_csv = 'Datasets/LT-SCR/noise_full.csv'

    x_train, y_train = load_data(train_csv, rnd)
    x_dev, y_dev = load_data(dev_csv, rnd)
    x_test, y_test = load_data(test_csv, rnd)

    classes = np.unique(y_train)

    bg_audio = load_data(noise_csv, rnd)[0]
    bg_audio = [noise * rnd.random() * 0.1 for noise in bg_audio]

    rnd.shuffle(bg_audio)
    x_test = np.concatenate((x_test, bg_audio[:10]), axis=0)
    x_dev = np.concatenate((x_dev, bg_audio[10:20]), axis=0)
    y_test.extend(["silence"]*10)
    y_dev.extend(["silence"]*10)

    classes = np.append(classes, ["silence"])

    cls2label = {label: i for i, label in enumerate(classes.tolist())}
    num_classes = len(classes)
    
    train_generator = DataGenerator(x_train, y_train, bg_audio[20:], cls2label, rnd)
    y_dev = [cls2label[y] for y in y_dev]
    y_test = [cls2label[y] for y in y_test]
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
    adam = tf.keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer = adam, metrics=['accuracy'])

    lr_scheduler = LR_Warmup(lr_base=args.lr,decay=args.lr_decay,warmup_epochs=20)
    
    save_path = "LT-SCR/weight/" + str(args.lr) + "-" + str(args.lr_decay) + "-{epoch:02d}-{val_loss:.4f}-{val_accuracy:.4f}.h5"
    if not os.path.exists('LT-SCR/weight'):
        os.makedirs('LT-SCR/weight')

            
    checkpoints = tf.keras.callbacks.ModelCheckpoint(save_path, monitor='val_accuracy', save_weights_only=True, save_best_only=True)
    earlystop = EarlyStopping(monitor='val_accuracy', patience=10, start_epoch=70, restore_best_weights=True)
    exp_callback = [earlystop, lr_scheduler, checkpoints]


    model.summary()

    batch_size = 32
    epochs = args.eps
    exp_history = model.fit(train_generator, batch_size=batch_size, epochs=epochs, verbose=1,
                            validation_data=(x_dev, y_dev),callbacks= exp_callback)


    score = model.evaluate(x_test, y_test, verbose=0)
    print('--- Test loss:', score[0])
    print('- Test accuracy:', score[1])

    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=int, default=2, help="Number of multi-mapping")
    parser.add_argument("--eps", type=int, default=200, help="Epochs") 
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.005, help="Learnig rate decay rate")
    args = parser.parse_args()
    
    rnd = np.random.RandomState(seed=1109)
    main(args, rnd)

