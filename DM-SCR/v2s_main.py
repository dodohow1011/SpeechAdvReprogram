import sys
sys.path.append("..")
import argparse

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import class_weight
import tensorflow.keras.backend as K


from funcs import LR_Warmup, EarlyStopping
from ts_dataloader import load_data, DataGenerator
from ts_model import AttRNN_Model, ARTLayer, WARTmodel, make_model
from sklearn.model_selection import KFold
# from vggish.model import Vggish_Model


# Learning phase is set to 0 since we want the network to use the pretrained moving mean/var
# K.clear_session()

def main(args, trial, rnd, seed):

    train_csv = '/home/dodohow1011/Voice2Series/Datasets/ZH-SpeechCommands/train_full.csv'
    dev_csv = '/home/dodohow1011/Voice2Series/Datasets/ZH-SpeechCommands/dev_full.csv'
    test_csv = '/home/dodohow1011/Voice2Series/Datasets/ZH-SpeechCommands/test_full.csv'

    x_train, y_train = load_data(train_csv, rnd)
    x_dev, y_dev = load_data(dev_csv, rnd)
    x_test, y_test = load_data(test_csv, rnd)
    
    classes = np.unique(y_train)
    """
    kf = KFold(random_state=rnd, shuffle=True)
    train_index, dev_index = next(kf.split(x_train))
    
    x_train, x_dev = x_train[train_index], x_train[dev_index]
    y_train, y_dev = y_train[train_index], y_train[dev_index]
    """
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
    if args.net == 0:
        pr_model = AttRNN_Model(args.baseline, args.load_pr)
    else:
        pr_model = Vggish_Model()


    ## # of Source classes in Pre-trained Model
    if args.net == 0: ## choose pre-trained network 
        source_classes = 36 ## Google Speech Commands
    else:
        source_classes = 512 ## VGGish

    target_shape = (x_train[0].shape[0], 1)

    ## Adv Program Time Series (ART)
    mapping_num = args.mapping
    try:
        assert mapping_num*num_classes <= source_classes
    except AssertionError:
        print("Error: The mapping num should be smaller than source_classes / num_classes: {}".format(source_classes//num_classes)) 
        exit(1)

    model = WARTmodel(target_shape, pr_model, args.baseline, source_classes, mapping_num, num_classes, args.dropout)

    ## Loss
    adam = tf.keras.optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer = adam, metrics=['accuracy'])

    lr_scheduler = LR_Warmup(lr_base=args.lr,min_lr=0.001,decay=args.lr_decay,warmup_epochs=30)
    
    if args.baseline:
        if args.load_pr:
            save_path = "weight/transfer/" + str(args.lr) + "-" + str(args.lr_decay) + "-trials" + str(t) + "-{epoch:02d}.h5"
        else:
            save_path = "weight/baseline/" + str(args.lr) + "-" + str(args.lr_decay) + "-trials" + str(t) + "-{epoch:02d}.h5"
    else:
        save_path = "weight/repr/" + str(args.lr) + "-" + str(args.lr_decay) + "-trials" + str(t) + "-{epoch:02d}.h5"

            
    checkpoints = tf.keras.callbacks.ModelCheckpoint(save_path,save_weights_only=True)
    exp_callback = [lr_scheduler, checkpoints]


    model.summary()

    batch_size = 32
    epochs = args.eps
    exp_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks= exp_callback)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('--- Test loss:', score[0])
    print('- Test accuracy:', score[1])
    y_pred = model.predict(x_test, batch_size=None, verbose=0, steps=None)
    y_pred = np.argmax(y_pred, axis=1)
    
    y_gt = np.argmax(y_test, axis=1)
    print('- Test accuracy:', np.sum(y_pred == y_gt) / len(y_gt))

    
    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=int, default=0, help="Pretrained (0), AttRNN (#32), (1) VGGish (#512)")
    parser.add_argument("--mapping", type=int, default=2, help="Number of multi-mapping")
    parser.add_argument("--eps", type=int, default=100, help="Epochs") 
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout")
    parser.add_argument("--lr", type=float, default=0.008, help="Initial learning rate")
    parser.add_argument("--lr_decay", type=float, default=0.005, help="Learnig rate decay rate")
    parser.add_argument("--baseline", type=bool, default=False, help="Train baseline system")
    parser.add_argument("--load_pr", type=bool, default=False, help="Load pretrained model or not")
    args = parser.parse_args()
    
    acc = list()
    trials = 10
    seeds = [824, 1011, 719, 129, 1109, 117, 9487, 666, 1008, 848]
    for t in range(trials):
        seed = seeds[t]
        tf.random.set_seed(seed)
        rnd = np.random.RandomState(seed=seed)
        score = main(args, t, rnd, seed)
        acc.append(score[1])

    avg = sum(acc) / trials
    with open("exp/Accuracy", "a") as f:
        if args.baseline:
            if args.load_pr:
                f.write("transfer ")
            f.write("baseline ")
        f.write("dropout {}, lr {} lr_decay {} Acc ".format(args.dropout, args.lr, args.lr_decay))
        for a in acc:
            f.write("{:.3f}, ".format(a))
        f.write("Avg: {:.3f}".format(avg))
        f.write("\n")

