import sys
sys.path.append("..")
import argparse

import numpy as np
from ts_dataloader import load_data, DataGenerator
from ts_model import  AttRNN_Model, ARTLayer, WARTmodel

rnd = np.random.RandomState(1109)

parser = argparse.ArgumentParser()
parser.add_argument("--weight", type = str, default = "repr/3/0.01-0.003-trials1-49-2.0944-0.7529.h5", help = "weight in weight/")
parser.add_argument("--mapping", type= int, default= 2, help = "number of multi-mapping")
parser.add_argument("--baseline", type=bool, default=False, help="Train baseline system")
args = parser.parse_args()

# load data and classes
train_csv = '/home/dodohow1011/Voice2Series/Datasets/LT-SpeechCommands/train_full.csv'
test_csv = '/home/dodohow1011/Voice2Series/Datasets/LT-SpeechCommands/test_full.csv'
noise_csv = '/home/dodohow1011/Voice2Series/Datasets/LT-SpeechCommands/noise_full.csv'

x_train, y_train = load_data(train_csv, rnd)
x_test, y_test = load_data(test_csv, rnd)

bg_audio = load_data(noise_csv, rnd)[0]
bg_audio = [noise * rnd.random() * 0.1 for noise in bg_audio]
rnd.shuffle(bg_audio)
x_test = np.concatenate((x_test, bg_audio[:10]), axis=0)
y_test.extend(["silence"]*10)

classes = np.unique(y_train)
classes = np.append(classes, ["silence"])
cls2label = {label: i for i, label in enumerate(classes.tolist())}
num_classes = len(classes)

x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
target_shape = x_test[0].shape

# load model
pr_model = AttRNN_Model(args.baseline)
wart_model = WARTmodel(target_shape, pr_model, args.baseline, 36, args.mapping, num_classes, 1109)
wart_model.summary()
wart_model.load_weights("weight/" + args.weight)

y_gt = [cls2label[y] for y in y_test]
y_pred = wart_model.predict(x_test, batch_size=None, verbose=0, steps=None)
y_pred = np.argmax(y_pred, axis=1)

print('- Test accuracy:', np.sum(y_pred == y_gt) / len(y_gt))
