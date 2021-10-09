import os
import random
import pandas as pd

data_dir = "/work/dodohow1011/AR_SpeechCommands/"
train_full = list()
dev_full = list()
test_full = list()

command_dir = os.path.join(data_dir, "commands")
number_dir = os.path.join(data_dir, "numbers")

for wav in os.listdir(command_dir):
    coin = random.random()
    command = wav[0]
    if coin <= 0.2:
        test_full += [[os.path.join(command_dir, wav), command]]
    else:
        c = random.random()
        if c <= 0.1:
            dev_full += [[os.path.join(command_dir, wav), command]]
        else:
            train_full += [[os.path.join(command_dir, wav), command]]

for wav in os.listdir(number_dir):
    coin = random.random()
    number = wav[0]
    if coin <= 0.2:
        test_full += [[os.path.join(number_dir, wav), number]]
    else:
        c = random.random()
        if c <= 0.1:
            dev_full += [[os.path.join(number_dir, wav), number]]
        else:
            train_full += [[os.path.join(number_dir, wav), number]]

train_full_csv = pd.DataFrame(train_full, columns=["filename", "label"])
train_full_csv.to_csv("train_full.csv", sep='\t', index=False)
dev_full_csv = pd.DataFrame(dev_full, columns=["filename", "label"])
dev_full_csv.to_csv("dev_full.csv", sep='\t', index=False)
test_full_csv = pd.DataFrame(test_full, columns=["filename", "label"])
test_full_csv.to_csv("test_full.csv", sep='\t', index=False)
