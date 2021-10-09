import os
import random
import pandas as pd

data_dir = "/work/dodohow1011/dysarthria_zh_command/"
train_full = list()
test_full = list()

for i in [1,2,3]:
    spk_dir = data_dir + "D_SPK{}".format(i)
    for (dirpath, dirnames, filenames) in os.walk(spk_dir):
        s = random.sample(range(10), 3)
        label = dirpath.split('/')[-1]
        for i,f in enumerate(filenames):
            if not f.endswith(".wav"):
                continue
            
            if i in s:
                test_full += [[os.path.join(dirpath, f), label]]
            else:
                train_full += [[os.path.join(dirpath, f), label]]


train_full_csv = pd.DataFrame(train_full, columns=["filename", "label"])
train_full_csv.to_csv("train_full.csv", sep='\t', index=False)
test_full_csv = pd.DataFrame(test_full, columns=["filename", "label"])
test_full_csv.to_csv("test_full.csv", sep='\t', index=False)
