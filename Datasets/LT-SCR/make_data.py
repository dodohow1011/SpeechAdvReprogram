import os
import re
import random
import hashlib
import pandas as pd

limit = 20

data_dir = "/work/dodohow1011/lt_speech_commands/dataset"
train_full, dev_full, test_full = list(), list(), list()
bg_audio_list = list()

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1
words = 'ne,ačiū,stop,įjunk,išjunk,į_viršų,į_apačią,į_dešinę,į_kairę,startas,pauzė,labas,iki'
bg = '_background_noise_'

def which_set(fname, dev_percentage=10., test_percentage=10.):
    """
    See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/speech_commands/input_data.py#L70
    """
    base_name = fname
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    hash_name_hashed = hashlib.sha1(hash_name.encode('UTF-8')).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) * (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < dev_percentage:
        return "DEV"
    if percentage_hash < dev_percentage + test_percentage:
        return "TEST"
    return "TRAIN"

if __name__ == '__main__':
    train_label_cnt = dict()
    for command in os.listdir(data_dir):
        if command == bg:
            bg_dir = os.path.join(data_dir, bg)
            for wav in os.listdir(bg_dir):
                bg_audio_list.append([os.path.join(bg_dir, wav), "silence"])
            continue

        command_dir = os.path.join(data_dir, command)
        for wav in os.listdir(command_dir):
            t = which_set(wav, 10., 10.)
            if t == "TEST":
                if command not in words:
                    label = "unknown"
                else:
                    label = command
                test_full += [[os.path.join(command_dir, wav), label]]
            elif t == "DEV":
                if command not in words:
                    label = "unknown"
                else:
                    label = command
                dev_full += [[os.path.join(command_dir, wav), label]]
            else:
                if command not in train_label_cnt:
                    train_label_cnt[command] = 0
                train_label_cnt[command] += 1
                if train_label_cnt[command] > limit:
                    continue
                else:
                    if command not in words:
                        label = "unknown"
                    else:
                        label = command
                    train_full += [[os.path.join(command_dir, wav), label]]

    
    train_full_csv = pd.DataFrame(train_full, columns=["filename", "label"])
    train_full_csv.to_csv("train_limit{}.csv".format(limit), sep='\t', index=False)

    dev_full_csv = pd.DataFrame(dev_full, columns=["filename", "label"])
    dev_full_csv.to_csv("dev_full.csv", sep='\t', index=False)
    test_full_csv = pd.DataFrame(test_full, columns=["filename", "label"])
    test_full_csv.to_csv("test_full.csv", sep='\t', index=False)

    bg_full_csv = pd.DataFrame(bg_audio_list, columns=["filename", "label"])
    bg_full_csv.to_csv("noise_full.csv", sep='\t', index=False)
