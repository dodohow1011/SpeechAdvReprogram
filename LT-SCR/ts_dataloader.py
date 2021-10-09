import librosa
import numpy as np
import pandas as pd
import soundfile as sound

from tensorflow import keras

sr = 16000
def load_data(data_csv, rnd):
    data_df = pd.read_csv(data_csv, sep='\t')   
    wavpath = data_df['filename'].tolist()
    labels = data_df['label'].to_list()

    x, y = list(), list()
    for wav, label in zip(wavpath, labels):
        stereo, fs = sound.read(wav)
        stereo = stereo / np.abs(stereo).max()
        if fs != sr:
            stereo = librosa.resample(stereo, fs, sr)
        if stereo.shape[0] > sr:
            start = rnd.choice(len(stereo) - sr + 1)
            x.append(stereo[start:start+sr])
        else:
            x.append(np.pad(stereo, (0, sr-stereo.shape[0])))
        
        y.append(label)
    
    return np.array(x), y


class DataGenerator(keras.utils.Sequence):
    def __init__(self, datas, labels, bg_audio, classes, rnd, batch_size=32, shuffle=True):
        self.unknowns = list()
        self.commands = list()

        self._split_unknown(datas, labels)

        self.datas = list()

        self.bg_audio = bg_audio

        self.classes = classes
        self.add_noise = "silence" in self.classes
        
        self.rnd = rnd

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch = 0
        self.on_epoch_end()

    def _split_unknown(self, datas, labels):
        for wav, label in zip(datas, labels):
            if label == "unknown":
                self.unknowns.append((wav, label))
            else:
                self.commands.append((wav, label))

    def __len__(self):
        return int(np.floor(len(self.datas) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        batch = list()
        for k in indexes:
            batch.append(self.datas[k])

        X, y = self.__data_generation(batch)

        return X, y

    def on_epoch_end(self):
        self.datas = list()
        """
        self.datas.extend(self.commands)
        
        for _ in range(int(len(self.commands)*0.1)):
            unk = np.random.choice(len(self.unknowns))
            unknown = self.unknowns[unk]
            self.datas.append(unknown)
            
            if self.add_noise:
                sil = np.random.choice(len(self.bg_audio))
                silence = self.bg_audio[sil]
                self.datas.append((silence, "silence"))
        """
        c = 0
        u = 0
        for _ in range(self.batch_size*7):
            coin = self.rnd.random()
            if coin < 0.1:
                unk = self.rnd.choice(len(self.unknowns))
                unknown = self.unknowns[unk]
                self.datas.append(unknown)
                u += 1
            elif coin < 0.15:
                sil = self.rnd.choice(len(self.bg_audio))
                silence = self.bg_audio[sil]
                self.datas.append((silence, "silence"))
            else:
                com = self.rnd.choice(len(self.commands))
                command = self.commands[com]
                self.datas.append(command)
                c += 1
    
        print ("{} {} ".format(c,u))
        
        self.indexes = np.arange(len(self.datas))
        if self.shuffle == True:
            self.rnd.shuffle(self.indexes)

    def __data_generation(self, batch):
        X, y = list(), list()
        for wav, label in batch:
            X.append(wav)
            y.append(self.classes[label])

        X = np.array(X)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        return X, keras.utils.to_categorical(y, num_classes=len(self.classes))
