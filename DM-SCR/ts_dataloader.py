import librosa
import numpy as np
import pandas as pd
import soundfile as sound

from tensorflow import keras


def load_data(data_csv, is_noise=False):
    data_df = pd.read_csv(data_csv, sep='\t')   
    wavpath = data_df['filename'].tolist()
    labels = data_df['label'].to_list()

    x, y = list(), list()
    omits = [10,13,14,15,16,19]
    for wav, label in zip(wavpath, labels):
        if label in omits:
            continue
        stereo, sr = sound.read(wav)
        stereo, index = librosa.effects.trim(stereo, top_db=20)
        if sr != 16000:
            stereo = librosa.resample(stereo, sr, 16000)
        if stereo.shape[0] > 16000:
            start = (stereo.shape[0] - 16000) // 2
            x.append(stereo[start:start+16000])
        else:
            x.append(np.pad(stereo, (0, 16000-stereo.shape[0])))
        
        y.append(label)
    
    return np.array(x), np.array(y)


class DataGenerator(keras.utils.Sequence):
    def __init__(self, datas, labels, bg_audio, classes, batch_size=32, shuffle=True):
        self.unknowns = list()
        self.commands = list()

        self._split_unknown(datas, labels)

        self.datas = list()

        self.bg_audio = bg_audio

        self.classes = classes
        self.add_noise = "silence" in self.classes
        print ("Adding noise data")
        
        self.batch_size = batch_size
        self.shuffle = shuffle
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
        for _ in range(self.batch_size*7):
            coin = random.random()
            if coin < 0.1:
                unk = self.rnd.choice(len(self.unknowns))
                unknown = self.unknowns[unk]
                self.datas.append(unknown)
            else:
                com = self.rnd.choice(len(self.commands))
                command = self.commands[com]
                self.datas.append(command)
        self.indexes = np.arange(len(self.datas))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch):
        X, y = list(), list()
        for wav, label in batch:
            X.append(wav)
            y.append(self.classes[label])

        X = np.array(X)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        return X, keras.utils.to_categorical(y, num_classes=len(self.classes))
