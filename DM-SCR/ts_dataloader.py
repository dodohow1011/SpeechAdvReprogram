import librosa
import numpy as np
import pandas as pd
import soundfile as sound

from tensorflow import keras


def load_data(data_csv):
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
