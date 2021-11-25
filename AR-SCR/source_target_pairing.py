import os
import sys
sys.path.append(os.getcwd())

import SpeechDownloader
import SpeechGenerator
from similarity_mapping import source_target_mapping
from ts_dataloader import load_data


if __name__ == "__main__":
    gscInfo, n_Categs = SpeechDownloader.PrepareGoogleSpeechCmd(version=2, task='35word')
    SourceGen = SpeechGenerator.SpeechGen(gscInfo['train']['files'], gscInfo['train']['labels'], shuffle=False)
    target_csv = 'Datasets/AR-SCR/train_full.csv'
    target_audios, target_labels = load_data(target_csv)
    source_target_mapping(SourceGen, target_audios, target_labels)


