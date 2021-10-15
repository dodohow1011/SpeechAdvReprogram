## Adversarial Reprogramming on Speech Command Recognition

<img src="https://github.com/dodohow1011/SpeechAdvReprogram/blob/main/illustration.png" width="500">

### Environment

Tensorflow 2.2 (CUDA=10.0) and Kapre 0.2.0. 

- option 1 (from yml)

```shell
conda env create -f repr-scr.yml
source activate repr-scr
```

- option 2 (from clean python 3.6)

```shell
pip install tensorflow-gpu==2.1.0
pip install kapre==0.2.0
pip install h5py==2.10.0
```

### Dataset

Arabic Speech Commands dataset

- Please download the Arabic Speech Commands dataset [here](https://github.com/ltkbenamer/AR_Speech_Database.git).

```shell
./prepare_ar_data.sh
```

Lithuanian Speech Commands dataset

- Please download the Lithuanian Speech Commands dataset [here](https://github.com/kolesov93/lt_speech_commands).

```shell
./prepare_lt_data.sh
```

Dysarthric Speech Commands dataset

- Please download the Lithuanian Speech Commands dataset [here](https://reurl.cc/a5vAG4).

```shell
./prepare_dm_data.sh
```

### Training

For training and evaluating the three speech command recognition results.

```shell
./run_ar.sh
./run_lt.sh
./run_dm.sh
```

For more details please refer to [AR-SCR](https://github.com/dodohow1011/SpeechAdvReprogram/blob/main/AR-SCR/main.py), [LT-SCR](https://github.com/dodohow1011/SpeechAdvReprogram/blob/main/LT-SCR/main.py) and [DM-SCR](https://github.com/dodohow1011/SpeechAdvReprogram/blob/main/DM-SCR/main.py)


### Reference

- A Study of Low-Resource Speech Commands Recognition Based on Adversarial Reprogramming

- Voice2Series: Reprogramming Acoustic Models for Time Series Classification


```bib

@InProceedings{pmlr-v139-yang21j,
  title = 	 {Voice2Series: Reprogramming Acoustic Models for Time Series Classification},
  author =       {Yang, Chao-Han Huck and Tsai, Yun-Yun and Chen, Pin-Yu},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {11808--11819},
  year = 	 {2021},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
}

```

- Database for Arabic Speech Commands Recognition [Paper](https://www.researchgate.net/publication/346962582_Database_for_Arabic_Speech_Commands_Recognition)

- Voice Activation for Low-Resource Languages [Paper](https://www.mdpi.com/2076-3417/11/14/6298)

- Unsupervised Pre-Training for Voice Activation [Paper](https://www.mdpi.com/2076-3417/10/23/8643)

- A Speech Command Control-Based Recognition System for Dysarthric Patients Based on Deep Learning Technology [Paper](https://www.mdpi.com/2076-3417/11/6/2477)
