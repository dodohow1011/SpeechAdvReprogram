import numpy as np
import tensorflow.keras.backend as K

from ts_model import AttRNN_Model

GSm_w2nMapping = {
    'unknown': 0,
    'nine': 1,
    'yes': 2,
    'no': 3,
    'up': 4,
    'down': 5,
    'left': 6,
    'right': 7,
    'on': 8,
    'off': 9,
    'stop': 10,
    'go': 11,
    'zero': 12,
    'one': 13,
    'two': 14,
    'three': 15,
    'four': 16,
    'five': 17,
    'six': 18,
    'seven': 19,
    'eight': 20,
    'backward': 21,
    'bed': 22,
    'bird': 23,
    'cat': 24,
    'dog': 25,
    'follow': 26,
    'forward': 27,
    'happy': 28,
    'house': 29,
    'learn': 30,
    'marvin': 31,
    'sheila': 32,
    'tree': 33,
    'visual': 34,
    'wow': 35
}


Gsm_n2wMapping= {
    0: 'unknown',
    1: 'nine',
    2: 'yes',
    3: 'no',
    4: 'up',
    5: 'down',
    6: 'left',
    7: 'right',
    8: 'on',
    9: 'off',
    10: 'stop',
    11: 'go',
    12: 'zero',
    13: 'one',
    14: 'two',
    15: 'three',
    16: 'four',
    17: 'five',
    18: 'six',
    19: 'seven',
    20: 'eight',
    21: 'backward',
    22: 'bed',
    23: 'bird',
    24: 'cat',
    25: 'dog',
    26: 'follow',
    27: 'forward',
    28: 'happy',
    29: 'house',
    30: 'learn',
    31: 'marvin',
    32: 'sheila',
    33: 'tree',
    34: 'visual',
    35: 'wow'
}


def source_target_mapping(sourceGen, target_audios, target_labels):

    target_classes = np.unique(target_labels, axis=0)
    target_cls2label = {label: i for i, label in enumerate(target_classes.tolist())}
    target_label2cls = {i: label for label, i in target_cls2label.items()}
    target_labels = [target_cls2label[cls] for cls in target_labels]


    # We wants the last hidden output for clustering
    Gsm_model = AttRNN_Model()
    Gsm_model.summary()
    Gsm_model = K.function([Gsm_model.input], [Gsm_model.layers[-2].output])

    source_centers = [[] for i in range(36)]
    target_centers = [[] for i in range(len(target_classes))]

    for audio, label in zip(target_audios, target_labels):
        audio = audio.reshape((1,-1))
        out = Gsm_model([audio])
        center = out[0]
        target_centers[label].append(center)

    for i in range(len(target_centers)):
        target_centers[i] = np.concatenate(target_centers[i], axis=0).mean(axis=0)
    
    for i in range(len(sourceGen)):
        audios, labels = sourceGen.__getitem__(i)
        batch_size, _ = audios.shape
        batch_center = Gsm_model([audios])[0]
        for j in range(batch_size):
            center = batch_center[j].reshape(1, -1)
            label = labels[j]
            source_centers[label].append(center)
       
    for i in range(len(source_centers)):
        source_centers[i] = np.concatenate(source_centers[i], axis=0).mean(axis=0)


    # Now we have a representative vector for each target and source class.
    # Next we need to pair then. 


    counter = [0]*len(target_centers)
    pair_result = {i: [] for i in range(len(target_centers))}


    for s in range(len(source_centers)):
        sims = {}
        for t in range(len(target_centers)):
            sim = np.dot(source_centers[s], target_centers[t]) / (np.linalg.norm(source_centers[s])*np.linalg.norm(target_centers[t]))
            sims[t] = sim
        sorted_sims = {k: v for k, v in sorted(sims.items(), key=lambda x:x[1], reverse=True)}

        is_paired = False
        for k, v in sorted_sims.items():
            if counter[k] < 2:
                pair_result[k].append(s)
                counter[k]+=1
                is_paired = True
                break

        if is_paired:
            continue
        # Every body have 2. Now let them be three
        for k, v in sorted_sims.items():
            if counter[k] < 3:
                pair_result[k].append(s)
                counter[k]+=1
                break
    
    pairing = []
    for t, ss in pair_result.items():
        print("source class: {} is paired with target class: {}".format( [Gsm_n2wMapping[s] for s in ss], target_label2cls[t]))
        pairing.append(ss)


    print("The label_map result is {}".format(pairing))
   
    #np.save('source_centers.npy', source_centers)
    #np.save('target_centers.npy', target_centers)
