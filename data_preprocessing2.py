import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.nn.functional as F

import audio_utils

import numpy as np
import librosa
import random
import os
from librosa.util import find_files


def get_speaker_from_filename(filename):
    code = filename[4] + filename[-8]

    conversion = {'1F':0, '1M':1, '2F':2, '2M':3, '3F':4, '3M':5, '4F':6, '4M':7,
                    '5F': 8, '5M':9}

    label = conversion[code]

    return label

def get_emotion_from_label(category):

    if category == 'xxx' or category =='dis' or category =='fea' or category == 'oth':
        return -1
    if category == 'exc' or category == 'fru' or category == 'sur':
        return -1

    conversion = {'hap':0, 'sad':1, 'ang':2, 'neu':3, 'exc':4, 'fru':5, 'sur':6}

    label = conversion[category]

    return label

def getOneHot(label, n_labels):

    onehot = np.zeros(n_labels)
    onehot[label] = 1

    return onehot

def cont2list(cont, binned = False):

    list = [0,0,0]
    list[0] = float(cont[1:6])
    list[1] = float(cont[9:14])
    list[2] = float(cont[17:22])

    #Option to make the values discrete: low(0), med(1) or high(2)
    if binned:
        for i, val in enumerate(list):
            if val <= 2:
                list[i] = 0
            elif val < 4:
                list[i] = 1
            else:
                list[i] = 2
        return list
    else:
        return list

def concatenate_labels(emo, speaker, dims, dims_dis):

    all_labels = torch.zeros(8)
    # print(all_labels)

    # for i, row in enumerate(all_labels):
    all_labels[0] = emo
    all_labels[1] = speaker
    all_labels[2] = dims[0]
    all_labels[3] = dims[1]
    all_labels[4] = dims[2]
    all_labels[5] = dims_dis[0]
    all_labels[6] = dims_dis[1]
    all_labels[7] = dims_dis[2]


    return all_labels

def get_wav_and_labels(filename, data_dir):

    # folder = filename[:-9]
    wav_path = data_dir + "/audio/" + filename
    label_path = data_dir + "/Annotations/" + filename[:-9] + ".txt"

    with open(label_path, 'r') as label_file:

        category = ""
        dimensions = ""
        speaker = ""

        for row in label_file:
            if row[0] == '[':
                split = row.split("\t")
                if split[1] == filename[:-4]:
                    category = get_emotion_from_label(split[2])
                    dimensions = cont2list(split[3])
                    dimensions_dis = cont2list(split[3], binned = True)
                    speaker = get_speaker_from_filename(filename)


    audio = audio_utils.load_wav(wav_path)
    audio = np.array(audio, dtype = np.float32)
    labels = concatenate_labels(category, speaker, dimensions, dimensions_dis)

    return audio, labels

def get_filenames(data_dir):

    files = find_files(data_dir, ext = 'wav')
    filenames = []

    for f in files:
        f = os.path.basename(f)[:-4]
        filenames.append(f)

    return filenames

if __name__ == '__main__':

    data_dir = '/Users/Max/MScProject/data'
    annotations_dir = os.path.join(data_dir, "audio")
    files = find_files(data_dir, ext = 'wav')

    filenames = []
    for f in files:
        f = os.path.basename(f)
        filenames.append(f)
    print(filenames)

    i = 0
    mels_made = 0
    for f in filenames:

        wav, labels = get_wav_and_labels(f, data_dir)
        mel = audio_utils.wav2melspectrogram(wav)
        labels = np.array(labels)
        if labels[0] != -1:
            np.save(data_dir + "/mels/" + f[:-4] + ".npy", mel)
            np.save(data_dir + "/labels/" + f[:-4] + ".npy", labels)
            mels_made += 1

        i += 1
        if i % 100 == 0:
            print(i, " complete.")
            print(mels_made, "mels made.")
