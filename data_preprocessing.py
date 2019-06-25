from IPython.display import Audio
from scipy.io import wavfile
import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import audio_utils
import pickle

# ./google-cloud-sdk/bin/gcloud compute ssh --project mscproject1 --zone europe-west4-a pytorch-1-1-vm -- -L 8080:localhost:8080
# zipped wav data at https://storage.googleapis.com/emo-vc-storage/IEMOCAP.zip

dataset_dir = "/Users/Max/MScProject/datasets/IEMOCAP"
n_speakers = 10
n_emotions = 7
# dataset_dir = "/Users/Max/MScProject/datasets/test_dir"

# def speaker2onehot(filename):
#
#     onehot = np.zeros(n_speakers)
#     code = filename[4] + filename[-8]
#
#     conversion = {'1F':0, '1M':1, '2F':2, '2M':3, '3F':4, '3M':5, '4F':6, '4M':7,
#                     '5F': 8, '5M':9}
#
#     onehot[conversion[code]] = 1
#
#     return onehot
#
# def category2onehot(category):
#
#     onehot = np.zeros(7, dtype = np.float32)
#
#     if category == 'xxx' or category =='dis' or category =='fea' or category == 'oth':
#         return onehot
#
#     conversion = {'hap':0, 'sad':1, 'ang':2, 'exc':3, 'fru':4, 'sur':5, 'neu':6}
#
#     onehot[conversion[category]] = 1
#
#     return onehot

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

    list = np.zeros(3)
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

def concatenate_labels(emo_labels, speakers, dims, dims_dis):

    all_labels = np.zeros((len(emo_labels), 8))
    # print(all_labels)

    for i, row in enumerate(all_labels):
        row[0] = emo_labels[i]
        row[1] = speakers[i]
        row[2:5] = dims[i]
        row[5:8] = dims_dis[i]

    return all_labels

def get_wav_and_labels(filename, session_dir):

    folder = filename[:-9]
    wav_path = session_dir + "/" + folder + "/" + filename
    label_path = session_dir + "/Annotations/" + folder + ".txt"

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

    return audio, category, dimensions, dimensions_dis, speaker

def get_session_data(data_dir, exclude_unlabelled = True):

    # print(data_dir)
    os.chdir(data_dir)

    filenames = []
    specs = []
    mels = []
    labels = []
    conts = []
    conts_dis = []
    speakers = []
    for foldername in os.listdir(data_dir):

        if not (foldername == "Annotations" or foldername == ".DS_Store"):

            for filename in os.listdir(data_dir + "/" + foldername):

                if not filename == ".DS_Store":
                    wav, label, cont, cont_dis, speaker = get_wav_and_labels(filename,
                                                                    data_dir)

                    if not (exclude_unlabelled and label == -1): #ignore some rare emotions
                        mel = audio_utils.wav2melspectrogram(wav)
                        spec = audio_utils.wav2spectrogram(wav)

                        filenames.append(filename)
                        mels.append(torch.Tensor(mel))
                        specs.append(torch.Tensor(spec))

                        labels.append(label)
                        conts.append(cont)
                        conts_dis.append(cont_dis)
                        speakers.append(speaker)

        print(foldername + " completed.")

    return filenames, mels, specs, labels, conts, conts_dis,  speakers

def all_wavs_and_labels(exclude_unlabelled = True):

    os.chdir(dataset_dir)

    filenames = []
    specs = []
    mels = []
    labels = []
    conts = []
    conts_dis = []
    speakers = []

    for foldername in os.listdir(dataset_dir):

        if not (foldername == "Annotations" or foldername == ".DS_Store" or foldername == "Processed"):

            for filename in os.listdir(dataset_dir + "/" + foldername):

                if not filename == ".DS_Store":
                    wav, label, cont, cont_dis, speaker = audio_utils.get_wav_and_labels(filename)

                    if not (exclude_unlabelled and label == -1): #ignore some rare emotions

                        spec = audio_utils.wav2spectrogram(wav)
                        mel = audio_utils.spectrogram2melspectrogram(spec)

                        filenames.append(filename)
                        mels.append(torch.Tensor(mel))
                        specs.append(torch.Tensor(spec))

                        labels.append(label)
                        conts.append(cont)
                        conts_dis.append(cont_dis)
                        speakers.append(speaker)
        print(foldername + " done.")

    return filenames, mels, specs, labels, conts, conts_dis, speakers

def save_data(data, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)

def load_data(filename):
    with open(filename, 'rb') as fp:

        data = pickle.load(fp)

    return data

def load_session_data(session_num, directory= dataset_dir + "/Processed_data"):
    '''
    str: session_num
    str: directory
    returns:
        lists for filenames, labels, melspecs of that session
        melspecs are of the form (n_mels, seq_len)
    '''
    session_num = str(session_num)
    names = load_data(directory + "/filenames" + session_num)
    labels = load_data(directory + "/labels" + session_num)
    speakers = load_data(directory + "/speakers" + session_num)
    dims = load_data(directory + "/conts" + session_num)
    dims_dis = load_data(directory + "/conts_dis" + session_num)
    # specs = pp.load_data(directory + "/specs" + session_num)
    melspecs = load_data(directory + "/melspecs" + session_num)

    return names, melspecs, labels, speakers, dims, dims_dis

if __name__ == "__main__":

    # for i in range(1,6):
    #
    #     ses_number = str(i)
    #
    #     session_dir = dataset_dir + "/Session" + ses_number
    #
    #     filenames, mels, specs, labels, conts, conts_dis, speakers = get_session_data(
    #                                         session_dir, exclude_unlabelled = True)
    #
    #     # print(len(mels))
    #     save_data(filenames, dataset_dir + '/Processed_data/filenames' + ses_number)
    #     save_data(mels, dataset_dir + '/Processed_data/melspecs' + ses_number)
    #     save_data(specs, dataset_dir + '/Processed_data/specs' + ses_number)
    #     save_data(labels, dataset_dir + '/Processed_data/labels' + ses_number)
    #     save_data(conts, dataset_dir + '/Processed_data/conts' + ses_number)
    #     save_data(conts_dis, dataset_dir + '/Processed_data/conts_dis' + ses_number)
    #     save_data(speakers, dataset_dir + '/Processed_data/speakers' + ses_number)
    #     print('Done ' + ses_number + ".")

    wav = get_wav_and_labels("Ses01F_impro03_F001.wav", dataset_dir + "/Session1")[0]
    # wav2 = get_wav_and_labels("Ses01F_impro03_F002.wav", dataset_dir + "/Session1")[0]

    spec = audio_utils.wav2spectrogram(wav)
    # spec2 = audio_utils.wav2spectrogram(wav2)
    print(spec.shape)
    # print(spec2.shape)
    audio_utils.plot_spec(spec, type = 'log')
    #
    melspec = audio_utils.spectrogram2melspectrogram(spec)
    print(melspec.shape)
    audio_utils.plot_spec(melspec)
    #
    # reproduced = audio_utils.spectrogram2wav(spec)
    #
    # audio_utils.save_wav(reproduced, "/Users/Max/MScProject/datasets/Produced/test1.wav")
