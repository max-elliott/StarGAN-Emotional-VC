from IPython.display import Audio
from scipy.io import wavfile
import os
import yaml
import copy
import pickle

import librosa
import librosa.display

import pyworld
from pyworld import decode_spectral_envelope, synthesize

import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

# dataset_dir = "/Users/Max/MScProject/datasets/IEMOCAP"
# dataset_dir = "/Users/Max/MScProject/test_dir"

class hyperparams(object):
    def __init__(self):
        self.sr = 16000 # Sampling rate. Paper => 24000
        self.n_fft = 1024 # fft points (samples)
        self.frame_shift = 0.0125 # seconds
        self.frame_length = 0.05 # seconds
        self.hop_length = int(self.sr*self.frame_shift) # samples  This is dependent on the frame_shift.
        self.win_length = int(self.sr*self.frame_length) # samples This is dependent on the frame_length.
        self.n_mels = 80 # Number of Mel banks to generate
        self.power = 1.2 # Exponent for amplifying the predicted magnitude
        self.n_iter = 100 # Number of inversion iterations
        self.use_log_magnitude = True # if False, use magnitude
        self.preemph = 0.97

        self.config = yaml.load(open('./config.yaml', 'r'))
        self.sample_set_dir = self.config['logs']['sample_dir']

        self.normalise_mels = self.config['data']['normalise_mels']
        self.max_norm_value = 3226.99139880277
        self.min_norm_value = 3.8234146815389095e-10

        self.sp_max_norm_value = 6.482182376067761
        self.sp_min_norm_value = -18.50642857581744

        with open('./f0_dict.pkl', 'rb') as fp:
            self.f0_dict = pickle.load(fp)

        with open('./f0_relative_dict.pkl', 'rb') as fp:
            self.f0_relative_dict = pickle.load(fp)

        # for tag, val in self.f0_dict.items():
        #     print(f'Emotion {tag} stats:')
        #     for tag2, val2 in val.items():
        #         print(f'{tag2} = {val2[0]}, {val2[1]}')

hp = hyperparams()

def load_wav(path):
    wav = wavfile.read(path)[1]
    wav = copy.deepcopy(wav)/32767.0
    return wav

def save_wav(wav, path):
    # print(np.max(np.abs(wav)))
    # print(np.mean(np.abs(wav)))
    # wav *= 32767 / max(0.01, np.max(np.abs(wav)))

    wav *= 48000
    wav = np.clip(wav, -32767, 32767)
    wavfile.write(path, hp.sr, wav.astype(np.int16))

def wav2spectrogram(y, sr = hp.sr):

    '''
    Produces log-magnitude spectrogram of audio data y
    '''

    spec = librosa.core.stft(y, n_fft = hp.n_fft, hop_length = hp.hop_length,
                                                win_length = hp.win_length)
    spec_mag = amp_to_db(np.abs(spec))
    # spec_angle = np.angle(spec)
    # spec_mag = lowpass(spec_mag, 400)

    return spec_mag

def _normalise_mel(mel):
    mel = (mel - hp.min_norm_value)/(hp.max_norm_value - hp.min_norm_value)
    return mel

def _unnormalise_mel(mel):
    mel = (hp.max_norm_value - hp.min_norm_value) * mel + hp.min_norm_value
    return mel

def _normalise_coded_sp(sp):
    sp = (sp - hp.sp_min_norm_value)/(hp.sp_max_norm_value - hp.sp_min_norm_value)

    return sp

def _unnormalise_coded_sp(sp):
    sp = (hp.sp_max_norm_value - hp.sp_min_norm_value) * sp + hp.sp_min_norm_value
    np.clip(sp, hp.sp_min_norm_value, hp.sp_max_norm_value)
    return sp

def wav2melspectrogram(y, sr = hp.sr, n_mels = hp.n_mels):
    '''
    y = input wav file
    '''

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels = n_mels,
        n_fft = hp.n_fft, hop_length = hp.hop_length)
    # mel_spec = librosa.core.amplitude_to_db(y)
    if hp.normalise_mels:
        mel_spec = _normalise_mel(mel_spec)

    return mel_spec

def spectrogram2melspectrogram(spec, n_fft = hp.n_fft, n_mels = hp.n_mels):

    if isinstance(spec, torch.Tensor):
        spec = spec.numpy()

    mels = librosa.filters.mel(hp.sr, n_fft, n_mels = n_mels)
    return mels.dot(spec**hp.power)

def melspectrogram2wav(mel):
    '''
    Needs doing: impossible?
    '''
    return 0

def spectrogram2wav(spectrogram):
    '''
    Griffin-Lim Algorithm
    spectrogram: [t, f], i.e. [t, nfft // 2 + 1]
    '''
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.cpu().numpy()

    spectrogram = db_to_amp(spectrogram)#**hp.power

    X_best = copy.deepcopy(spectrogram)  # [f, t]
    for i in range(hp.n_iter):

        X_t = invert_spectrogram(X_best)
        # print(X_t.shape())
        est = librosa.stft(X_t, hp.n_fft, hp.hop_length, win_length=hp.win_length)  # [f, t]
        phase = est / np.maximum(1e-8, np.abs(est))  # [f, t]
        X_best = spectrogram * phase  # [f, t]
    X_t = invert_spectrogram(X_best)

    return np.real(X_t)

def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hp.hop_length, win_length=hp.win_length, window="hann")

def wav2mcep(y, sr = hp.sr, n_mfcc = 30):
    '''
    y = input wav file
    '''
    mfccs = librosa.feature.mfcc(y=ym, sr=sr, n_mfcc=n_mfcc)

    return mfccs

def mcep2wav(mfccs):
    '''
    Needs doing
    '''
    return 0

def amp_to_db(spec):
    return librosa.core.amplitude_to_db(spec)

def db_to_amp(spec):
    return librosa.core.db_to_amplitude(spec)

def plot_spec(spec, type = 'mel'):

    plt.figure(figsize=(6, 4))

    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()

    if hp.normalise_mels:
        spec = _unnormalise_mel(spec)

    librosa.display.specshow(librosa.power_to_db(spec), y_axis=type, sr=hp.sr,
                                hop_length=hp.hop_length)
                                                    # fmin=None, fmax=4000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Power spectrogram')
    plt.show()

def save_spec(spec, model_name, filename, type = 'mel'):
    '''
    spec: [n_feats, seq_len] - np.array or torch.Tensor
    model_name: str - just the basename, no directory
    filename: str
    '''
    fig = plt.figure(figsize=(6,4))

    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    if hp.normalise_mels:
        spec = _unnormalise_mel(spec)

    path = os.path.join(hp.sample_set_dir, model_name)

    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join(path, filename)

    np.save(path, spec)

    # print("Saved.")

def save_spec_plot(spec, model_name, filename, type = 'mel'):
    '''
    spec: [n_feats, seq_len] - np.array or torch.Tensor
    model_name: str - just the basename, no directory
    filename: str
    '''
    fig = plt.figure(figsize=(6,4))

    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    if hp.normalise_mels:
        spec = _unnormalise_mel(spec)

    librosa.display.specshow(librosa.power_to_db(spec), y_axis=type, sr=hp.sr,
                            hop_length=hp.hop_length)
                                                    # fmin=None, fmax=4000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Power spectrogram')

    path = os.path.join(hp.sample_set_dir, model_name)

    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join(path, filename)

    plt.savefig(path)
    plt.close(fig)
    plt.close("all")
    # print("Saved.")

def save_world_wav(feats, model_name, filename):

    # feats = [f0, sp, ap, sp_coded, labels]

    if isinstance(feats[3], torch.Tensor):
        feats[3] = feats[3].cpu().numpy()
    if hp.normalise_mels:
        feats[3] = _unnormalise_coded_sp(feats[3])

    path = os.path.join(hp.sample_set_dir, model_name)

    if not os.path.exists(path):
        os.makedirs(path)

    path = os.path.join(path, filename)

    # print("Made path.")
    feats[3] = np.ascontiguousarray(feats[3], dtype=np.float64)
    # print("Made contiguous.")
    # print(feats[3].shape)
    decoded_sp = decode_spectral_envelope(feats[3], hp.sr, fft_size = hp.n_fft)
    # print("Decoded.")
    # f0_converted = norm.pitch_conversion(f0, speaker, target)
    wav = synthesize(feats[0], decoded_sp, feats[1], hp.sr)
    # Audio(wav,rate=hp.sr)
    # librosa.display.waveplot(y=wav, sr=hp.sr)
    # print("Sythesized wav.")
    save_wav(wav, path)
    # print("Saved wav.")

def f0_pitch_conversion(f0, source_labels, target_labels):
    '''
    Logarithm Gaussian normalization for Pitch Conversions
    (np.array) f0 - array to be converted
    (tuple) source_labels - (emo, speaker) discrete labels
    (tuple) target_labels - (emo, speaker) discrete labels
    '''
    src_emo = int(source_labels[0])
    src_spk = int(source_labels[1])
    trg_emo = int(target_labels[0])
    trg_spk = int(target_labels[1])

    mean_log_src = hp.f0_dict[src_emo][src_spk][0]
    std_log_src = hp.f0_dict[src_emo][src_spk][1]

    mean_log_target = hp.f0_dict[trg_emo][src_spk][0]
    std_log_target = hp.f0_dict[trg_emo][src_spk][1]


    # ----- Absolute transformation -----
    f0_converted = np.exp((np.ma.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)

    # ----- Relative transformation -----
    # Has power(std,2) as accidently used std not variance when making statistics
    # f0_converted = np.exp(np.ma.log(f0)*(np.power(hp.f0_relative_dict[src_emo][trg_emo][1], 2)+1) + hp.f0_relative_dict[src_emo][trg_emo][0])
    # logf0 = np.ma.log(f0)
    # mean = np.mean(logf0)
    # var = np.var(logf0)
    # f0_converted = np.exp((logf0-mean)/var*(hp.f0_relative_dict[src_emo][trg_emo][1]+var) + mean + hp.f0_relative_dict[src_emo][trg_emo][0])

    return f0_converted

if __name__ == '__main__':

    # file = './samples/f0s/world2_crop_4d_200_200_testSet/Ses01F_impro02_F014_1to0.wav'
    # file = '../data/audio/Ses01F_impro02_F017.wav' 3
    # file = '../data/audio/Ses01F_impro04_F028.wav' 4
    # file = '../data/audio/Ses01F_impro05_F022.wav' #5
    file = '../data/audio/Ses01F_script01_1_F037.wav' #6
    file = '../data/audio/Ses01F_script01_1_F042.wav' #7


    wav = load_wav(file)
    spec = wav2melspectrogram(wav)

    spec = spec[:, 16:-16]
    print("Original size =", spec.shape)

    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2, 2, 1)

    plt.subplot(2, 2, 1)
    librosa.display.specshow(librosa.power_to_db(spec), y_axis='mel', sr=hp.sr,
                            hop_length=hp.hop_length)
                                                    # fmin=None, fmax=4000)
    # plt.colorbar(format='%+2.0f dB')
    plt.title('Power Spectrogram Original')


    file = './samples/f0s/world3_4d_newloader_cont_080_testSet/Ses01F_script01_1_F042_0to0.wav'

    wav = load_wav(file)
    spec = wav2melspectrogram(wav)

    print("New size =", spec.shape)

    ax2 = fig.add_subplot(2, 2, 2)

    # plt.subplot(2, 2, 2)
    librosa.display.specshow(librosa.power_to_db(spec), y_axis='mel', sr=hp.sr,
                            hop_length=hp.hop_length)
                                                    # fmin=None, fmax=4000)
    # plt.colorbar(format='%+2.0f dB')
    plt.title('Power Spectrogram Angry')

    file = './samples/f0s/world3_4d_newloader_cont_080_testSet/Ses01F_script01_1_F042_0to1.wav'

    wav = load_wav(file)
    spec = wav2melspectrogram(wav)

    # if hp.normalise_mels:
    #     spec = _unnormalise_mel(spec)

    ax3 = fig.add_subplot(2, 2, 3, sharey=ax1)
    librosa.display.specshow(librosa.power_to_db(spec), y_axis='mel', sr=hp.sr,
                            hop_length=hp.hop_length)
                                                    # fmin=None, fmax=4000)
    # plt.colorbar(format='%+2.0f dB')
    plt.title('Power Spectrogram Sad')

    file = './samples/f0s/world3_4d_newloader_cont_080_testSet/Ses01F_script01_1_F042_0to2.wav'

    wav = load_wav(file)
    spec = wav2melspectrogram(wav)

    # if hp.normalise_mels:
    #     spec = _unnormalise_mel(spec)

    ax4 = fig.add_subplot(2, 2, 4, sharey=ax1)
    librosa.display.specshow(librosa.power_to_db(spec), y_axis='mel', sr=hp.sr,
                            hop_length=hp.hop_length)
                                                    # fmin=None, fmax=4000)
    # plt.colorbar(format='%+2.0f dB')
    plt.title('Power Spectrogram Happy')
    #
    plt.savefig('../graphs/specs/3-emo_a-s-h_example7.png')
    plt.close(fig)
    plt.close("all")

    # name = os.path.basename(file)
    # save_spec_plot(mel, "world2_4d_mels", name + "linear.png")


    # emo2emo_dict = {}
    #
    # for e1 in range(0,4):
    #
    #     emo2emo_dict[e1] = {}
    #
    #     for e2 in range(0,4):
    #
    #         mean_list = []
    #         std_list = []
    #
    #         for s in range(0,10):
    #             mean_diff = hp.f0_dict[e2][s][0] - hp.f0_dict[e1][s][0]
    #             std_diff = hp.f0_dict[e2][s][1] - hp.f0_dict[e1][s][1]
    #             mean_list.append(mean_diff)
    #             std_list.append(std_diff)
    #
    #         mean_mean = np.mean(mean_list)
    #         std_mean = np.mean(std_list)
    #         emo2emo_dict[e1][e2] = (mean_mean, std_mean)
    #
    # for tag, val in emo2emo_dict.items():
    #     print(f'Emotion {tag} stats:')
    #     for tag2, val2 in val.items():
    #         print(f'{tag2} = {val2[0]}, {val2[1]}')
    #
    # with open('f0_relative_dict2.pkl', 'wb') as f:
    #     pickle.dump(emo2emo_dict, f, pickle.HIGHEST_PROTOCOL)


    # for tag, val in hp.f0_dict.items():
    #     print(f'Emotion {tag} stats:')
    #     for tag2, val2 in val.items():
    #         print(f'{tag2} = {val2[0]}, {val2[1]}')
    #
    # for tag, val in hp.f0_relative_dict.items():
    #     print(f'Emotion {tag} stats:')
    #     for tag2, val2 in val.items():
    #         print(f'{tag2} = {val2[0]}, {val2[1]}')

    # files = librosa.util.find_files("/Users/Max/MScProject/StarGAN-Emotional-VC/samples/DA/all/3-emo-augmented", ext = "wav")
    # # files = [os.path.basename(f) for f in files]
    # print(files)
    # numbers = []
    # for f in files:
    #     # numbers.append(f[-5])
    #     if f[-8] != '3':
    #         # if f[-5] != '0':
    #
    #         name = os.path.basename(f)[:-10] + os.path.basename(f)[-8:]
    #         numbers.append(name)
    #         os.rename(f, "/Users/Max/MScProject/StarGAN-Emotional-VC/samples/DA/all/3-emo-augmented/" + name)
    #         # os.rename(f, "/Users/Max/MScProject/StarGAN-Emotional-VC/samples/DA/all/actual/" + name)
    #
    # print(numbers)
