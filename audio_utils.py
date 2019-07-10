from IPython.display import Audio
from scipy.io import wavfile
import os
import yaml
import copy


import librosa
import librosa.display

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

        self.normalise_mels = True
        self.max_norm_value = 3226.99139880277
        self.min_norm_value = 3.8234146815389095e-10


hp = hyperparams()

def load_wav(path):
    wav = wavfile.read(path)[1]
    wav = copy.deepcopy(wav)/32767.0
    return wav

def save_wav(wav, path):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
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
    mel = (hp.max_norm_value - hp. min_norm_value) * mel + hp.min_norm_value
    return mel

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
        spectrogram = spectrogram.numpy()

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
        spec = spec.numpy()

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
        spec = spec.numpy()
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
        spec = spec.numpy()
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

if __name__ == '__main__':

    files = librosa.util.find_files("/Users/Max/MScProject/datasets/test_dir")
    # print(files[0])
    filepath = files[0]
    wav = load_wav(filepath)
    mel = wav2melspectrogram(wav)
    save_spec_plot(mel, "None", "Test2.png")
