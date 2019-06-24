from IPython.display import Audio
from scipy.io import wavfile
import os
import librosa
import librosa.display
import numpy as np
import torch
import torch.nn as nn
import copy
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
        self.n_mels = 128 # Number of Mel banks to generate
        self.power = 1.2 # Exponent for amplifying the predicted magnitude
        self.n_iter = 100 # Number of inversion iterations
        self.use_log_magnitude = True # if False, use magnitude
        self.preemph = 0.97

hp = hyperparams()

def load_wav(path):
    return wavfile.read(path)[1]

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

def lowpass(spec, frequency):
    return spec[:frequency, :]

def wav2melspectrogram(y, sr = hp.sr, n_mels = hp.n_mels):
    '''
    y = input wav file
    '''
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels = n_mels,
        n_fft = hp.n_fft, hop_length = hp.hop_length, power = hp.power)

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
        spec = spec .numpy()
    librosa.display.specshow(spec, y_axis=type, sr=hp.sr, hop_length=hp.hop_length,
                                                    fmin=None, fmax=4000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Power spectrogram')
    plt.show()

if __name__ == '__main__':
    
    files = librosa.util.find_files("/Users/Max/MScProject/datasets/test_dir")
    print(files)
