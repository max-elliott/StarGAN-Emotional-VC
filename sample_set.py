import os
import yaml
import numpy as np

import audio_utils
import data_preprocessing2 as pp
import preprocess_world as pw

import torch
import librosa


class Sample_Set():

    def __init__(self, config):

        self.config = config

        self.filenames = librosa.util.find_files(config['data']['sample_set_dir'])

        self.set = {}

        for file in self.filenames:

            filename = os.path.basename(file)
            if filename[-1] == '/':
                filename = filename[0:-1]

            audio, labels = pp.get_samples_and_labels(filename, config)

            if config['data']['type'] == 'mel':

                spec = torch.Tensor(audio_utils.wav2spectrogram(audio)).t()
                melspec = torch.Tensor(audio_utils.wav2melspectrogram(audio)).t()

                self.set[filename] = [melspec, labels, spec]
            else:

                audio = np.array(audio, dtype = np.float64)

                f0, ap, sp, coded_sp = pw.cal_mcep(audio)
                coded_sp = torch.Tensor(coded_sp.T)
                self.set[filename] = [f0, ap, sp, coded_sp, labels]



    def get_set(self):
        '''
        Return dict of all samples
        Each value in dict is (mel, labels, spec) = ((len,n_mels),(8),(len2, n_ffts/2+1))
        '''
        return self.set

if __name__ == '__main__':

    config = yaml.load(open('./config.yaml', 'r'))

    s = Sample_Set(config)

    for tag, val in s.get_set().items():
        print(tag, val[0].size())
