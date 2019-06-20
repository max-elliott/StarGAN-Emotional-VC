import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import audio_utils
import yaml
from classifiers import *
from average_weighted_attention import Average_Weighted_Attention


class StarGAN_emo_VC1(object):
    '''
    The proposed model of this project.
    '''
    def __init__(self, config):
        '''
        Need config for input_size, hidden_size, num_layers, num_classes, bi = False
        '''
        super(StarGAN_emo_VC1, self).__init__()
        self.config = config
        # Need completing
        print("Test")
        self.build_model(self.config)


    def set_train_mode(self):
        self.G.train()
        self.D.train()
        self.emo_cls.train()
        self.speaker_cls.train()
        # self.dimension_cls.train()

    def set_eval_mode(self):
        self.G.eval()
        self.D.eval()
        self.emo_cls.eval()
        self.speaker_cls.eval()
        # self.dimension_cls.eval()

    def to_device(self, device = torch.device('cuda')):
        if torch.cuda.is_available():
            self.G.to(device = device)
            self.D.to(device = device)
            self.emo_cls.to(device = device)
            self.speaker_cls.to(device = device)
            # self.dimension_cls.to(device = device)
        else:
            print("Device not available")

    def build_model(self, config):

        self.num_input_feats = config['model']['main_model']['num_feats']
        self.hidden_size = 128
        self.num_layers = 2
        self.num_emotions = 4
        self.num_speakers = 10
        self.bi = True

        print("Building components")

        self.G = Generator()
        self.D = Discriminator()
        self.emo_cls = Emotion_Classifier(self.num_input_feats, self.hidden_size,
                                          self.num_layers, self.num_emotions,
                                          bi = self.bi)
        self.speaker_cls = Emotion_Classifier(self.num_input_feats, self.hidden_size,
                                          self.num_layers, self.num_speakers,
                                          bi = self.bi)
        # self.dimension_cls = Dimension_Classifier(self.num_input_feats, self.hidden_size,
        #                                   self.num_layers, bi = self.bi)

        print("Building optimizers")

        con_opt = config['model']['optimizer']
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), con_opt['g_lr'], [con_opt['beta1'], con_opt['beta2']])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), con_opt['d_lr'], [con_opt['beta1'], con_opt['beta2']])
        self.emo_cls_optimizer = torch.optim.Adam(self.emo_cls.parameters(), con_opt['emo_cls_lr'],[con_opt['beta1'], con_opt['beta2']])
        self.speaker_cls_optimizer = torch.optim.Adam(self.speaker_cls.parameters(), con_opt['speaker_cls_lr'],[con_opt['beta1'], con_opt['beta2']])
        # self.dim_cls_optimizer = torch.optim.Adam(self.dim_cls.parameters(), config.dim_cls_lr,[config.beta1, config.beta2])

        print("Network parameter list:")

        self.print_network(self.G, 'G')
        self.print_network(self.D, 'D')
        self.print_network(self.emo_cls, 'Emotion Classifier')
        self.print_network(self.speaker_cls, 'Speaker Classifier')
        # self.print_network(self.dim_cls, 'Dimensional Emotions Classifier')

        self.to_device()

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.emo_cls_optimizer.zero_grad()
        self.speaker_cls_optimizer.zero_grad()
        # self.dim_cls_optimizer.zero_grad()


    # def foward(self, x, c_new, c_original):
    #     '''
    #     Can't actually do this properly in a class, move to solver.
    #     Needs to:
    #         - update D more often than G (GAN theory)
    #         - split x_g_mels into segments for classification (unless it can be
    #           done with variable sequence lengths)
    #     '''
    #     # pass through generator to get x' = x_g
    #     x_g = self.G(x,c_new)
    #
    #     # Convert x to mel, NEED A BATCH VERSION
    #     x_g_mels = audio_utils.spectrogram2melspectrogram(x_g)
    #
    #     # for reconstruction to make x_rec
    #     x_rec = self.G(x_g_mels, c_original)
    #
    #     # get D_real, D_emo, D_spk, D_dim
    #     y_real = self.D(x_g_mels, c_new)
    #     y_emo = self.emo_cls(x_g_mels)
    #     y_spk = self.speaker_cls(x_g_mels)
    #     y_dim = self.dimension_cls(x_g_mels)
    #
    #     return x_rec, y_real, y_emo, y_spk, y_dim

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        #Create layers
        self.temp_layer = nn.Linear(10,10)

    def forward(self, x, c):

        x = self.temp_layer(x)

        return x

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        #Create layers
        self.temp_layer = nn.Linear(10,10)

    def forward(self, x, c):

        x = self.temp_layer(x)

        return x

if __name__ == '__main__':

    config = yaml.load(open('./config.yaml', 'r'))

    print("Made config.")

    model = StarGAN_emo_VC1(config)
