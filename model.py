import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import audio_utils
import os
from classifiers import *
from average_weighted_attention import Average_Weighted_Attention


class StarGAN_emo_VC1(object):
    '''
    The proposed model of this project.
    '''
    def __init__(self, config, name):
        '''
        Need config for input_size, hidden_size, num_layers, num_classes, bi = False
        '''
        super(StarGAN_emo_VC1, self).__init__()
        self.config = config
        self.save_dir = config['logs']['model_save_dir']
        self.name = name
        # Need completing

        self.build_model(self.config)


    def set_train_mode(self):
        self.G.train()
        self.D.train()
        self.emo_cls.train()
        # self.speaker_cls.train()
        # self.dimension_cls.train()

    def set_eval_mode(self):
        self.G.eval()
        self.D.eval()
        self.emo_cls.eval()
        # self.speaker_cls.eval()
        # self.dimension_cls.eval()

    def to_device(self, device = torch.device('cuda')):
        if torch.cuda.is_available():
            self.G.to(device = device)
            self.D.to(device = device)
            self.emo_cls.to(device = device)
            # self.speaker_cls.to(device = device)
            # self.dimension_cls.to(device = device)
        else:
            print("Device not available")

    def build_model(self, config):

        self.num_input_feats = config['model']['num_feats']
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
        # self.speaker_cls = Emotion_Classifier(self.num_input_feats, self.hidden_size,
        #                                   self.num_layers, self.num_speakers,
        #                                   bi = self.bi)
        # self.dimension_cls = Dimension_Classifier(self.num_input_feats, self.hidden_size,
        #                                   self.num_layers, bi = self.bi)

        print("Building optimizers")

        con_opt = config['optimizer']
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), con_opt['g_lr'], [con_opt['beta1'], con_opt['beta2']])
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), con_opt['d_lr'], [con_opt['beta1'], con_opt['beta2']])
        self.emo_cls_optimizer = torch.optim.Adam(self.emo_cls.parameters(), con_opt['emo_cls_lr'],[con_opt['beta1'], con_opt['beta2']])
        # self.speaker_cls_optimizer = torch.optim.Adam(self.speaker_cls.parameters(), con_opt['speaker_cls_lr'],[con_opt['beta1'], con_opt['beta2']])
        # self.dim_cls_optimizer = torch.optim.Adam(self.dim_cls.parameters(), config.dim_cls_lr,[config.beta1, config.beta2])

        if self.config['verbose']:
            print("Network parameter list:")

            G_count = self.print_network(self.G, 'G')
            D_count = self.print_network(self.D, 'D')
            emo_count = self.print_network(self.emo_cls, 'Emotion Classifier')
            # self.print_network(self.speaker_cls, 'Speaker Classifier')
            # self.print_network(self.dim_cls, 'Dimensional Emotions Classifier')

            total = G_count + D_count + emo_count

            print("TOTAL NUMBER OF PARAMETERS = {}".format(total))

        self.to_device()

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))
        return num_params

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()
        self.emo_cls_optimizer.zero_grad()
        # self.speaker_cls_optimizer.zero_grad()
        # self.dim_cls_optimizer.zero_grad()

    def save(self, save_dir = None, iter = 0):

        if save_dir == None:
            save_dir = self.save_dir

        path = os.path.join(save_dir, self.name)
        if not os.path.exists(path):
            os.makedirs(path)

        state = {'D': self.D.state_dict(),
                 'G': self.G.state_dict(),
                 'emo': self.emo_cls.state_dict(),
                 'd_opt': self.d_optimizer.state_dict(),
                 'g_opt': self.g_optimizer.state_dict(),
                 'emo_opt': self.emo_cls_optimizer.state_dict(),
                }

        path = os.path.join(path, "{:05}.ckpt".format(iter))

        torch.save(state, path)
        # torch.save(self.G.state_dict(), G_path)
        # torch.save(self.emo_cls.state_dict(), emo_path)

        print("Saved model as {}.".format(path))

    def load(self, load_dir, iter):

        if load_dir[-1] == '/':
            load_dir = load_dir[0:-1]

        self.name = os.path.basename(load_dir)

        path = os.path.join(load_dir, "{:05}.ckpt".format(iter))
        # G_path = os.path.join(load_dir, "{:05}_G.ckpt".format(iter))
        # emo_path = os.path.join(load_dir, "{:05}_C_emo.ckpt".format(iter))

        dictionary = torch.load(path)
        # G_dict = torch.load(G_path)
        # emo_dict = torch.load(emo_path)

        self.D.load_state_dict(dictionary['D'])
        self.G.load_state_dict(dictionary['G'])
        self.emo_cls.load_state_dict(dictionary['emo'])

        self.d_optimizer.load_state_dict(dictionary['d_opt'])
        self.g_optimizer.load_state_dict(dictionary['g_opt'])
        self.emo_cls_optimizer.load_state_dict(dictionary['emo_opt'])

        print("Model and optimizers loaded.")

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

class Down2d(nn.Module):
    """docstring for Down2d."""
    def __init__(self, in_channel ,out_channel, kernel, stride, padding):
        super(Down2d, self).__init__()

        self.c1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.n1 = nn.InstanceNorm2d(out_channel)
        self.c2 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.n2 = nn.InstanceNorm2d(out_channel)

    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.n1(x1)

        x2 = self.c2(x)
        x2 = self.n2(x2)

        x3 =  x1 * torch.sigmoid(x2)

        return x3


class Up2d(nn.Module):
    """docstring for Up2d."""
    def __init__(self, in_channel ,out_channel, kernel, stride, padding):
        super(Up2d, self).__init__()
        self.c1 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.n1 = nn.InstanceNorm2d(out_channel)
        self.c2 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.n2 = nn.InstanceNorm2d(out_channel)

    def forward(self, x):
        x1 = self.c1(x)
        x1 = self.n1(x1)

        x2 = self.c2(x)
        x2 = self.n2(x2)

        x3 =  x1 * torch.sigmoid(x2)

        return x3


class Generator(nn.Module):
    """docstring for Generator."""
    def __init__(self):
        super(Generator, self).__init__()
        self.downsample = nn.Sequential(
            Down2d(1, 32, (3,9), (1,1), (1,4)),
            Down2d(32, 64, (4,8), (2,2), (1,3)),
            Down2d(64, 128, (4,8), (2,2), (1,3)),
            Down2d(128, 64, (3,5), (1,1), (1,2)),
            Down2d(64, 5, (9,5), (9,1), (1,2))
        )


        self.up1 = Up2d(9, 64, (9,5), (9,1), (0,2))
        self.up2 = Up2d(68, 128, (3,5), (1,1), (1,2))
        self.up3 = Up2d(132, 64, (4,8), (2,2), (1,3))
        self.up4 = Up2d(68, 32, (4,8), (2,2), (1,3))

        self.deconv = nn.ConvTranspose2d(36, 1, (3,9), (1,1), (1,4))

    def forward(self, x, c):
        x = x.unsqueeze(1)
        x = self.downsample(x)
        c = c.view(c.size(0), c.size(1), 1, 1)

        print(x.size())
        c1 = c.repeat(1, 1, x.size(2), x.size(3))
        print(c1.size())
        x = torch.cat([x, c1], dim=1)
        print(x.size())
        x = self.up1(x)

        c2 = c.repeat(1,1,x.size(2), x.size(3))
        x = torch.cat([x, c2], dim=1)
        x = self.up2(x)

        c3 = c.repeat(1,1,x.size(2), x.size(3))
        x = torch.cat([x, c3], dim=1)
        x = self.up3(x)

        c4 = c.repeat(1,1,x.size(2), x.size(3))
        x = torch.cat([x, c4], dim=1)
        x = self.up4(x)

        c5 = c.repeat(1,1, x.size(2), x.size(3))
        x = torch.cat([x, c5], dim=1)
        x = self.deconv(x)
        return x


class Discriminator(nn.Module):
    """docstring for Discriminator."""
    def __init__(self):
        super(Discriminator, self).__init__()

        self.d1 = Down2d(5, 32, (3,9), (1,1), (1,4))
        self.d2 = Down2d(36, 32, (3,8), (1,2), (1,3))
        self.d3 = Down2d(36, 32, (3,8), (1,2), (1,3))
        self.d4 = Down2d(36, 32, (3,6), (1,2), (1,2))

        self.conv = nn.Conv2d(36, 1, (36,5), (36,1), (0,2))
        self.pool = nn.AvgPool2d((1,64))
    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)

        c1 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c1], dim=1)
        x = self.d1(x)

        c2 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c2], dim=1)
        x = self.d2(x)

        c3 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c3], dim=1)
        x = self.d3(x)

        c4 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c4], dim=1)
        x = self.d4(x)

        c5 = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c5], dim=1)
        x = self.conv(x)

        x = self.pool(x)
        x = torch.squeeze(x)
        x = torch.tanh(x)
        return x

# class Generator(nn.Module):
#
#     def __init__(self):
#         super(Generator, self).__init__()
#
#         #Create layers
#         self.temp_layer = nn.Linear(10,10)
#
#     def forward(self, x, c):
#
#         x = self.temp_layer(x)
#
#         return x
#
# class Discriminator(nn.Module):
#
#     def __init__(self):
#         super(Discriminator, self).__init__()
#
#         #Create layers
#         self.temp_layer = nn.Linear(10,10)
#
#     def forward(self, x, c):
#
#         x = self.temp_layer(x)
#
#         return x

if __name__ == '__main__':

    import yaml
    config = yaml.load(open('./config.yaml', 'r'))

    print("Made config.")

    model = StarGAN_emo_VC1(config, "NewTest")
    # print(model.name)
    load_dir = './checkpoints/NewTest/'

    # model.load(load_dir, 4)

    # print(model.name)

    # model.save(iter = 4)
    model.load(load_dir, 4)
