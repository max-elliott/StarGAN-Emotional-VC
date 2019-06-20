import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import audio_utils

class PostNet(nn.Module):

    def __init__(self, n_input_feats, n_output_feats):
        super(PostNet, self).__init__()

        self.kernel_size = 7
        self.channels = 32

        self.conv1 = nn.ConvTranspose2d(1, self.channels, self.kernel_size)
        self.conv2 = nn.ConvTranspose2d(self.channels, self.channels, self.kernel_size)
        self.conv3 = nn.ConvTranspose2d(self.channels, self.channels, self.kernel_size)
        self.conv4 = nn.ConvTranspose2d(self.channels, self.channels, self.kernel_size)
        self.conv5 = nn.ConvTranspose2d(self.channels, 1, self.kernel_size)

        self.bn1 = nn.BatchNorm2d(self.channels)
        self.bn2 = nn.BatchNorm2d(self.channels)
        self.bn3 = nn.BatchNorm2d(self.channels)
        self.bn4 = nn.BatchNorm2d(self.channels)

        self.relu = nn.LeakyReLU(0.05)

    def forward(self, x):

        edge = int((self.kernel_size-1)/2)
        sizes = x.size()
        # x = x[:,:,edge:-edge,edge:-edge]

        x = self.conv1(x)
        x = self.bn1(self.relu(x))
        x = F.interpolate(x, size = (sizes[2], 224))

        x = self.conv2(x)
        x = self.bn2(self.relu(x))
        x = F.interpolate(x, size = (sizes[2], 320))

        x = self.conv3(x)
        x = self.bn3(self.relu(x))
        x = F.interpolate(x, size = (sizes[2], 416))

        x = self.conv4(x)
        x = self.bn4(self.relu(x))
        x = F.interpolate(x, size = (sizes[2], 507))

        x = self.conv5(x)
        x = x[:,:,edge:-edge,:]

        return x

if __name__ == '__main__':

    model = PostNet(20, 40)

    x = torch.rand(2, 1 ,30,20)

    y = model(x)

    print(y.size())
    print(model.parameters)
    num_params=0
    for p in model.parameters():
        num_params += p.numel()

    print(num_params)
