import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import audio_utils

class UpConvBlock1D(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_length, stride = 1, padding = 1):
        super(UpConvBlock1D, self).__init__()

        self.conv1 = nn. ConvTranspose1d(in_channel, out_channel, kernel_size = kernel_length,
                                        stride= stride, padding = padding)
        self.conv_norm = nn.InstanceNorm1d(out_channel)

        self.gate1 = nn. ConvTranspose1d(in_channel, out_channel, kernel_size = kernel_length,
                                        stride= stride, padding = padding)
        self.gate_norm = nn.InstanceNorm1d(out_channel)

    def forward(self, x):

        #Unsqueeze?

        conv_x = self.conv_norm(F.relu(self.conv1(x))) #Need relu?
        gate_x = self.gate_norm(self.gate1(x))

        x_out = conv_x * torch.sigmoid(gate_x)

        return x_out

class PostNet2(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_length, stride = 1, padding = 1):
        super(PostNet2, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        # self.increase = (out_channel - in_channel)

        self.cb1 = UpConvBlock1D(self.in_channel, 224, kernel_length,
                                        stride = stride, padding = padding)
        self.cb2 = UpConvBlock1D(224, 320, kernel_length,
                                        stride = stride, padding = padding)
        self.cb3 = UpConvBlock1D(320, 416, kernel_length,
                                        stride = stride, padding = padding)
        self.cb4 = UpConvBlock1D(416, self.out_channel, kernel_length,
                                        stride = stride, padding = padding)

    def forward(self, x):

        x = self.cb4(self.cb3(self.cb2(self.cb1(x))))

        return x

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

    model = PostNet2(128, 513, 3)

    x = torch.rand(2, 128 , 30)

    y = model(x)

    print(y.size())

    # print(y.size())
    # print(model.parameters)
    # num_params=0
    # for p in model.parameters():
    #     num_params += p.numel()
    #
    # print(num_params)
