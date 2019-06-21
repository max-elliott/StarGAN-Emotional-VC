import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import audio_utils

class Solver(object):

    def __init__(self, model, train_loader, test_loader, config):

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config

    def train(self):
        '''
        Main training loop
        '''
