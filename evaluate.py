import argparse
import torch
import torch.nn.functional as F
import yaml
import numpy as np
import random
import os


if __name__=='__main__':

    # Parse args:
    #   model checkpoint
    #   files to be converted
    #   save directory

    # Retrieve files
    # Make emotion targets (using config file)

    # For each f:
    #   Transpose f
    #   Make batch of (num_classes, 1, f.size(0), f.size(1))
    #   Pass through generator

    # Make .npy arrays
    # Make audio
    # Make spec plots

    # Save all to directory
