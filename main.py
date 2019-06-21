import argparse
import torch
import yaml
import numpy as np
import data_preprocessing as pp
import audio_utils


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='StarGAN-emo-VC')

    # ADD ALL CONFIG ARGS

    # beta1, beta2
    # num_feats (how many mels)
    # G_lr, D_lr, emo_cls_lr, speaker_cls_lr, dim_cls_lr

    config = yaml.load(open('./config.yaml', 'r'))
    config_opt = config['model']['optimizer']
    print(config_opt)


    # l1 = [0,1,2,3]
    # l2 = [4,5,6,7]
    # l3 = [np.array([1,2,3]), np.array([4,5,6]), np.array([7,8,9]), np.array([10,11,12])]
    # l4 = [np.array([1,2,3]), np.array([4,5,6]), np.array([7,8,9]), np.array([10,11,12])]
    #
    # print(concatenate_targets(l1,l2,l3,l4))
