import argparse
import torch
import yaml

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='StarGAN-emo-VC')

    # ADD ALL CONFIG ARGS

    # beta1, beta2
    # num_feats (how many mels)
    # G_lr, D_lr, emo_cls_lr, speaker_cls_lr, dim_cls_lr

    config = yaml.load(open('./config.yaml', 'r'))
    config_opt = config['model']['optimizer']
    print(config_opt)
