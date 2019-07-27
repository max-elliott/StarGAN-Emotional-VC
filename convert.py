import argparse
import torch
import torch.nn.functional as F
import yaml
import numpy as np
import random
import os

import librosa
from librosa.util import find_files
import pyworld
from pyworld import decode_spectral_envelope, synthesize

from matplotlib import pyplot as plt

import solver
import model
import audio_utils
import data_preprocessing2 as pp
import preprocess_world

if __name__=='__main__':

    # Parse args:
    #   model checkpoint
    #   directory of wav files to be converted
    #   save directory
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', type = str,
                        help = "Model checkpoint to use for conversion.")
    parser.add_argument('-i', '--in_dir', type = str)
    parser.add_argument('-o', '--out_dir', type = str)
    # parser.add_argument('-f', '--features'), type = str,
                        # help = "mel or world features.")

    args = parser.parse_args()
    config = yaml.load(open('./config.yaml', 'r'))

    in_dir = '../data/samples/originals'
    checkpoint_dir = '../downloaded/checkpoints/world2_1_10000.ckpt'
    out_dir = './samples'

    #fix seeds to get consistent results
    SEED = 42
    # torch.backend.cudnn.deterministic = True
    # torch.backend.cudnn.benchmark = False
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


    # Use GPU
    USE_GPU = True

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(SEED)
        map_location='cuda'
    else:
        device = torch.device('cpu')
        map_location='cpu'

    # Load model
    model = model.StarGAN_emo_VC1(config, config['model']['name'])
    # model.load(args.checkpoint)
    model.load(checkpoint_dir, map_location= map_location)
    config = model.config
    model.to_device(device = device)
    model.set_eval_mode()

    # Make emotion targets (using config file)
    # s = solver.Solver(None, None, config, load_dir = None)
    # targets =
    num_emos = config['model']['num_classes']
    emo_labels = torch.Tensor(range(0,num_emos)).long()
    emo_targets = F.one_hot(emo_labels, num_classes = num_emos).float().to(device = device)
    print(f"Number of emotions = {num_emos}")

    # For each f:
    #   Transpose f
    #   Make batch of (num_classes, 1, f.size(0), f.size(1))
    #   Pass through generator
        # Retrieve files

    files = find_files(in_dir, ext = 'wav')

    filenames = []
    for f in files:
        f = os.path.basename(f)[:-4] + ".wav"
        filenames.append(f)

    print(filenames)

    wav, labels = pp.get_wav_and_labels(filenames[0], config['data']['dataset_dir'])
    wav = np.array(wav, dtype = np.float64)
    labels = np.array(labels)
    f0, ap, sp, coded_sp = preprocess_world.cal_mcep(wav)
    coded_sp = coded_sp.T

    coded_sp_torch = torch.Tensor(coded_sp).unsqueeze(0).unsqueeze(0).to(device = device)

    fake = model.G(coded_sp_torch, emo_targets[0].unsqueeze(0))

    # print(f[0:-4])
    # filename_wav =  f[0:-4] + "_" + str(int(labels[0].item())) + "to" + \
    #             str(0) + ".npy"

    fake = fake.squeeze()
    print("Sampled size = ",fake.size())
    # f = fake.data()
    converted_sp = fake.cpu().detach().numpy()
    converted_sp = np.array(converted_sp, dtype = np.float64)



    coded_sp = audio_utils._unnormalise_coded_sp(coded_sp)
    converted_sp = audio_utils._unnormalise_coded_sp(converted_sp)

    print(coded_sp[1000:1002,:])
    print(converted_sp[1000:1002,:])

    i1 = plt.figure(1)
    plt.imshow(coded_sp)
    i2 = plt.figure(2)
    plt.imshow(converted_sp)
    # i1.show()
    # i2.show()
    plt.show()

    ########################################
    #        WORLD CONVERSION LOOP         #
    ########################################
    # for f in filenames:
    #
    #     wav, labels = pp.get_wav_and_labels(f, config['data']['dataset_dir'])
    #     wav = np.array(wav, dtype = np.float64)
    #     labels = np.array(labels)
    #     f0, ap, sp, coded_sp = preprocess_world.cal_mcep(wav)
    #     coded_sp = coded_sp.T
    #     coded_sp = torch.Tensor(coded_sp).unsqueeze(0).unsqueeze(0).to(device = device)
    #
    #     with torch.no_grad():
    #         # print(emo_targets)
    #         for i in range (0, emo_targets.size(0)):
    #
    #             fake = model.G(coded_sp, emo_targets[i].unsqueeze(0))
    #
    #             print(f[0:-4])
    #             filename_wav =  f[0:-4] + "_" + str(int(labels[0].item())) + "to" + \
    #                         str(i) + ".wav"
    #
    #             fake = fake.squeeze()
    #             print("Sampled size = ",fake.size())
    #             # f = fake.data()
    #             converted_sp = fake.cpu().numpy()
    #             converted_sp = np.array(converted_sp, dtype = np.float64)
    #
    #             sample_length = fake.shape[0]
    #             if sample_length != ap.shape[0]:
    #                 ap = np.ascontiguousarray(ap[0:sample_length, :], dtype = np.float64)
    #                 f0 = np.ascontiguousarray(f0[0:sample_length], dtype = np.float64)
    #
    #             print("ap shape = ", ap.shape)
    #             print("f0 shape = ", f0.shape)
    #
    #             audio_utils.save_world_wav([f0,ap,sp,converted_sp], model.name + '_test', filename_wav)
    #     print(f, " converted.")

    ########################################
    #         MEL CONVERSION LOOP          #
    ########################################
    # Make .npy arrays
    # Make audio
    # Make spec plots

    # Save all to directory
