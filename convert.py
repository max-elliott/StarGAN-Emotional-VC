import argparse
import torch
import torch.nn.functional as F
import yaml
import numpy as np
import random
import os
import pickle

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

def _single_conversion(filename, model, one_hot_emo):
    '''
    Call only from __main__ section in this module. Generates sample converted
    into each emotion.

    (str) filename - name.wav file to be converted
    (StarGAN-emo-VC1) model - pretrained model to perform conversion
    (torch.Tensor(long)) one_hot_emo - one hot encoding of emotion to convert to
    '''
    wav, labels = pp.get_wav_and_labels(filenames[5], config['data']['dataset_dir'])
    wav = np.array(wav, dtype = np.double)

    f0, ap, sp, coded_sp = preprocess_world.cal_mcep(wav)

    coded_sp = coded_sp.T

    coded_sp_torch = torch.Tensor(coded_sp).unsqueeze(0).unsqueeze(0).to(device = device)

    fake = model.G(coded_sp_torch, one_hot_emo.unsqueeze(0))
    fake = fake.squeeze()

    print("Sampled size = ",fake.size())

    converted_sp = fake.cpu().detach().numpy()
    converted_sp = np.array(converted_sp, dtype = np.float64)

    sample_length = converted_sp.shape[0]
    if sample_length != ap.shape[0]:
        ap = np.ascontiguousarray(ap[0:sample_length, :], dtype = np.float64)
        f0 = np.ascontiguousarray(f0[0:sample_length], dtype = np.float64)

    f0 = np.ascontiguousarray(f0[40:-40], dtype = np.float64)
    ap = np.ascontiguousarray(ap[40:-40,:], dtype = np.float64)
    converted_sp = np.ascontiguousarray(converted_sp[40:-40,:], dtype = np.float64)

    coded_sp = np.ascontiguousarray(coded_sp[40:-40,:], dtype = np.float64)

    target = np.argmax(one_hot_emo)
    out_name = filename[:-4] + str(labels[1]) + "to" + target + ".wav"


    audio_utils.save_world_wav([f0,ap,sp,converted_sp], model.name + '_test', out_name)

    # print(converted_sp[0, :])
    # converted_sp[0:3, :] = converted_sp[0:3, :]/1.15
    # print(converted_sp[0, :])

    # audio_utils.save_world_wav([f0,ap,sp,converted_sp], 'tests', 'after.wav')

    # DON'T DO: IS DONE IN SAVE FUNCTION
    # coded_sp = audio_utils._unnormalise_coded_sp(coded_sp)
    # converted_sp = audio_utils._unnormalise_coded_sp(converted_sp)

    # i1 = plt.figure(1)
    # plt.imshow(coded_sp[:40,:])#[1200:1250,2:])
    # i2 = plt.figure(2)
    # plt.imshow(converted_sp[:40,:])#[1200:1250,2:])
    # plt.show()

    # h1 = plt.figure(1)
    # n, bins, patches = plt.hist(coded_sp, bins = 20)
    # h1 = plt.figure(2)
    # n, bins, patches = plt.hist(converted_sp, bins = 20)
    # plt.xlabel('Sequence length')
    # plt.ylabel('Count')
    # plt.title(r'New histogram of sequence lengths for 4 emotional categories')
    # plt.show()

if __name__=='__main__':

    # Parse args:
    #   model checkpoint
    #   directory of wav files to be converted
    #   save directory
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type = str,
                        help = "Model to use for conversion.")
    parser.add_argument('-in', '--in_dir', type = str)
    parser.add_argument('-out', '--out_dir', type = str)
    parser.add_argument('-i', '--iteration', type = str)
    # parser.add_argument('-n', '--num_emotions', type = int, default = None)
    # parser.add_argument('-f', '--features'), type = str,
                        # help = "mel or world features.")

    args = parser.parse_args()
    config = yaml.load(open('./config.yaml', 'r'))

    checkpoint_dir = '../downloaded/checkpoints/' + args.model + '/' + args.iteration + '.ckpt'
    # checkpoint_dir = '../downloaded/checkpoints/world2_full_2/160000.ckpt'

    print("Loading model at ", checkpoint_dir)

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

    if args.in_dir == 'sample':
        in_dir = '../data/samples/originals'
        files = find_files(in_dir, ext = 'wav')

        filenames = []
        for f in files:
            f = os.path.basename(f)[:-4] + ".wav"
            filenames.append(f)

        print("Converting sample set.")
    elif args.in_dir == 'neutral':
        in_dir = '../data/audio'
        files = find_files(in_dir, ext = 'wav')
        filenames = [os.path.basename(f)[:-4] + ".wav" for f in files]

        filenames = [f for f in filenames if pp.get_wav_and_labels(f, config['data']['dataset_dir'])[1][0]==3]
        random.shuffle(filenames)
        filenames = filenames[0:30]

        print(len(filenames))

    else:
        if num_emos == 2:
            with open('./2_emotions_testset.pkl', 'rb') as fp:
                filenames = pickle.load(fp)
        elif num_emos == 3:
            with open('./3_emotions_testset.pkl', 'rb') as fp:
                filenames = pickle.load(fp)
        filenames = [f+".wav" for f in filenames]
        filenames = [f for f in filenames if pp.get_wav_and_labels(f, config['data']['dataset_dir'])[1][0]<num_emos]
        filenames = [f for f in filenames if pp.get_wav_and_labels(f, config['data']['dataset_dir'])[1][0]==2]
        print(filenames)
        print("Converting test set.")
    # for one_hot in emo_targets:
    #     _single_conversion(filenames[0], model, one_hot)

    # filenames = ['Ses01F_impro02_F014',     #ANRGY
    #             'Ses01M_impro01_F007',
    #             'Ses01M_impro01_F021',
    #             'Ses01M_script03_2_M023',
    #             'Ses03F_script01_1_M022',
    #             'Ses03F_script02_2_F005',
    #             'Ses04F_impro01_M008',
    #             'Ses04F_impro08_M023',
    #             'Ses04M_script01_1_F022',
    #             'Ses05M_script03_2_M032',
    #             'Ses02F_script02_2_F020',    #SAD
    #             'Ses02M_script01_1_M003',
    #             'Ses02M_script02_2_F020',
    #             'Ses03F_impro02_F031',
    #             'Ses03F_impro06_F015',
    #             'Ses03F_script01_3_M032',
    #             'Ses03M_impro02_F025',
    #             'Ses03M_impro02_F026',
    #             'Ses04M_script01_1_M021',
    #             'Ses05M_impro06_F009'
    #             'Ses01F_script01_3_M019',   #HAPPY
    #             'Ses01M_script03_1_F003',
    #             'Ses02M_impro07_F024',
    #             'Ses03F_impro07_M035',
    #             'Ses03M_impro03_F026',
    #             'Ses03M_script01_3_F020',
    #             'Ses04M_impro03_F024',
    #             'Ses04M_script01_3_F013',
    #             'Ses05F_impro03_F046',
    #             'Ses05M_impro08_F028'
    #             ]
    # filenames = [f+".wav" for f in filenames]
    #
    # shuffle_dir = "./samples/final/Evaluation/Stage_B/Angrier"
    # filenames = librosa.util.find_files(shuffle_dir, ext='wav')
    # filenames = [os.path.basename(f)[:-4] + ".wav" for f in filenames]
    #
    # for f in filenames:
    #     print(f[:-4])

    # in_dir = '../data/labels'
    # files = find_files(in_dir, ext = 'npy')
    # filenames = [os.path.basename(f)[:-4] + ".wav" for f in files]
    # print("Found", len(filenames), " files.")
    #
    # filenames = [f for f in filenames if pp.get_wav_and_labels(f, config['data']['dataset_dir'])[1][1] in range(0,6)]
    # random.shuffle(filenames)
    # filenames = filenames[:10]
    # print(filenames)
    # print("Number of files to be converted = ", len(filenames))

    filenames = ["../data/stairway.wav"]
    wav = audio_utils.load_wav(filenames[0])
    labels = [3, 1, 0,0,0,0,0,0]
    ########################################
    #       BASELINE SYNTHESIS LOOP        #
    ########################################
    # for f in filenames:
    #
    #     wav, labels = pp.get_wav_and_labels(f, config['data']['dataset_dir'])
    #     wav = np.array(wav, dtype = np.float64)
    #     labels = np.array(labels)
    #     f0, ap, sp, coded_sp = preprocess_world.cal_mcep(wav)
    #     coded_sp = coded_sp.T
    #     # coded_sp_temp = np.copy(coded_sp).T
    #     # print(coded_sp_temp.shape)
    #     filename_wav =  f[0:-4] + "_" + str(int(labels[0].item())) + ".wav"
    #     print(coded_sp.shape)
    #     it = str(args.iteration)[0:3]
    #     audio_utils.save_world_wav([f0,ap,sp,coded_sp], args.out_dir + '_evalSet', filename_wav)

    ########################################
    #        WORLD CONVERSION LOOP         #
    ########################################
    for file_num, f in enumerate(filenames):

        # wav, labels = pp.get_wav_and_labels(f, config['data']['dataset_dir'])
        wav = np.array(wav, dtype = np.float64)
        labels = np.array(labels)
        f0_real, ap_real, sp, coded_sp = preprocess_world.cal_mcep(wav)
        # coded_sp_temp = np.copy(coded_sp).T
        # print(coded_sp_temp.shape)
        coded_sp = coded_sp.T
        coded_sp = torch.Tensor(coded_sp).unsqueeze(0).unsqueeze(0).to(device = device)

        with torch.no_grad():
            # print(emo_targets)
            for i in range (0, emo_targets.size(0)):


                f0 = np.copy(f0_real)
                ap = np.copy(ap_real)
                # coded_sp_temp_copy = np.copy(coded_sp_temp)
                # coded_sp = np.copy(coded_sp)
                f0 = audio_utils.f0_pitch_conversion(f0, (labels[0],labels[1]),
                                                         (i, labels[1]))

                fake = model.G(coded_sp, emo_targets[i].unsqueeze(0))

                # print(f"Converting {f[0:-4]}.")
                # filename_wav =  f[0:-4] + str(int(labels[0].item())) + "to" + \
                #             str(i) + ".wav"
                filename_wav = "stairway" + str(int(labels[0].item())) + "to" + \
                            str(i) + ".wav"

                fake = fake.squeeze()
                # print("Sampled size = ",fake.size())
                # f = fake.data()
                converted_sp = fake.cpu().numpy()
                converted_sp = np.array(converted_sp, dtype = np.float64)

                sample_length = converted_sp.shape[0]
                if sample_length != ap.shape[0]:
                    # coded_sp_temp_copy = np.ascontiguousarray(coded_sp_temp_copy[0:sample_length, :], dtype = np.float64)
                    ap = np.ascontiguousarray(ap[0:sample_length, :], dtype = np.float64)
                    f0 = np.ascontiguousarray(f0[0:sample_length], dtype = np.float64)

                f0 = np.ascontiguousarray(f0[20:-20], dtype = np.float64)
                ap = np.ascontiguousarray(ap[20:-20,:], dtype = np.float64)
                converted_sp = np.ascontiguousarray(converted_sp[20:-20,:], dtype = np.float64)
                # coded_sp_temp_copy = np.ascontiguousarray(coded_sp_temp_copy[40:-40,:], dtype = np.float64)

                # print("ap shape = ", ap.shape)
                # print("f0 shape = ", f0.shape)
                print(converted_sp.shape)
                it = str(args.iteration)[0:3]
                audio_utils.save_world_wav([f0,ap,sp,converted_sp], args.out_dir +"_"+ it, filename_wav)
        # print(f, " converted.")
        if (file_num+1) % 20 == 0:
            print(file_num+1, " done.")

    ########################################
    #         MEL CONVERSION LOOP          #
    ########################################
    ### NEVER IMPLEMENTED AS ENDED UP NOT USING MEL SPECTROGRAMS
    # Make .npy arrays
    # Make audio
    # Make spec plots

    # Save all to directory
