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
    # parser.add_argument('-f', '--features'), type = str,
                        # help = "mel or world features.")

    args = parser.parse_args()
    config = yaml.load(open('./config.yaml', 'r'))

    checkpoint_dir = '../downloaded/checkpoints/' + args.model + '/' + args.iteration + '.ckpt'
    # checkpoint_dir = '../downloaded/checkpoints/world2_full_2/160000.ckpt'
    out_dir = './samples'
    print("Loading model at ", checkpoint_dir)

    if args.in_dir == 'sample':
        in_dir = '../data/samples/originals'
        files = find_files(in_dir, ext = 'wav')

        filenames = []
        for f in files:
            f = os.path.basename(f)[:-4] + ".wav"
            filenames.append(f)

        print("Converting sample set.")
    else:
        # Is Test Set
        filenames = ['Ses05F_impro04_F023', 'Ses05M_script01_1_M035', 'Ses05F_impro02_F005',
         'Ses03M_script03_2_M003', 'Ses04M_script01_1_F013', 'Ses01M_script01_1_F039',
          'Ses04M_script03_2_F030', 'Ses05F_script03_2_M039', 'Ses01F_impro04_F028',
          'Ses04F_impro01_M021', 'Ses02M_impro06_F006', 'Ses03M_script03_2_F011',
          'Ses03M_impro06_F014', 'Ses04F_script01_1_M012', 'Ses03F_script01_3_M042',
           'Ses03F_script02_2_F005', 'Ses01M_impro01_F008', 'Ses04F_script01_1_M013',
            'Ses05M_script03_2_M029', 'Ses04M_script01_1_F007', 'Ses05F_impro06_M009',
             'Ses01M_script03_2_F035', 'Ses04F_script02_2_F034', 'Ses01M_script03_2_F015',
             'Ses01M_script01_1_F016', 'Ses01F_script03_2_F017', 'Ses05F_impro02_F031',
              'Ses01M_impro06_F004', 'Ses03M_script01_1_F038', 'Ses03M_script01_3_F007',
            'Ses01M_script01_3_M041', 'Ses05M_script01_1_F004', 'Ses03F_script01_1_M027',
                 'Ses03F_script03_2_F040',
                 'Ses04F_impro08_M023', 'Ses03M_impro02_M004', 'Ses04F_script01_1_F004',
                 'Ses04M_script03_2_M055', 'Ses02M_impro06_M018', 'Ses03M_script03_2_M007',
              'Ses05M_script03_2_M008', 'Ses04M_script03_2_M051', 'Ses02M_impro05_M013',
               'Ses05M_script03_2_M038', 'Ses02M_impro06_M023', 'Ses01M_script02_2_F015',
                'Ses03M_impro05b_M025', 'Ses04F_script03_2_F044', 'Ses03F_script03_2_F018',
                 'Ses03M_script01_1_M012', 'Ses04M_impro01_M024', 'Ses02F_script02_2_F035',
              'Ses03F_impro02_F031', 'Ses04M_script03_2_F019', 'Ses04F_impro04_F016',
           'Ses02F_script02_2_F024','Ses01M_impro01_F021', 'Ses02M_script01_1_F035',
           'Ses03F_script03_2_M001', 'Ses03F_impro02_F034', 'Ses05F_script01_1_M035',
            'Ses05M_script03_2_M028', 'Ses05F_script03_2_F018', 'Ses02F_script03_2_M043',
             'Ses01F_script03_2_M021', 'Ses05F_impro06_F004', 'Ses05F_impro04_F022',
              'Ses05M_impro02_F021', 'Ses03F_impro02_M024', 'Ses04M_script01_1_F035',
               'Ses02F_script03_2_F034', 'Ses04F_impro02_F006', 'Ses04M_script01_1_M021',
                'Ses04F_script03_2_F038', 'Ses02M_script01_1_M003', 'Ses03F_script01_3_M031',
                 'Ses03M_impro06_F001', 'Ses01M_script03_2_M024', 'Ses05M_script03_2_M040',
                  'Ses04M_script03_2_F049', 'Ses03M_script03_2_F038', 'Ses01M_script03_2_M042',
                   'Ses04F_script03_2_F026', 'Ses04F_script01_3_M027', 'Ses01M_impro02_F019',
                    'Ses04F_script03_2_M031', 'Ses04M_script01_1_F022', 'Ses04M_script02_1_F003',
                     'Ses02M_script01_1_F033', 'Ses04M_impro06_F014', 'Ses02F_script02_2_F020',
              'Ses03F_impro06_M015', 'Ses03F_script01_2_M015', 'Ses01M_script03_2_M023',
               'Ses03F_script01_2_F006', 'Ses05F_script03_2_M016', 'Ses04F_script03_2_F027',
                        'Ses03M_impro05b_M008', 'Ses05M_script02_2_F031', 'Ses05M_script02_2_F018',
                         'Ses01M_impro06_M019', 'Ses05M_script01_1_F036', 'Ses05M_script03_2_M023',
                          'Ses05M_impro06_F009', 'Ses01M_impro02_M021', 'Ses02M_script02_2_F020',
                           'Ses05F_impro05_F034', 'Ses02F_script03_2_M039', 'Ses04M_impro05_M018',
            'Ses01F_script03_2_F014', 'Ses01M_impro02_F010', 'Ses04M_script02_1_F010',
            'Ses02F_script01_1_F036', 'Ses04F_impro06_M003', 'Ses03F_script01_1_M022',
            'Ses05M_impro06_M020', 'Ses05M_script01_3_F021', 'Ses04F_script02_1_F019',
            'Ses04F_script02_2_M040', 'Ses05F_script03_2_F037', 'Ses02M_script02_2_M038',
            'Ses04F_impro01_M008', 'Ses01M_impro02_M019', 'Ses03F_script01_3_M029',
             'Ses01M_script01_2_F013', 'Ses03M_script03_2_F045', 'Ses03M_impro05b_M026',
             'Ses04M_script03_2_F050', 'Ses01F_script02_2_F036', 'Ses05F_script01_2_F012',
              'Ses02M_impro06_M022', 'Ses04F_script01_2_F006', 'Ses03F_impro06_F024',
              'Ses05M_impro02_M021', 'Ses03F_impro08_M011', 'Ses01M_impro06_M010',
              'Ses03F_script01_3_M032', 'Ses01M_impro06_M000', 'Ses05M_script03_2_M036',
              'Ses03F_impro06_F007', 'Ses05F_impro02_M032', 'Ses02F_script02_2_M037',
              'Ses01M_script03_2_F025', 'Ses05M_script03_2_F035', 'Ses03F_impro06_F015',
              'Ses03M_impro02_F026', 'Ses04M_script01_3_F022', 'Ses01M_script03_2_F034',
              'Ses05F_script01_2_F010', 'Ses05F_impro02_M004', 'Ses01F_impro02_F017',
              'Ses05M_script01_1_M023', 'Ses05F_script01_2_M017', 'Ses02M_script01_2_F003',
              'Ses04F_script01_1_F029', 'Ses03M_impro05a_M021', 'Ses02F_script03_2_F036',
              'Ses03M_impro02_F016', 'Ses05M_script03_2_M032', 'Ses04M_script02_1_F002',
              'Ses04F_script03_2_M037', 'Ses04M_script03_2_F000', 'Ses02F_script01_2_F013',
              'Ses04F_impro02_M001', 'Ses01F_impro06_F022', 'Ses04F_script01_1_M038',
              'Ses03M_script01_2_F010', 'Ses02M_impro01_M009', 'Ses02F_script03_2_F029',
              'Ses01M_impro06_M002', 'Ses01F_script01_1_F037', 'Ses01F_script01_2_F004',
              'Ses03M_impro02_F025', 'Ses04F_script01_1_F035', 'Ses01M_impro05_M023',
              'Ses03M_script03_2_M041', 'Ses05M_script02_2_F005', 'Ses04M_script03_2_M038',
              'Ses04M_impro06_M011', 'Ses01M_impro06_M022', 'Ses04M_script03_2_M027',
              'Ses01M_script02_1_F014', 'Ses02F_script03_2_F042', 'Ses02M_impro04_F009',
              'Ses02M_script01_1_M038', 'Ses04M_script03_2_M041', 'Ses01F_impro06_F013',
              'Ses01M_script01_1_F029', 'Ses04F_script03_2_F037']
        filenames = [f+".wav" for f in filenames]
        print("Converting test set.")
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

    # for one_hot in emo_targets:
    #     _single_conversion(filenames[0], model, one_hot)

    # print(filenames)


    ########################################
    #        WORLD CONVERSION LOOP         #
    ########################################
    for f in filenames:

        wav, labels = pp.get_wav_and_labels(f, config['data']['dataset_dir'])
        wav = np.array(wav, dtype = np.float64)
        labels = np.array(labels)
        f0_real, ap_real, sp, coded_sp = preprocess_world.cal_mcep(wav)
        coded_sp = coded_sp.T
        coded_sp = torch.Tensor(coded_sp).unsqueeze(0).unsqueeze(0).to(device = device)

        with torch.no_grad():
            # print(emo_targets)
            for i in range (0, emo_targets.size(0)):

                f0 = np.copy(f0_real)
                ap = np.copy(ap_real)
                # coded_sp = np.copy(coded_sp)

                fake = model.G(coded_sp, emo_targets[i].unsqueeze(0))

                # print(f"Converting {f[0:-4]}.")
                filename_wav =  f[0:-4] + "_" + str(int(labels[0].item())) + "to" + \
                            str(i) + ".wav"

                fake = fake.squeeze()
                print("Sampled size = ",fake.size())
                # f = fake.data()
                converted_sp = fake.cpu().numpy()
                converted_sp = np.array(converted_sp, dtype = np.float64)

                sample_length = converted_sp.shape[0]
                if sample_length != ap.shape[0]:
                    ap = np.ascontiguousarray(ap[0:sample_length, :], dtype = np.float64)
                    f0 = np.ascontiguousarray(f0[0:sample_length], dtype = np.float64)

                f0 = np.ascontiguousarray(f0[40:-40], dtype = np.float64)
                ap = np.ascontiguousarray(ap[40:-40,:], dtype = np.float64)
                converted_sp = np.ascontiguousarray(converted_sp[40:-40,:], dtype = np.float64)

                print("ap shape = ", ap.shape)
                print("f0 shape = ", f0.shape)

                name = str(args.iteration)[0:3]
                audio_utils.save_world_wav([f0,ap,sp,converted_sp], model.name +"_"+ name+'_testSet', filename_wav)
        print(f, " converted.")

    ########################################
    #         MEL CONVERSION LOOP          #
    ########################################
    # Make .npy arrays
    # Make audio
    # Make spec plots

    # Save all to directory
