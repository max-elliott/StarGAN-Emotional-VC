import argparse
import torch
import torch.nn.functional as F
import yaml
import numpy as np
import random
import os

import data_preprocessing as pp
import audio_utils
import my_dataset
from my_dataset import get_filenames
import solver
import solver_recon

if __name__ == '__main__':

    # ADD ALL CONFIG ARGS
    parser = argparse.ArgumentParser(description='StarGAN-emo-VC')
    parser.add_argument("-n", "--name", type = str, default = None,
                    help="Model name for training.")
    parser.add_argument("-c","--checkpoint", type=str, default = None,
                    help="Directory of checkpoint to resume training from")
    parser.add_argument("-s", "--segment_len", type = int, default = None,
                    help="Set utterance length if using fixed lengths")
    parser.add_argument("-e", "--evaluate", action = 'store_true',
                    help="False = train, True = evaluate model")
    parser.add_argument("-a", "--alter", action = 'store_true')
    parser.add_argument("-r", "--recon", action = 'store_true')

    args = parser.parse_args()

    config = yaml.load(open('./config.yaml', 'r'))
    if args.name != None:
        config['model']['name'] = args.name
        print(config['model']['name'])

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
    else:
        device = torch.device('cpu')

    print(torch.cuda.is_available())
    print(torch.__version__)
    print(torch.cuda)

    # MAKE TRAIN + TEST SPLIT
    mel_dir = os.path.join(config['data']['dataset_dir'], "mels")
    files = get_filenames(mel_dir)

    #UNCOMMENT LATER
    files = [f for f in files if os.path.basename(f)[0:6]=='Ses01F']
    print(len(files))

    files = my_dataset.shuffle(files)

    train_test_split = config['data']['train_test_split']
    split_index = int(len(files)*train_test_split)
    train_files = files[:split_index]
    test_files = files[split_index:]

    print(len(train_files))
    print(len(test_files))
    # print(train_files[0:20])
    print(test_files)

    train_dataset = my_dataset.MyDataset(config, train_files)
    test_dataset = my_dataset.MyDataset(config, test_files)

    batch_size = config['model']['batch_size']

    train_loader, test_loader = my_dataset.make_variable_dataloader(train_dataset,
                                                                    test_dataset,
                                                                    batch_size = batch_size)

    # Run solver
    # load_dir = './checkpoints/NewSolver/00006.ckpt'
    load_dir = args.checkpoint

    if args.recon:
        print("Performing reconstructoin training.")
        s = solver_recon.Solver(train_loader, test_loader, config, load_dir = load_dir)
    else:
        print("Performing whole network training.")
        s = solver.Solver(train_loader, test_loader, config, load_dir = load_dir)

    if args.alter:
        print("Changing loaded config to new config.")
        s.config = config
        s.set_configuration()

    if not args.evaluate:
        print("Training model.")
        s.train()
    else:
        print("No training. Model loaded in evaluation mode.")

    # for i, (x,y) in train_loader:
    #

    # TEST MODEL COMPONENTS
    # data_iter = iter(train_loader)
    #
    # x, y = next(data_iter)
    #
    # x_lens = x[1]
    # x = x[0].unsqueeze(1)
    # # x = x[:,:,0:80]
    # # print(x.size(), y.size())
    #
    # targets = s.make_random_labels(4, batch_size)
    # targets_one_hot = F.one_hot(targets, num_classes = 4).float()
    #
    # # out = s.model.G(input, targets)
    # g_out = s.model.G(x, targets_one_hot)
    # print('g_out = ', g_out.size())
    # d_out = s.model.D(g_out, targets_one_hot)
    # print('d_out = ', d_out)
    # # WHY DIFFERNT LENGTH OUTPUT????
    # out = s.model.emo_cls(g_out, x_lens)
    # print('c_out = ',out)
