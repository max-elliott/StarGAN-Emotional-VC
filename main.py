import argparse
import torch
import torch.nn.functional as F
import yaml
import numpy as np
import random
import os
import librosa
from librosa.util import find_files

import data_preprocessing as pp
import audio_utils
import my_dataset
from my_dataset import get_filenames
import solver
import solver_recon

def make_weight_vector(filenames, data_dir):

    label_dir = os.path.join(data_dir, 'labels')

    emo_labels = []

    for f in filenames:

        label = np.load(label_dir + "/" + f + ".npy")
        emo_labels.append(label[0])

    categories = list(set(emo_labels))
    total = len(emo_labels)

    counts = [total/emo_labels.count(emo) for emo in range(len(categories))]

    weights = torch.Tensor(counts)

    return weights

    # self.emo_loss_weights = torch.Tensor([4040./549, 4040./890,
    #                                          4040./996, 4040./1605]).to(self.device)


if __name__ == '__main__':

    # ADD ALL CONFIG ARGS
    parser = argparse.ArgumentParser(description='StarGAN-emo-VC main method')
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
    parser.add_argument("-f", "--features", type = str)

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

    # Get correct data directory depending on features being used
    if config['data']['type'] == 'world':
        print("Using WORLD features.")
        assert config['model']['num_feats'] == 36
        config['data']['type'] = 'world'
        data_dir = os.path.join(config['data']['dataset_dir'], "world")
    else:
        print("Using mel spectrograms.")
        assert config['model']['num_feats'] == 80
        config['data']['type'] = 'mel'
        data_dir = os.path.join(config['data']['dataset_dir'], "mels")

    print("Data directory = ", data_dir)

    # MAKE TRAIN + TEST SPLIT
    files = get_filenames(data_dir)


    # , 'Ses01M', 'Ses02F', 'Ses02M', 'Ses03F', 'Ses03M'
    sessions = ['Ses01F']
    label_dir = os.path.join(config['data']['dataset_dir'], 'labels')
    num_emos = config['model']['num_classes']
    # files = [f for f in files if os.path.basename(f)[0:6] in sessions]
    files = [f for f in files if np.load(label_dir + "/" + f + ".npy")[0] < num_emos]
    print(len(files), " files used.")
    weight_vector = make_weight_vector(files, config['data']['dataset_dir'])
    # print(weight_vector)

    files = my_dataset.shuffle(files)

    train_test_split = config['data']['train_test_split']
    split_index = int(len(files)*train_test_split)
    train_files = files[:split_index]
    test_files = files[split_index:]

    # print(test_files)

    print(f"Training samples: {len(train_files)}")
    print(f"Test samples: {len(test_files)}")
    # print(train_files[0:20])
    # print(test_files)

    # print(np.load(data_dir + "/" + files[0] + ".npy").shape)

    train_dataset = my_dataset.MyDataset(config, train_files)
    test_dataset = my_dataset.MyDataset(config, test_files)

    batch_size = config['model']['batch_size']

    train_loader, test_loader = my_dataset.make_variable_dataloader(train_dataset,
                                                                    test_dataset,
                                                                    batch_size = batch_size)

    # train_iter = iter(train_loader)
    # x,y = next(train_iter)
    # print(x[0].size())
    # Run solver
    # load_dir = './checkpoints/NewSolver/00006.ckpt'
    load_dir = args.checkpoint

    if args.recon:
        print("Performing reconstruction training.")
        s = solver_recon.Solver(train_loader, test_loader, config, load_dir = load_dir)
    else:
        print("Performing whole network training.")
        s = solver.Solver(train_loader, test_loader, config, load_dir = load_dir)

    if args.alter:
        print("Changing loaded config to new config.")
        s.config = config
        s.set_configuration()

    s.set_classification_weights(weight_vector)

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
    # targets = s.make_random_labels(num_emos, batch_size)
    # targets_one_hot = F.one_hot(targets, num_classes = num_emos).float()
    #
    # print('g_in =', x.size())
    # # out = s.model.G(input, targets)
    # g_out = s.model.G(x, targets_one_hot)
    # print('g_out = ', g_out.size())
    # d_out = s.model.D(g_out, targets_one_hot)
    # print('d_out = ', d_out)
    # # WHY DIFFERNT LENGTH OUTPUT????
    # out = s.model.emo_cls(g_out, x_lens)
    # print('c_out = ',out)
