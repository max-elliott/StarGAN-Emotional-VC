import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os

import audio_utils
import my_dataset
import classifiers

import torchvision
import sklearn
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

USE_GPU = True

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def save_checkpoint(state, filename='./checkpoint.pth'):

    print("Saving a new best model")
    torch.save(state, filename)  # save checkpoint


def load_checkpoint(model, optimiser, filename = './checkpoint.pth'):

    checkpoint = torch.load(filename)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
    epoch = checkpoint['epoch']
    loss_fn = checkpoint['loss_fn']

    return model, optimiser, loss_fn, epoch

def train_model(model, optimiser, train_data_loader, val_data_loader, loss_fn,
                model_type = 'cls', epochs=1, print_every = 1, var_len_data = False, start_epoch = 1):

    model = model.to(device=device)  # move the model parameters to CPU/GPU

    best_model_score = 0.  #best f1_score for saving checkpoints

    for e in range(start_epoch, epochs+1):

        total_loss = 0


        for t, (x, y) in enumerate(train_data_loader):
            model.train()  # put model to training mode

            if(var_len_data):
                x_real = x[0].to(device = self.device).unsqueeze(1)
                x_lens = x[1].to(device = self.device)
            else:
                x = x.to(device=device, dtype=torch.float)


            y = y[:,0].to(device=device, dtype=torch.float)


#             tf.summary.scalar("lr", optimiser.state_dict()['param_groups'][0]['lr'])
            # Zero out all of the gradients for the variables which the optimiser
            # will update.
            optimiser.zero_grad()

            predictions = model(x)

            #       predictions = predictions.squeeze(0)

            loss = loss_fn(predictions.float(), y.long())

#             tf.summary.scalar("loss", loss.item())
            loss.backward()

            optimiser.step()

            total_loss += loss.item()

        if t % print_every == 0:

            print(f'| Epoch: {e:02} | Train Loss: {total_loss:.3f}')

            acc, f1, UAR = test_model(model, val_data_loader,
                                 var_len_data = var_len_data,
                                 model_type = model_type)

#             log_writer.add_scalar('f1', f1)
#             log_writer.add_scalar('lr', optimiser.state_dict()['param_groups'][0]['lr'])

            print("Accuracy = ",acc*100,"%")
            print(f"Macro-f1 score =", f1)
#             print(f"UA-Recall =", UAR)
            print()

            if model_type == 'cls':

                if f1 > best_model_score:

                    print(f"######################## New best model. f1 = {f1: .3f} ########################")
                    best_model_score = f1

                    state = {
                            'epoch': e,
                            'model_state_dict': model.state_dict(),
                            'optimiser_state_dict': optimiser.state_dict(),
                            'loss_fn': loss_fn}
                    save_checkpoint(state)

def test_model(model, test_loader, var_len_data = False, model_type = 'cls'):

    model = model.to(device=device)
    model.eval()

    actual_preds = torch.rand(0).to(device = device, dtype = torch.long)

    total_y = torch.rand(0).to(device = device, dtype = torch.long)

    for i, (x,y) in enumerate(test_loader):

        if(var_len_data):
            x_real = x[0].to(device = self.device).unsqueeze(1)
            x_lens = x[1].to(device = self.device)
        else:
            x = x.to(device=device, dtype=torch.float)


        y = y[:,0].to(device=device, dtype=torch.float)

        preds = model(x)

        preds = torch.max(preds, dim = 1)[1]

        actual_preds = torch.cat((actual_preds, preds), dim=0)
        total_y = torch.cat((total_y, y), dim=0)



    print(actual_preds.size()[0], "total validation predictions.")
    print(actual_preds[0:100])

    acc = accuracy_score(total_y.cpu(), actual_preds.cpu())
    f1 = f1_score(total_y.cpu(), actual_preds.cpu(), average = 'macro')
    UAR = recall_score(total_y.cpu(), actual_preds.cpu(), average = 'weighted')

    return acc, f1, UAR

if __name__=='__main__':

    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    num_classes = 4
    n_epochs = 200
    hidden_size = 128
    input_size = 80
    num_layers = 2

    config = yaml.load(open('./config.yaml', 'r'))

    # MAKE TRAIN + TEST SPLIT
    mel_dir = os.path.join(config['data']['dataset_dir'], "mels")
    files = get_filenames(mel_dir)
    files = my_dataset.shuffle(files)

    train_test_split = config['data']['train_test_split']
    split_index = int(len(files)*train_test_split)
    train_files = files[:split_index]
    test_files = files[split_index:]

    print(len(train_files))
    print(len(test_files))

    train_dataset = my_dataset.MyDataset(config, train_files)
    test_dataset = my_dataset.MyDataset(config, test_files)

    batch_size = 2

    train_loader, test_loader = my_dataset.make_variable_dataloader(train_dataset,
                                                                    test_dataset,
                                                                    batch_size = batch_size)


    print("Making model")
    model = classifiers.Emotion_Classifier(input_size, hidden_size,
                     num_layers = num_layers, num_classes = num_classes, bi = True)
    optimiser = optim.Adam(model.parameters(), lr=0.0001, weight_decay = 0.000001)
    loss_fn = nn.CrossEntropyLoss()

    print("Running training")
    train_model(model, optimiser, train_loader, test_loader, loss_fn,
                epochs = n_epochs, var_len_data = True)
