'''
solver.py
Author: Max Elliott

Solver class for training new models or previously saved models from a
checkpoint.

Structure inspired by hujinsen.
'''
import os
import random
import numpy as np
import copy
import time
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.nn.functional as F

import audio_utils
import model
# import data_loader
from logger import Logger
from sample_set import Sample_Set

import sklearn
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score


class Solver(object):

    def __init__(self, train_loader, test_loader, model_name, config):

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.sample_set = Sample_Set(config)
        self.config = config

        # These are the INITIAL lr's. They are updated within the optimizers
        # over the training iterations
        self.g_lr = config['optimizer']['g_lr']
        self.d_lr = config['optimizer']['d_lr']
        self.emo_lr = config['optimizer']['emo_cls_lr']
        self.speaker_lr = config['optimizer']['speaker_cls_lr']
        self.dim_lr = config['optimizer']['dim_cls_lr']

        self.lambda_gp = config['loss']['lambda_gp']
        self.lambda_g_emo_cls = config['loss']['lambda_g_emo_cls']
        self.lambda_cycle = config['loss']['lambda_cycle']
        self.lambda_id = config['loss']['lambda_id']
        self.lambda_g_spk_cls = config['loss']['lambda_g_spk_cls']
        self.lambda_g_dim_cls = config['loss']['lambda_g_dim_cls']

        self.use_speaker = config['model']['use_speaker']
        self.use_dimension = config['model']['use_dimension']

        self.batch_size = config['model']['batch_size']

        self.num_iters = config['loss']['num_iters']
        self.num_iters_decay = config['loss']['num_iters_decay']
        self.resume_iters = config['loss']['resume_iters']
        self.current_iter = self.resume_iters

        # Number of D/emo_cls updates for each G update
        self.d_to_g_ratio = config['loss']['d_to_g_ratio']

        self.use_tensorboard = config['logs']['use_tensorboard']
        self.log_every = config['logs']['log_every']

        self.sample_dir = config['logs']['sample_dir']
        self.sample_every = config['logs']['sample_every']

        self.model_save_dir = config['logs']['model_save_dir']
        self.model_save_every = config['logs']['model_save_every']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.use_tensorboard:
            self.logger = Logger(config['logs']['log_dir'], model_name)

        self.model_name = model_name
        self.model = model.StarGAN_emo_VC1(config, model_name)

        if self.resume_iters != 0:
            self.load_checkpoint()

    def load_checkpoint(self):

        path = os.path.join(self.model_save_dir, self.model_name)
        self.model.load(path, self.resume_iters)
        self.model_name = self.model.name


    def train(self):
        '''
        Main training loop
        '''
        print('################ BEGIN TRAINING LOOP ################')

        start_iter = self.resume_iters + 1 # == 1 if new model

        self.update_lr(start_iter)

        # norm = Normalizer() #-------- ;;;WHAT DO I DO HERE ---------#
        data_iter = iter(self.train_loader)

        start_time = datetime.now()
        print('Started at {}'.format(start_time))

        # main training loop
        for i in range(start_iter, self.num_iters+1):

            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Loop {:02} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~".format(i))
            print("Iteration {:02} lr = {:.6f}".format(i, self.model.d_optimizer.param_groups[0]['lr']))
            self.model.to_device(device = self.device)
            self.model.set_train_mode()

            self.current_iter = i

            # Get data from data loader
            print('Getting mini-batch.')
            try:
                x, labels = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x, labels = next(data_iter)


            x_real = x[0].to(device = self.device).unsqueeze(1)
            x_lens = x[1].to(device = self.device)

            emo_labels = labels[:,0].to(device = self.device)
            spk_labels = labels[:,1].to(device = self.device)
            # ;;;;;;; GET DIM LABELS

            # Generate target domain labels randomly.
            num_emos = 4
            emo_targets = self.make_random_labels(num_emos, emo_labels.size(0))
            emo_targets = emo_targets.to(device = self.device)

            # one-hot versions of labels
            emo_labels_ones = F.one_hot(emo_labels, num_classes = 4).float().to(device = self.device)
            emo_targets_ones = F.one_hot(emo_targets, num_classes = 4).float().to(device = self.device)

            #############################################################
            #                    TRAIN CLASSIFIERS                      #
            #############################################################
            print('Training Classifiers...')
            self.model.reset_grad()
            ce_loss_fn = nn.CrossEntropyLoss()

            # Train with x_real
            preds_emo_real = self.model.emo_cls(x_real, x_lens)

            c_emo_real_loss = ce_loss_fn(preds_emo_real, emo_labels)

            c_emo_real_loss.backward()
            self.model.emo_cls_optimizer.step()

            if self.model.use_speaker:
                self.model.reset_grad()

                # Train with x_real
                preds_speaker_real = self.model.speaker_cls(x_real, x_lens)

                c_speaker_real_loss = ce_loss_fn(preds_speaker_real, spk_labels)

                c_speaker_real_loss.backward()
                self.model.speaker_cls_optimizer.step()

            if self.model.use_dimension:
                self.model.reset_grad()

                # Train with x_real
                preds_dimension_real = self.model.dimension_cls(x_real, x_lens)

                #;;; DO FOR MULTILABEL
                c_dimension_real_loss = ce_loss_fn(preds_dimension_real, dim_labels)

                c_speaker_real_loss.backward()
                self.model.speaker_cls_optimizer.step()


            # repeat above for other classifiers when implemented

            #############################################################
            #                    TRAIN DISCRIMINATOR                    #
            #############################################################
            print('Training Discriminator...')
            self.model.reset_grad()

            # Get results for x_fake
            x_fake = self.model.G(x_real, emo_targets_ones)
            # ;;; GET NEW X_LENS HERE

            # Get real/fake predictions
            d_preds_real = self.model.D(x_real, emo_labels_ones)
            d_preds_fake = self.model.D(x_fake.detach(), emo_targets_ones)

            #Calculate loss
            grad_penalty = self.gradient_penalty(x_real, x_fake, emo_targets_ones) # detach(), one hots?

            d_loss = -d_preds_real.mean() + d_preds_fake.mean() + \
                     self.lambda_gp * grad_penalty

            d_loss.backward()
            self.model.d_optimizer.step()

            #############################################################
            #                      TRAIN GENERATOR                      #
            #############################################################
            if i % self.d_to_g_ratio == 0:
                print('Training Generator...')

                self.model.reset_grad()

                x_fake = self.model.G(x_real, emo_targets_ones)
                # ;;; GET NEW X_LENS HERE
                x_fake_lens = x_lens

                x_cycle = self.model.G(x_fake, emo_labels_ones)
                x_id = self.model.G(x_real, emo_labels_ones)
                d_preds_for_g = self.model.D(x_fake, emo_targets_ones)
                preds_emo_fake = self.model.emo_cls(x_fake, x_fake_lens)


                x_cycle = self.make_equal_length(x_cycle, x_real)
                x_id = self.make_equal_length(x_id, x_real)

                l1_loss_fn = nn.L1Loss()

                loss_g_fake = - d_preds_for_g.mean()
                loss_cycle = l1_loss_fn(x_cycle, x_real)
                loss_id = l1_loss_fn(x_id, x_real)
                loss_g_emo_cls = ce_loss_fn(preds_emo_fake, emo_targets)

                g_loss = loss_g_fake + self.lambda_cycle * loss_cycle + \
                                       self.lambda_id * loss_id + \
                                       self.lambda_g_emo_cls * loss_g_emo_cls# + \
                                       # self.lambda_gp * grad_penalty
                if self.use_speaker:

                    preds_spk_fake = self.model.speaker_cls(x_fake, x_fake_lens)
                    loss_g_spk_cls = ce_loss_fn(x_fake, x_fake_lens)
                    g_loss += self.lambda_g_spk_cls * loss_g_spk_cls

                if self.use_dimension:

                    preds_dim_fake = self.model.speaker_cls(x_fake, x_fake_lens)
                    loss_g_dim_cls = ce_loss_fn(x_fake, x_fake_lens)
                    g_loss += self.lambda_g_dim_cls * loss_g_dim_cls





                g_loss.backward()
                self.model.g_optimizer.step()
            else:
                print("No Generator update this iteration.")

            #############################################################
            #                  PRINTING/LOGGING/SAVING                  #
            #############################################################
            elapsed = datetime.now() - start_time
            print('{} elapsed. Iteration {:04} complete'.format(elapsed, i))

            if i % self.log_every == 0:
                loss = {}
                loss['C/emo_real_loss'] = c_emo_real_loss.item()
                loss['D/total_loss'] = d_loss.item()
                loss['G/total_loss'] = g_loss.item()
                loss['G/emo_loss'] = loss_g_emo_cls.item()
                loss['gradient_penalty'] = grad_penalty.item()
                loss['loss_cycle'] = loss_cycle.item()
                loss['loss_id'] = loss_id.item()
                loss['D/preds_real'] = d_preds_real.mean().item()
                loss['D/preds_fake'] = d_preds_fake.mean().item()

                for name, val in loss.items():
                    print("{:20} = {:.4f}".format(name, val))

                if self.use_tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i)
            else:
                print("No log output this iteration.")

            # save checkpoint
            if i % self.model_save_every == 0:
                self.model.save(save_dir = self.model_save_dir, iter = self.current_iter)
            else:
                print("No model saved this iteration.")

            # generate example samples from test set ;;; needs doing
            if i % self.sample_every == 0:
                self.test()
                # filler_var = 1
                # print("Will sample here.")

            # update learning rates
            self.update_lr(i)


    def test(self):

        # test_iter = iter(self.test_loader)
        print("Sampling with generator...")
        self.model.set_eval_mode()

        fake_preds = torch.rand(0).to(device = self.device, dtype = torch.long)
        id_preds = torch.rand(0).to(device = self.device, dtype = torch.long)
        cycle_preds = torch.rand(0).to(device = self.device, dtype = torch.long)

        total_labels = torch.rand(0).to(device = self.device, dtype = torch.long)
        total_targets = torch.rand(0).to(device = self.device, dtype = torch.long)

        for i, (x, labels) in enumerate(self.test_loader):

            x_real = x[0].to(device = self.device)
            x_lens = x[1].to(device = self.device)

            x_real = x_real.unsqueeze(1)

            emo_labels = labels[:,0].to(device = self.device)
            spk_labels = labels[:,1].to(device = self.device)

            # Generate target domain labels randomly.
            num_emos = 4
            emo_targets = self.make_random_labels(num_emos, emo_labels.size(0))
            emo_targets = emo_targets.to(device = self.device)

            # one-hot versions of labels
            emo_labels_ones = F.one_hot(emo_labels, num_classes = 4).float().to(device = self.device)
            emo_targets_ones = F.one_hot(emo_targets, num_classes = 4).float().to(device = self.device)

            with torch.no_grad():

                x_fake = self.model.G(x_real, emo_targets_ones)
                x_id = self.model.G(x_real, emo_labels_ones)
                x_cycle = self.model.G(x_fake, emo_labels_ones)

                new_lens = x_lens #;;; needs doing

                c_emo_fake = self.model.emo_cls(x_fake, x_lens)
                c_emo_id = self.model.emo_cls(x_id, x_lens)
                c_emo_cycle = self.model.emo_cls(x_cycle, x_lens)

                c_emo_fake = torch.max(c_emo_fake, dim = 1)[1]
                c_emo_id = torch.max(c_emo_id, dim = 1)[1]
                c_emo_cycle = torch.max(c_emo_cycle, dim = 1)[1]

                # D as well

                fake_preds = torch.cat((fake_preds, c_emo_fake), dim=0)
                id_preds = torch.cat((id_preds, c_emo_id), dim=0)
                cycle_preds = torch.cat((cycle_preds, c_emo_cycle), dim=0)

                total_labels = torch.cat((total_labels, emo_labels), dim=0)
                total_targets = torch.cat((total_targets, emo_targets), dim=0)

        accuracy_fake = accuracy_score(total_targets, fake_preds)
        accuracy_id = accuracy_score(total_labels, id_preds)
        accuracy_cycle = accuracy_score(total_labels, cycle_preds)

        l = ["Accuracy_fake", "Accuracy_id", "Accuracy_cycle"]

        print('{:20} = {:.3f}'.format(l[0], accuracy_fake))
        print('{:20} = {:.3f}'.format(l[1], accuracy_id))
        print('{:20} = {:.3f}'.format(l[2], accuracy_cycle))

        if self.use_tensorboard:
            self.logger.scalar_summary("test_accuracy_fake", accuracy_fake, self.current_iter)
            self.logger.scalar_summary("test_accuracy_id", accuracy_id, self.current_iter)
            self.logger.scalar_summary("test_accuracy_cycle", accuracy_cycle, self.current_iter)

    def sample_at_training(self):
        '''
        Passes each performance sample through G for every target emotion. They
        are saved to 'config(sample_dir)/model_name/filename-<emo>to<trg>.png'
        '''

        print("Saving samples...")

        self.model.to_device(device = self.device)
        self.model.set_eval_mode()

        # Make one-hot vector for each emotion category
        emo_labels = torch.Tensor([0,1,2,3]).long()
        emo_targets = F.one_hot(emo_labels, num_classes = 4).float().to(device = self.device)

        for tag, val in self.sample_set.get_set().items():
            # tag is filename, val is [mel, labels, spec]
            mel = val[0].unsqueeze(0).unsqueeze(0).to(device = self.device)
            labels = val[1]

            with torch.no_grad():
                # print(emo_targets)
                for i in range (0, emo_targets.size(0)):

                    fake = self.model.G(mel, emo_targets[i].unsqueeze(0))

                    filename_png =  tag[0:-4] + "_" + str(int(labels[0].item())) + "to" + \
                                str(emo_labels[i].item()) + '_i=' +\
                                str(self.current_iter) + ".png"

                    filename_npy =  tag[0:-4] + "_" + str(int(labels[0].item())) + "to" + \
                                str(emo_labels[i].item()) + '_i=' +\
                                str(self.current_iter) + ".npy"

                    fake = fake.squeeze()
                    audio_utils.save_spec_plot(fake.t(), self.model_name, filename)
                    audio_utils.save_spec(fake.t(), self.model_name, filename)








    def update_lr(self, i):
        """Decay learning rates of the generator and discriminator and classifier."""
        if self.num_iters - self.num_iters_decay < i:
            decay_delta_d = self.d_lr / self.num_iters_decay
            decay_delta_g = self.g_lr / self.num_iters_decay
            decay_delta_emo = self.emo_lr / self.num_iters_decay

            decay_start = self.num_iters - self.num_iters_decay
            decay_iter = i - decay_start

            d_lr = self.d_lr - decay_iter * decay_delta_d
            g_lr = self.g_lr - decay_iter * decay_delta_g
            emo_lr = self.emo_lr - decay_iter * decay_delta_emo

            for param_group in self.model.g_optimizer.param_groups:
                param_group['lr'] = g_lr
            for param_group in self.model.d_optimizer.param_groups:
                param_group['lr'] = d_lr
            for param_group in self.model.emo_cls_optimizer.param_groups:
                param_group['lr'] = emo_lr

    def make_random_labels(self, num_domains, num_labels):
        '''
        Creates random labels for generator.
        num_domains: number of unique labels
        num_labels: total number of labels to generate
        '''
        domain_list = np.arange(0, num_domains)
        # print(domain_list)
        labels = torch.zeros((num_labels))
        for i in range(0, num_labels):
            labels[i] = random.choice(domain_list).item()

        return labels.long()

    def gradient_penalty(self, x_real, x_fake, targets):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2.
        Taken from https://github.com/hujinsen/pytorch-StarGAN-VC"""
        # Compute loss for gradient penalty.
        alpha = torch.rand(x_real.size(0), 1, 1, 1).to(self.device)
        x_hat = (alpha * x_real.data + (1 - alpha) * x_fake.data).requires_grad_(True)
        # print("x_hat size: ", x_hat.size())
        out_src = self.model.D(x_hat, targets)

        weight = torch.ones(out_src.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=out_src,
                                   inputs=x_hat,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1)+ 1e-12)
        return torch.mean((dydx_l2norm-1)**2)

    def make_equal_length(self, x_out, x_real):
        ''' Needs implementing'''

        return x_out




if __name__ == '__main__':

    import yaml
    import data_preprocessing
    import data_loader

    # from os.path import exists
    # from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
    # platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag())
    # cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
    # accelerator = cuda_output[0] if exists('/dev/nvidia0') else 'cpu'

    # LOAD IN DATA TO DATA_LOADERS
    config = yaml.load(open('./config.yaml', 'r'))

    names, mels, labels = data_preprocessing.load_session_data(1)

    print("Number mels: ", len(mels), len(labels))

    mels = [m.t() for m in mels]


    mels, labels = data_loader.crop_sequences(mels, labels, 300)

    print("Number cropped mels: ", len(mels), len(labels))

    # mels = torch.stack(mels)
    # labels = torch.stack(labels)
    # print(labels.size())
    # print(mels.size())

    # print(mels.size())
    batch_size = config['model']['batch_size']
    train_loader, test_loader = data_loader.make_variable_dataloader(mels, labels,
                                batch_size = batch_size,
                                train_test_split = config['model']['train_test_split'])

    # print("Train loader size = {}".format(train_loader.__len__()))
    # print("Test loader size = {}".format(test_loader.__len__()))



    s = Solver(train_loader, test_loader, 'DebugModel', config)
    s.train()

    # for tag, val in s.sample_set.get_set().items():
    #     print(tag, val[0].size())
    #
    # s.sample_at_training()


    # data_iter = iter(train_loader)
    #
    # x, y = next(data_iter)
    #
    # x_lens = x[1]
    # x = x[0]
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
    # print('c_out = ',out.size())
