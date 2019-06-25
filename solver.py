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


class Solver(object):

    def __init__(self, train_loader, test_loader, model_name, config):

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config

        # These are the INITIAL lr's. They are updated within the optimizers
        # over the training iterations
        self.g_lr = config['optimizer']['g_lr']
        self.d_lr = config['optimizer']['d_lr']
        self.emo_lr = config['optimizer']['emo_cls_lr']

        self.lambda_gp = config['loss']['lambda_gp']
        self.lambda_g_emo_cls = config['loss']['lambda_g_emo_cls']
        self.lambda_cycle = config['loss']['lambda_cycle']
        self.lambda_id = config['loss']['lambda_id']

        self.batch_size = config['model']['batch_size']

        self.num_iters = config['loss']['num_iters']
        self.num_iters_decay = config['loss']['num_iters_decay']
        self.resume_iters = config['loss']['resume_iters']

        # Number of D/emo_cls updates for each G update
        self.d_to_g_ratio = config['loss']['d_to_g_ratio']

        self.use_tensorboard = config['logs']['tensorboard']
        self.log_every = config['logs']['log_every']

        self.sample_dir = config['logs']['sample_dir']
        self.sample_every = config['logs']['sample_every']

        self.model_save_dir = config['logs']['model_save_dir']
        self.model_save_every = config['logs']['model_save_every']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.use_tensorboard:
            self.logger = Logger(config['logs']['log_dir'])

        self.model_name = model_name
        self.model = model.StarGAN_emo_VC1(config, model_name)

        if self.resume_iters != 1:
            self.load_checkpoint()

    def load_checkpoint(self):

        path = os.path.join(self.model_save_dir, self.model_name)
        self.model.load(path, self.resume_iters)


    def train(self):
        '''
        Main training loop
        '''

        start_iter = resume_iters # == 1 if new model
        self.update_lr(start_iter)

        norm = Normalizer() #-------- ;;;WHAT DO I DO HERE ---------#
        data_iter = iter(self.data_loader)

        start_time = datetime.now()

        # main training loop
        for i in range(start_iter, num_iters+1):

            self.model.to_device(device = self.device)
            self.model.set_train_mode()

            # Get data from data loader
            try:
                x_real, x_lens, labels = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_real, x_lens, labels = next(data_iter)

            x_real = x_real.to(device = self.device)
            emo_labels = labels[:,0].to(device = self.device)
            spk_labels = labels[:,1].to(device = self.device)

            # Generate target domain labels randomly.
            num_emos = 4
            emo_targets = self.make_random_labels(num_emos, emo_labels.size(0))
            emo_targets = emo_targets.to(device = self.device)

            #############################################################
            #                    TRAIN CLASSIFIERS                      #
            #############################################################
            self.model.reset_grad()
            ce_loss_fn = nn.CrossEntropyLoss()

            # Train with x_real
            preds_emo_real = self.model.emo_cls(x_real)

            c_emo_real_loss = ce_loss_fn(preds_emo_real, emo_labels)
            c_emo_real_loss.backwards()
            self.model.emo_cls_optimizer.step()

            # repeat above for other classifiers when implemented

            #############################################################
            #                    TRAIN DISCRIMINATOR                    #
            #############################################################
            self.model.reset_grad()

            # Get results for x_fake
            x_fake = self.model.G(x_real, emo_targets)

            # Get real/fake predictions
            d_preds_real = self.model.D(x_real, emo_labels)
            d_pred_fake = self.model.D(x_fake.detach(), emo_targets)

            #Calculate loss
            grad_penalty = self.gradient_penalty(x_real, x_fake, emo_targets) # detach()?

            d_loss = -d_preds_real.mean() + d_pred_fake.mean() + \
                     self.lambda_gp * grad_penalty

            d_loss.backwards()
            self.model.d_optimizer.step()

            #############################################################
            #                      TRAIN GENERATOR                      #
            #############################################################
            if i % self.d_to_g_ratio == 0:

                self.model.reset_grad()

                x_fake = self.model.G(x_real, emo_targets)
                x_cycle = self.model.G(x_fake, emo_labels)
                x_id = self.model.G(x_real, emo_labels)
                d_preds_for_g = self.model.D(x_fake, emo_targets)
                preds_emo_fake = self.model.emo_cls(x_fake)

                x_cycle = self.make_equal_length(x_cycle, x_real)
                x_id = self.make_equal_length(x_id, x_real)

                l1_loss_fn = nn.L1Loss()

                loss_g_fake = - d_preds_for_g.mean()
                loss_cycle = l1_loss_fn(x_cycle, x_real)
                loss_id = l1_loss_fn(x_id, x_real)
                loss_g_emo_cls = ce_loss_fn(preds_emo_fake, emo_targets)

                g_loss = loss_g_fake + self.lambda_cycle * loss_cycle + \
                                       self.lambda_id * loss_id + \
                                       self.lambda_g_emo_cls * loss_g_emo_cls + \
                                       self.lambda_gp * grad_penalty


                g_loss.backwards()
                self.model.g_optimizer.step()

            #############################################################
            #                  PRINTING/LOGGING/SAVING                  #
            #############################################################
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

                elapsed = datetime.now() - start_time

                str = '{} elapsed. Iteration {:04} complete'.format(elapsed, i)
                for name, val in loss.items():
                    str += ", {} = {:.4f}".format(name, val)

                if self.tensorboard:
                    for tag, value in loss.items():
                        self.logger.scalar_summary(tag, value, i)

            # save checkpoint
            if i % self.model_save_every == 0:
                self.model.save(save_dir = self.model_save_dir, iter = i)

            # generate example samples from test set ;;; needs doing
            if i % self.sample_every == 0:
                pass

            # update learning rates
            self.update_lr(i)

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
        out_src = self.D(x_hat, targets)


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

    def make_equal_length(x_out, x_real):
        ''' Needs implementing'''

        return x_out




if __name__ == '__main__':

    import yaml
    import data_preprocessing
    import data_loader

    config = yaml.load(open('./config.yaml', 'r'))

    s = Solver(1,1, 'DebugModel', config)

    names, mels, labels, speakes, dims, dims_dis = data_preprocessing.load_session_data(1)

    print("Number mels: ", len(mels))

    mels = [m.t() for m in mels]

    print(mels[0].size())
    batch_size = 16
    # mels = torch.stack(mels)
    # labels = torch.Tensor(labels)
    train_loader, test_loader = data_loader.make_variable_dataloader(mels, labels,
                                                    batch_size = batch_size)

    data_iter = iter(train_loader)

    x, y = next(data_iter)

    x_lens = x[1]
    x = x[0]
    print(x.size(), x_lens.size(), y.size())

    targets = s.make_random_labels(4, batch_size)
    targets_one_hot = F.one_hot(targets, num_classes = 4).float()

    # out = s.model.G(input, targets)
    g_out = s.model.G(x, targets_one_hot)
    print(g_out.size())
    # WHY DIFFERNT LENGTH OUTPUT????
    # out = s.model.emo_cls(x[0], x[1])

    # print(out.size())




    # t = torch.Tensor([1,2,3,4,5,6,7,8,9])
    # rand_idx = torch.randperm(t.size(0))
    # targets = t[rand_idx]
    # print(targets)
    # print(rand_idx)
