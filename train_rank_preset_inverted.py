# train.py

import torch
import torchvision
import torchvision.utils as vutils
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim, compare_psnr
import time
import os
import numpy as np
import plugins
from losses import RankOrderLoss
from evaluate import Logits_Classification

class Trainer():
    def __init__(self, args, modelD, modelG, Encoder, criterion, prevD, prevG):

        self.args = args
        self.modelD = [modelD for i in range(2)]
        self.modelG = [modelG for i in range(2)]
        self.Encoder = Encoder
        self.prevD = prevD
        self.prevG = prevG
        self.criterion = criterion
        self.cuda = args.cuda
        self.device = torch.device("cuda" if (self.cuda and torch.cuda.is_available()) else "cpu")
        self.logits_loss = RankOrderLoss(self.device)
        self.logits_eval = Logits_Classification(threshold=0.5)
        self.plot_update_interval = args.plot_update_interval

        self.port = args.port
        self.env = args.env
        self.result_path = args.result_path
        self.save_path = args.save
        self.log_path = args.logs
        self.dataset_fraction = args.dataset_fraction
        self.len_dataset = 0
        self.use_encoder = args.use_encoder

        self.stage_epochs = args.stage_epochs
        self.start_stage = args.start_stage
        self.nchannels = args.nchannels
        self.batch_size = args.batch_size
        self.resolution_high = args.resolution_high
        self.resolution_wide = args.resolution_wide
        self.nz = args.nz
        self.gp = args.gp
        self.gp_lambda = args.gp_lambda
        self.scheduler_patience = args.scheduler_patience
        self.scheduler_maxlen = args.scheduler_maxlen

        self.weight_gan_final = args.weight_gan_final
        self.weight_vae_init = args.weight_vae_init
        self.weight_kld = args.weight_kld
        self.margin = args.margin
        self.num_stages = args.num_stages
        self.nranks = args.nranks

        self.lr_vae = args.learning_rate_vae
        self.lr_dis = args.learning_rate_dis
        self.lr_gen = args.learning_rate_gen
        self.lr_decay = args.learning_rate_decay
        self.momentum = args.momentum
        self.adam_beta1 = args.adam_beta1
        self.adam_beta2 = args.adam_beta2
        self.weight_decay = args.weight_decay
        self.optim_method = args.optim_method
        self.vae_loss_type = args.vae_loss_type

        # for classification
        self.fixed_noise = torch.FloatTensor(self.batch_size, self.nz).normal_(0, 1).to(self.device)#, volatile=True)
        self.epsilon = torch.randn(self.batch_size, self.nz).to(self.device)
        self.target_real = torch.ones(self.batch_size, self.nranks-1).to(self.device)
        self.target_G2 = torch.cat((torch.ones(self.batch_size, self.nranks-2), torch.zeros(self.batch_size, self.nranks-2)), 1).to(self.device)
        self.target_G1 = torch.zeros(self.batch_size, self.nranks-1).to(self.device)
        self.sigmoid = torch.sigmoid

        # Initialize optimizer
        self.optimizerE = self.initialize_optimizer(self.Encoder, lr=self.lr_vae, optim_method='Adam')
        self.optimizerG = self.initialize_optimizer(self.modelG[0], lr=self.lr_vae, optim_method='Adam')
        self.optimizerD = self.initialize_optimizer(self.modelD[0], lr=self.lr_dis, optim_method='Adam', weight_decay=0.01*self.lr_dis)
        # self.schedulerE = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerE, factor=self.lr_decay, patience=self.scheduler_patience, min_lr=1e-3*self.lr_vae)
        # self.schedulerG = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, factor=self.lr_decay, patience=self.scheduler_patience, min_lr=1e-3*self.lr_vae)
        # self.schedulerD = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, factor=self.lr_decay, patience=self.scheduler_patience, min_lr=1e-3*self.lr_vae)
        # Automatic scheduler
        self.schedulerE = plugins.AutomaticLRScheduler(self.optimizerE, maxlen=self.scheduler_maxlen, factor=self.lr_decay, patience=self.scheduler_patience)
        self.schedulerG = plugins.AutomaticLRScheduler(self.optimizerG, maxlen=self.scheduler_maxlen, factor=self.lr_decay, patience=self.scheduler_patience)
        self.schedulerD = plugins.AutomaticLRScheduler(self.optimizerD, maxlen=self.scheduler_maxlen, factor=self.lr_decay, patience=self.scheduler_patience)

        # logging training
        self.log_loss_train = plugins.Logger(args.logs, 'TrainLogger.txt')
        self.params_loss_train = ['Loss_D0', 'Loss_G0', 'MSE', 'KLD', 'D0_I', 'D1_I', 'D0_G2', 'D1_G2', 'D0_G1', 'D1_G1', 'Acc_Real', 'Acc_G2', 'Acc_G1']
        self.log_loss_train.register(self.params_loss_train)

        # monitor training
        self.monitor_train = plugins.Monitor()
        self.params_monitor_train = ['Loss_D0', 'Loss_G0', 'MSE', 'KLD', 'D0_I', 'D1_I', 'D0_G2', 'D1_G2', 'D0_G1', 'D1_G1', 'Acc_Real', 'Acc_G2', 'Acc_G1']
        self.monitor_train.register(self.params_monitor_train)

        # Define visualizer plot type for given dataset
        if args.net_type == 'gmm':
            self.plot_update_interval = 300
            if self.args.gmm_dim == 1:
                output_dtype, output_vtype = 'vector', 'histogram'
            elif self.args.gmm_dim == 2:
                output_dtype, output_vtype = 'vector', 'scatter'
        else:
            output_dtype, output_vtype = 'images', 'images'
            self.fixed_noise = self.fixed_noise.unsqueeze(-1).unsqueeze(-1)

        # visualize training
        self.visualizer_train = plugins.Visualizer(port=self.port, env=self.env, title='Train')
        self.params_visualizer_train = {
        'Loss_D0':{'dtype':'scalar', 'vtype':'plot', 'win': 'loss_gan', 'layout': {'windows': ['Loss_D0', 'Loss_G0'], 'id': 0}},
        'Loss_G0':{'dtype':'scalar','vtype':'plot', 'win': 'loss_gan', 'layout': {'windows': ['Loss_D0', 'Loss_G0'], 'id': 1}},
        'MSE':{'dtype':'scalar','vtype':'plot', 'win': 'enc_losses', 'layout': {'windows': ['MSE', 'KLD', 'Corr_Loss'], 'id': 0}},
        'KLD':{'dtype':'scalar','vtype':'plot', 'win': 'enc_losses', 'layout': {'windows': ['MSE', 'KLD', 'Corr_Loss'], 'id': 1}},
        'Mean':{'dtype':'scalar','vtype':'plot', 'win': 'norm_params', 'layout': {'windows': ['Mean', 'Sigma'], 'id': 0}},
        'Sigma':{'dtype':'scalar','vtype':'plot', 'win': 'norm_params', 'layout': {'windows': ['Mean', 'Sigma'], 'id': 1}},
        'D0_I':{'dtype':'scalar','vtype':'plot', 'win': 'loss_D0', 'layout': {'windows': ['D0_I', 'D0_G2', 'D0_G1'], 'id': 0}},
        'D0_G2':{'dtype':'scalar','vtype':'plot', 'win': 'loss_D0', 'layout': {'windows': ['D0_I', 'D0_G2', 'D0_G1'], 'id': 1}},
        'D0_G1':{'dtype':'scalar','vtype':'plot', 'win': 'loss_D0', 'layout': {'windows': ['D0_I', 'D0_G2', 'D0_G1'], 'id': 2}},
        'D1_I':{'dtype':'scalar','vtype':'plot', 'win': 'loss_D0', 'layout': {'windows': ['D1_I', 'D1_G2', 'D1_G1'], 'id': 0}},
        'D1_G2':{'dtype':'scalar','vtype':'plot', 'win': 'loss_D0', 'layout': {'windows': ['D1_I', 'D1_G2', 'D1_G1'], 'id': 1}},
        'D1_G1':{'dtype':'scalar','vtype':'plot', 'win': 'loss_D0', 'layout': {'windows': ['D1_I', 'D1_G2', 'D1_G1'], 'id': 2}},

        'Acc_Real':{'dtype':'scalar','vtype':'plot', 'win': 'acc', 'layout': {'windows': ['Acc_Real', 'Acc_G2', 'Acc_G1'], 'id': 0}},
        'Acc_G2':{'dtype':'scalar','vtype':'plot', 'win': 'acc', 'layout': {'windows': ['Acc_Real', 'Acc_G2', 'Acc_G1'], 'id': 1}},
        'Acc_G1':{'dtype':'scalar','vtype':'plot', 'win': 'acc', 'layout': {'windows': ['Acc_Real', 'Acc_G2', 'Acc_G1'], 'id': 2}},

        'Learning_Rate_E':{'dtype':'scalar','vtype':'plot', 'win': 'lr_E'},
        'Learning_Rate_G':{'dtype':'scalar','vtype':'plot', 'win': 'lr_G'},
        'Learning_Rate_D':{'dtype':'scalar','vtype':'plot', 'win': 'lr_D'},

        'Real': {'dtype': output_dtype, 'vtype': output_vtype, 'win': 'real'},
        'Fakes_Encoder': {'dtype': output_dtype, 'vtype': output_vtype, 'win': 'fakes_enc'},
        'Fakes_Normal': {'dtype': output_dtype, 'vtype': output_vtype, 'win': 'fakes_normal'},
        'Fakes_Previous': {'dtype': output_dtype, 'vtype': output_vtype, 'win': 'fakes_prev'},
        }
        self.visualizer_train.register(self.params_visualizer_train)

        # display training progress
        self.print_train = '[%d/%d][%d/%d] '
        for item in self.params_loss_train:
            self.print_train = self.print_train + item + " %.3f "

        self.giterations = 0
        self.d_iter_init = args.d_iter
        self.d_iter = self.d_iter_init
        self.g_iter_init = args.g_iter
        self.g_iter = self.g_iter_init
        print('Discriminator:', self.modelD[0])
        print('Generator:', self.modelG[0])
        print('Encoder:', self.Encoder)

        # define a zero tensor
        self.t_zero = torch.zeros(1)
        self.add_noise = args.add_noise
        self.noise_var = args.noise_var

    def initialize_optimizer(self, model, lr, optim_method='RMSprop', weight_decay=None):
        if weight_decay is None:
            weight_decay = self.weight_decay
        if optim_method == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(self.adam_beta1, self.adam_beta2), weight_decay=weight_decay)
        elif optim_method == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=self.momentum, weight_decay=weight_decay)
        elif optim_method == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=self.momentum, weight_decay=weight_decay)
        else:
            raise(Exception("Unknown Optimization Method"))
        return optimizer

    def model_train(self):
        self.modelD[0].train()
        self.modelG[0].train()
        self.Encoder.train()

    def setup_stage(self, stage, dataloader=None):
        if stage == 0:
            self.weight_vae, self.weight_gan = self.weight_vae_init, 0
            self.optimizerE = self.initialize_optimizer(self.Encoder, lr=self.lr_vae, optim_method='Adam')   #Adam
            self.optimizerG = self.initialize_optimizer(self.modelG[0], lr=self.lr_vae, optim_method=self.optim_method) #RMSprop
            # self.schedulerE = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerE, factor=self.lr_decay, patience=self.scheduler_patience, min_lr=1e-3*self.lr_vae)
            # self.schedulerG = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, factor=self.lr_decay, patience=self.scheduler_patience, min_lr=1e-3*self.lr_gen)
            self.schedulerE = plugins.AutomaticLRScheduler(self.optimizerE, maxlen=self.scheduler_maxlen, factor=self.lr_decay, patience=self.scheduler_patience)
            self.schedulerG = plugins.AutomaticLRScheduler(self.optimizerG, maxlen=self.scheduler_maxlen, factor=self.lr_decay, patience=self.scheduler_patience)
        elif stage == 1:
            self.optimizerD = self.initialize_optimizer(self.modelD[0], lr=self.lr_dis, optim_method='Adam', weight_decay=0.01*self.lr_dis)
            self.schedulerD = plugins.AutomaticLRScheduler(self.optimizerD, maxlen=self.scheduler_maxlen, factor=self.lr_decay, patience=self.scheduler_patience)
            self.target_G1 = self.target_real.clone()
            self.target_G1[:, self.nranks-1-stage] = 0
            self.target_G2 = self.target_G1.clone()
        else:
            # if stage == 2 and ((not self.gp) or (self.gp and self.start_stage == stage)):
            if stage > self.start_stage:
                self.target_G1 = self.target_G2.clone()
                self.target_G2[:,self.nranks-1-stage] = 0
                self.optimizerD = self.initialize_optimizer(self.modelD[0], lr=self.lr_dis, optim_method=self.optim_method, weight_decay=0.01*self.lr_dis)
                self.optimizerG = self.initialize_optimizer(self.modelG[0], lr=self.lr_gen, optim_method=self.optim_method)
            elif stage == self.start_stage:
                self.target_G1 = self.target_real.clone()
                self.target_G1[:,-1] = 0
                self.target_G2 = self.target_G1.clone()
                self.target_G2[:,-stage] = 0

            # Initialize previous stage models and set them to eval mode
            if stage == self.start_stage and self.prevD is not None:
                print("Loading previous discriminator")
                self.modelD[1] = self.prevD
            else:
                self.modelD[1] = copy.deepcopy(self.modelD[0])
            self.prevD = None

            if stage == self.start_stage and self.prevG is not None:
                print("Loading previous generator")
                self.modelG[1] = self.prevG
            else:
                self.modelG[1] = copy.deepcopy(self.modelG[0])
            self.prevG = None

            # self.modelG[1].eval()
            # self.modelD[1].eval()
            self.Encoder.eval()
            for p in self.modelD[1].parameters():
                p.requires_grad = False
            for p in self.modelG[1].parameters():
                p.requires_grad = False


    def train(self, stage, epoch, dataloader):
        self.monitor_train.reset()
        data_iter = iter(dataloader)
        self.len_dataset = int(len(dataloader) * self.dataset_fraction)
        if epoch == 0:
            print("Length of Dataset: {}".format(self.len_dataset))

        if stage == 0:
        ############################
        # Train VAE
        ############################
            self.Encoder.train()
            self.modelG[0].train()
            i = 0
            while i < self.len_dataset:
                input = data_iter.next()[0]
                i += 1
                batch_size = input.size(0)
                input = input.to(self.device)

                # zero grad
                self.Encoder.zero_grad()
                self.modelG[0].zero_grad()

                # get latents
                self.epsilon.resize_(batch_size, self.nz).normal_(0, 1)
                noise_mu, noise_logvar = self.Encoder(input)
                noise_sigma = torch.exp(torch.mul(noise_logvar, 0.5))
                latents = noise_mu + torch.mul(noise_sigma, self.epsilon)
                while(len(latents.size()) < len(input.size())):
                    latents = latents.unsqueeze(-1)

                # compute vae loss
                fake = self.modelG[0](latents)
                loss_mse = self.criterion.diff_loss(fake, input, type=self.vae_loss_type)
                # loss_correlation, corr = self.criterion.correlation_loss(self.input, fake)
                loss_correlation = self.t_zero
                loss_kld = self.criterion.kl_divergence(noise_mu, noise_logvar)
                net_error = self.weight_vae * loss_mse + self.weight_kld * loss_kld
                # net_error = self.weight_kld * loss_kld + 1*loss_correlation
                try:
                    net_error.backward()
                except Exception as e:
                    print(e)
                self.optimizerE.step()
                self.optimizerG.step()
                self.schedulerE.step(loss_kld.item())
                self.schedulerG.step(loss_mse.item())

                # Bookkeeping
                losses_train = {}
                losses_train['Loss_D0'] = 0
                losses_train['Loss_G0'] = 0
                losses_train['MSE'] = loss_mse.item()
                losses_train['KLD'] = loss_kld.item()
                losses_train['D0_I'] = 0
                losses_train['D0_G2'] = 0
                losses_train['D0_G1'] = 0
                losses_train['D1_I'] = 0
                losses_train['D1_G2'] = 0
                losses_train['D1_G1'] = 0
                losses_train['Acc_Real'] = 0
                losses_train['Acc_G1'] = 0
                losses_train['Acc_G2'] = 0
                self.monitor_train.update(losses_train, batch_size)
                print('Stage %d: [%d/%d][%d/%d] Loss_D0: %.3f Loss_G0: %.3f MSE: %.3f KLD: %.3f D0_I %.3f D0_G2 %.3f D0_G1 %.3f D1_I %.3f D1_G2 %.3f D1_G1 %.3f Acc_Real %.1f Acc_G2 %.1f Acc_G1 %.1f'
                    % (stage, epoch, self.stage_epochs[stage], i, self.len_dataset, 0, 0, loss_mse.item(), loss_kld.item(), 0, 0, 0, 0, 0, 0, 0, 0, 0))

                if i % self.plot_update_interval == 0:
                    losses_train['Learning_Rate_E'] = self.optimizerE.param_groups[0]['lr']
                    losses_train['Learning_Rate_G'] = self.optimizerG.param_groups[0]['lr']
                    losses_train['Learning_Rate_D'] = self.lr_dis
                    losses_train['Real'] = input.cpu()
                    losses_train['Fakes_Encoder'] = fake.detach().cpu()
                    losses_train['Fakes_Previous'] = torch.zeros(fake.size()).detach().cpu()
                    losses_train['Fakes_Normal'] = self.modelG[0](self.fixed_noise).detach().cpu()
                    losses_train['Mean'] = noise_mu.mean().item()
                    losses_train['Sigma'] = noise_sigma.mean().item()

                    self.visualizer_train.update(losses_train)

                if i % 250 == 0:
                    try:
                        fake_normal_z = self.modelG[0](self.fixed_noise)
                        fake_encoder_z = self.modelG[0](latents)
                        if len(self.input.size()) < 3:
                            fig = plt.figure()
                            plt.hist(self.input.squeeze().data.cpu().numpy(), bins=60)
                            fig.savefig("{}/stage_{}_epoch_{}_real.png".format(self.save_path, stage, i), dpi=fig.dpi)
                            fig = plt.figure()
                            plt.hist(fake_encoder_z.squeeze().data.cpu().numpy(), bins=60)
                            fig.savefig("{}/stage_{}_epoch_{}_fake_enc.png".format(self.save_path, stage, i), dpi=fig.dpi)
                            fig = plt.figure()
                            plt.hist(fake_normal_z.squeeze().data.cpu().numpy(), bins=60)
                            fig.savefig("{}/stage_{}_epoch_{}_fake_norm.png".format(self.save_path, stage, i), dpi=fig.dpi)
                        else:
                            vutils.save_image(fake_normal_z.data, '%s/fake_samples_stage_%03d_normal_z.png' % (self.save_path, stage), normalize=True)
                            vutils.save_image(fake_encoder_z.data, '%s/fake_samples_stage_%03d_encoder_z.png' % (self.save_path, stage), normalize=True)
                            vutils.save_image(input, '%s/real_samples.png' % self.save_path, normalize=True)
                    except Exception as e:
                        print(e)
        elif stage == 1:
        ################################################
        # Train Discriminator till optimality
        ################################################
            # self.Encoder.eval()
            # self.modelG[0].eval()
            self.modelD[0].train()
            i = 0
            while i < self.len_dataset:
                input = data_iter.next()[0].to(self.device)
                i += 1
                batch_size = input.size(0)
                self.modelD[0].zero_grad()

                # train with real
                if self.add_noise:
                    self.epsilon.resize_(input.size()).normal_(0, self.noise_var)
                    dis_input = input + self.epsilon
                    out_D = self.modelD[0](dis_input)
                else:
                    out_D = self.modelD[0](input)
                loss_real = self.logits_loss(out_D, self.target_real[:batch_size])
                logits_D = self.sigmoid(out_D)
                acc_real = 100*self.logits_eval(logits_D, self.target_real[:batch_size])

                # train with fake
                self.epsilon.data.resize_(batch_size, self.nz).normal_(0, 1)
                noise_mu, noise_logvar = self.Encoder(input)
                noise_sigma = torch.exp(torch.mul(noise_logvar, 0.5))
                latents = noise_mu + torch.mul(noise_sigma, self.epsilon)
                latents = latents.detach()
                while(len(latents.size()) < len(input.size())):
                    latents = latents.unsqueeze(-1)

                fake = self.modelG[0](latents).detach()
                if self.add_noise:
                    self.epsilon.data.resize_(input.size()).normal_(0, self.noise_var)
                    dis_input = fake + self.epsilon
                    out_G1 = self.modelD[0](dis_input)
                else:
                    out_G1 = self.modelD[0](fake)
                loss_G1 = self.logits_loss(out_G1, self.target_G1[:batch_size])
                logits_G1 = self.sigmoid(out_G1)
                acc_G1 = 100*self.logits_eval(logits_G1, self.target_G1[:batch_size])

                net_error = loss_real + loss_G1
                net_error.backward()

                self.optimizerD.step()
                self.schedulerD.step(net_error.item())

                # Bookkeeping
                losses_train = {}
                losses_train['Loss_D0'] = net_error.item()
                losses_train['Loss_G0'] = 0
                losses_train['MSE'] = 0
                losses_train['KLD'] = 0
                losses_train['D0_I'] = out_D[:,0].median().item()
                losses_train['D0_G2'] = 0
                losses_train['D0_G1'] = out_G1[:,0].median().item()
                losses_train['D1_I'] = out_D[:,1].median().item()
                losses_train['D1_G2'] = 0
                losses_train['D1_G1'] = out_G1[:,1].median().item()
                losses_train['Acc_Real'] = acc_real
                losses_train['Acc_G1'] = acc_G1
                losses_train['Acc_G2'] = 0
                self.monitor_train.update(losses_train, batch_size)
                print('Stage %d: [%d/%d][%d/%d] Loss_D0: %.3f Loss_G0: %.3f MSE: %.3f KLD: %.3f D0_I %.3f D0_G2 %.3f D0_G1 %.3f D1_I %.3f D1_G2 %.3f D1_G1 %.3f Acc_Real %.1f Acc_G2 %.1f Acc_G1 %.1f'
                    % (stage, epoch, self.stage_epochs[stage], i, self.len_dataset, net_error.item(), 0, 0, 0, out_D[:,0].median().item(), 0, out_G1[:,0].median().item(), out_D[:,1].median().item(), 0, out_G1[:,1].median().item(), acc_real, 0, acc_G1))

                if i % self.plot_update_interval == 0:
                    losses_train['Learning_Rate_E'] = self.optimizerE.param_groups[0]['lr']
                    losses_train['Learning_Rate_G'] = self.optimizerG.param_groups[0]['lr']
                    losses_train['Learning_Rate_D'] = self.optimizerD.param_groups[0]['lr']
                    losses_train['Real'] = input.detach().cpu()
                    losses_train['Fakes_Encoder'] = fake.detach().cpu()
                    losses_train['Fakes_Previous'] = torch.zeros(fake.size()).detach().cpu()
                    losses_train['Fakes_Normal'] = self.modelG[0](self.fixed_noise).detach().cpu()
                    losses_train['Mean'] = noise_mu.mean().item()
                    losses_train['Sigma'] = noise_sigma.mean().item()
                    # losses_train['Corr_Output'] = torch.zeros(fake.size())
                    self.visualizer_train.update(losses_train)

        else:
            ############################
            # Train GAN
            ############################
            i = 0
            while i < self.len_dataset:
                ############################
                # Update Discriminator Network
                ############################
                self.modelD[0].train()
                # self.modelG[0].eval()
                # if epoch !=0 and self.giterations % (self.len_dataset/5) == 0:
                #     d_iterations = self.len_dataset/10
                # else:
                #     d_iterations = self.d_iter
                d_iterations = self.d_iter

                j=0
                lossD = 0
                lossG = 0
                while j < d_iterations and i < self.len_dataset:
                    lossG = 0
                    j += 1
                    i += 1

                    input = data_iter.next()[0].to(self.device)
                    batch_size = input.size(0)

                    self.modelD[0].zero_grad()
                    self.modelG[0].zero_grad()

                    # train with real
                    if self.add_noise:
                        self.epsilon.data.resize_(input.size()).normal_(0, self.noise_var)
                        dis_input = input + self.epsilon
                        out_D = self.modelD[0](dis_input)
                    else:
                        out_D = self.modelD[0](input)
                    loss_real = self.logits_loss(out_D, self.target_real[:batch_size])
                    logits_D = self.sigmoid(out_D)
                    acc_real = 100*self.logits_eval(logits_D, self.target_real[:batch_size])

                    # train with fake from G1
                    self.epsilon.data.resize_(batch_size, self.nz).normal_(0, 1)
                    if self.use_encoder:
                        noise_mu, noise_logvar = self.Encoder(input)
                        noise_sigma = torch.exp(torch.mul(noise_logvar, 0.5))
                        latents = noise_mu + torch.mul(noise_sigma, self.epsilon)
                        latents = latents.detach()
                        while(len(latents.size()) < len(input.size())):
                            latents = latents.unsqueeze(-1)
                    else:
                        latents = self.epsilon
                        while(len(latents.size()) < len(input.size())):
                            latents = latents.unsqueeze(-1)

                    fake_G1 = self.modelG[1](latents).detach()
                    if self.add_noise:
                        self.epsilon.data.resize_(input.size()).normal_(0, self.noise_var)
                        dis_input = fake_G1 + self.epsilon
                        out_G1 = self.modelD[0](dis_input)
                    else:
                        out_G1 = self.modelD[0](fake_G1)
                    loss_G1 = self.logits_loss(out_G1, self.target_G1[:batch_size])
                    logits_G1 = self.sigmoid(out_G1)
                    acc_G1 = 100*self.logits_eval(logits_G1, self.target_G1[:batch_size])

                    # train with fake from G2
                    fake_G2 = self.modelG[0](latents).detach()
                    if self.add_noise:
                        self.epsilon.data.resize_(input.size()).normal_(0, self.noise_var)
                        dis_input = fake_G2 + self.epsilon
                        out_G2 = self.modelD[0](dis_input)
                    else:
                        out_G2 = self.modelD[0](fake_G2)
                    loss_G2 = self.logits_loss(out_G2, self.target_G2[:batch_size])
                    logits_G2 = self.sigmoid(out_G2)
                    acc_G2 = 100*self.logits_eval(logits_G2, self.target_G2[:batch_size])

                    net_error = loss_real + loss_G1 + loss_G2
                    loss_D = ((epoch+1)/self.stage_epochs[stage])**2 * net_error
                    loss_D.backward()
                    self.optimizerD.step()

                    lossD = net_error.item()
                    if j < d_iterations:
                        # Bookkeeping
                        losses_train = {}
                        losses_train['Loss_D0'] = net_error.item()
                        losses_train['Loss_G0'] = 0
                        losses_train['MSE'] = 0
                        losses_train['KLD'] = 0
                        losses_train['D0_I'] = out_D[:,0].median().item()
                        losses_train['D0_G2'] = out_G2[:,0].median().item()
                        losses_train['D0_G1'] = out_G1[:,0].median().item()
                        losses_train['D1_I'] = out_D[:,1].median().item()
                        losses_train['D1_G2'] = out_G2[:,1].median().item()
                        losses_train['D1_G1'] = out_G1[:,1].median().item()
                        losses_train['Acc_Real'] = acc_real
                        losses_train['Acc_G1'] = acc_G1
                        losses_train['Acc_G2'] = acc_G2
                        self.monitor_train.update(losses_train, batch_size)
                        print('Stage %d: [%d/%d][%d/%d] Loss_D0: %.3f Loss_G0: %.3f D0_I %.3f D0_G2 %.3f D0_G1 %.3f D1_I %.3f D1_G2 %.3f D1_G1 %.3f Acc_Real %.1f Acc_G2 %.1f Acc_G1 %.1f'
                            % (stage, epoch, self.stage_epochs[stage], i, self.len_dataset, net_error.item(), 0, out_D[:,0].median().item(), out_G2[:,0].median().item(), out_G1[:,0].median().item(), out_D[:,1].median().item(), out_G2[:,1].median().item(), out_G1[:,1].median().item(), acc_real, acc_G2, acc_G1))

                        if i % self.plot_update_interval == 0:
                            losses_train['Learning_Rate_E'] = self.optimizerE.param_groups[0]['lr']
                            losses_train['Learning_Rate_G'] = self.optimizerG.param_groups[0]['lr']
                            losses_train['Learning_Rate_D'] = self.optimizerD.param_groups[0]['lr']
                            losses_train['Real'] = input.detach().cpu()
                            losses_train['Fakes_Encoder'] = fake_G2.detach().cpu()
                            losses_train['Fakes_Previous'] = fake_G1.detach().cpu()
                            losses_train['Fakes_Normal'] = self.modelG[0](self.fixed_noise).detach().cpu()
                            if self.use_encoder:
                                losses_train['Mean'] = noise_mu.mean().item()
                                losses_train['Sigma'] = noise_sigma.mean().item()
                            else:
                                losses_train['Mean'] = 0
                                losses_train['Sigma'] = 0
                            self.visualizer_train.update(losses_train)
                            torch.save(self.modelD[0].state_dict(), '%s/stage_%d_netD.pth' % (self.save_path, stage))


                ############################
                # Update Generator Network
                ############################
                j = 0
                while j < self.g_iter and i < self.len_dataset:
                    j += 1
                    # self.modelD[0].eval()
                    self.modelG[0].train()
                    self.modelG[0].zero_grad()
                    self.modelD[0].zero_grad()

                    fake_G2 = self.modelG[0](latents)
                    if self.add_noise:
                        self.epsilon.data.resize_(input.size()).normal_(0, self.noise_var)
                        dis_input = fake_G2 + self.epsilon
                        out_G2 = self.modelD[0](dis_input)
                    else:
                        out_G2 = self.modelD[0](fake_G2)

                    loss_G2 = self.logits_loss(out_G2, self.target_real[:batch_size])
                    # logits_G2 = self.sigmoid(out_G2)
                    # acc_G2 = self.logits_eval(logits_G2, self.target_real[:batch_size])

                    loss_G = ((epoch+1)/self.stage_epochs[stage])**2 * loss_G2
                    loss_G.backward()
                    self.optimizerG.step()
                    self.giterations += 1

                    if j < self.g_iter:
                        i += 1
                        # Bookkeeping
                        losses_train = {}
                        losses_train['Loss_D0'] = net_error.item()
                        losses_train['Loss_G0'] = loss_G2.item()
                        losses_train['MSE'] = 0
                        losses_train['KLD'] = 0
                        losses_train['D0_I'] = out_D[:,0].median().item()
                        losses_train['D0_G2'] = out_G2[:,0].median().item()
                        losses_train['D0_G1'] = out_G1[:,0].median().item()
                        losses_train['D1_I'] = out_D[:,1].median().item()
                        losses_train['D1_G2'] = out_G2[:,1].median().item()
                        losses_train['D1_G1'] = out_G1[:,1].median().item()
                        losses_train['Acc_Real'] = acc_real
                        losses_train['Acc_G1'] = acc_G1
                        losses_train['Acc_G2'] = acc_G2
                        self.monitor_train.update(losses_train, batch_size)
                        print('Stage %d: [%d/%d][%d/%d] Loss_D0: %.3f Loss_G0: %.3f D0_I %.3f D0_G2 %.3f D0_G1 %.3f D1_I %.3f D1_G2 %.3f D1_G1 %.3f Acc_Real %.1f Acc_G2 %.1f Acc_G1 %.1f'
                            % (stage, epoch, self.stage_epochs[stage], i, self.len_dataset, net_error.item(), loss_G2.item(), out_D[:,0].median().item(), out_G2[:,0].median().item(), out_G1[:,0].median().item(), out_D[:,1].median().item(), out_G2[:,1].median().item(), out_G1[:,1].median().item(), acc_real, acc_G2, acc_G1))

                        if i % self.plot_update_interval == 0:
                            losses_train['Learning_Rate_E'] = self.optimizerE.param_groups[0]['lr']
                            losses_train['Learning_Rate_G'] = self.optimizerG.param_groups[0]['lr']
                            losses_train['Learning_Rate_D'] = self.optimizerD.param_groups[0]['lr']
                            losses_train['Real'] = input.detach().cpu()
                            losses_train['Fakes_Encoder'] = fake_G2.detach().cpu()
                            losses_train['Fakes_Previous'] = fake_G1.detach().cpu()
                            losses_train['Fakes_Normal'] = self.modelG[0](self.fixed_noise).detach().cpu()
                            if self.use_encoder:
                                losses_train['Mean'] = noise_mu.mean().item()
                                losses_train['Sigma'] = noise_sigma.mean().item()
                            else:
                                losses_train['Mean'] = 0
                                losses_train['Sigma'] = 0
                            self.visualizer_train.update(losses_train)

                        input = data_iter.next()[0]
                        input = input.to(self.device)
                        batch_size = input.size(0)
                        self.epsilon.data.resize_(batch_size, self.nz).normal_(0, 1)
                        if self.use_encoder:
                            noise_mu, noise_logvar = self.Encoder(input)
                            noise_sigma = torch.exp(torch.mul(noise_logvar, 0.5))
                            latents = noise_mu + torch.mul(noise_sigma, self.epsilon)
                            latents = latents.detach()
                            while(len(latents.size()) < len(input.size())):
                                latents = latents.unsqueeze(-1)
                            latents = latents.detach()
                        else:
                            latents = self.epsilon
                            while(len(latents.size()) < len(input.size())):
                                latents = latents.unsqueeze(-1)
                        lossD = 0

                # Bookkeeping
                losses_train = {}
                losses_train['Loss_D0'] = net_error.item()
                losses_train['Loss_G0'] = loss_G2.item()
                losses_train['MSE'] = 0
                losses_train['KLD'] = 0
                losses_train['D0_I'] = out_D[:,0].median().item()
                losses_train['D0_G2'] = out_G2[:,0].median().item()
                losses_train['D0_G1'] = out_G1[:,0].median().item()
                losses_train['D1_I'] = out_D[:,1].median().item()
                losses_train['D1_G2'] = out_G2[:,1].median().item()
                losses_train['D1_G1'] = out_G1[:,1].median().item()
                losses_train['Acc_Real'] = acc_real
                losses_train['Acc_G1'] = acc_G1
                losses_train['Acc_G2'] = acc_G2
                self.monitor_train.update(losses_train, batch_size)
                print('Stage %d: [%d/%d][%d/%d] Loss_D0: %.3f Loss_G0: %.3f D0_I %.3f D0_G2 %.3f D0_G1 %.3f D1_I %.3f D1_G2 %.3f D1_G1 %.3f Acc_Real %.1f Acc_G2 %.1f Acc_G1 %.1f'
                    % (stage, epoch, self.stage_epochs[stage], i, self.len_dataset, net_error.item(), loss_G2.item(), out_D[:,0].median().item(), out_G2[:,0].median().item(), out_G1[:,0].median().item(), out_D[:,1].median().item(), out_G2[:,1].median().item(), out_G1[:,1].median().item(), acc_real, acc_G2, acc_G1))

                if i % self.plot_update_interval == 0:
                    losses_train['Learning_Rate_E'] = self.optimizerE.param_groups[0]['lr']
                    losses_train['Learning_Rate_G'] = self.optimizerG.param_groups[0]['lr']
                    losses_train['Learning_Rate_D'] = self.optimizerD.param_groups[0]['lr']
                    losses_train['Real'] = input.detach().cpu()
                    losses_train['Fakes_Encoder'] = fake_G2.detach().cpu()
                    losses_train['Fakes_Previous'] = fake_G1.detach().cpu()
                    losses_train['Fakes_Normal'] = self.modelG[0](self.fixed_noise).detach().cpu()
                    if self.use_encoder:
                        losses_train['Mean'] = noise_mu.mean().item()
                        losses_train['Sigma'] = noise_sigma.mean().item()
                    else:
                        losses_train['Mean'] = 0
                        losses_train['Sigma'] = 0
                    self.visualizer_train.update(losses_train)

                if i % 250 == 0 or i == self.len_dataset:
                    try:
                        fake_normal_z = self.modelG[0](self.fixed_noise)
                        fake_encoder_z = fake_G2
                        if len(input.size()) < 3:
                            fig = plt.figure()
                            plt.hist(input.squeeze().cpu().numpy(), bins=60)
                            fig.savefig("{}/stage_{}_epoch_{}_real.png".format(self.save_path, stage, i), dpi=fig.dpi)
                            fig = plt.figure()
                            plt.hist(fake_encoder_z.squeeze().data.cpu().numpy(), bins=60)
                            fig.savefig("{}/stage_{}_epoch_{}_fake_enc.png".format(self.save_path, stage, i), dpi=fig.dpi)
                            fig = plt.figure()
                            plt.hist(fake_normal_z.squeeze().data.cpu().numpy(), bins=60)
                            fig.savefig("{}/stage_{}_epoch_{}_fake_norm.png".format(self.save_path, stage, i), dpi=fig.dpi)
                        else:
                            vutils.save_image(fake_normal_z, '%s/fake_samples_stage_%03d_epoch_%03d_normal_z.png' % (self.save_path, stage, epoch), normalize=True)
                            vutils.save_image(fake_encoder_z, '%s/fake_samples_stage_%03d_epoch_%03d_encoder_z.png' % (self.save_path, stage, epoch), normalize=True)
                            vutils.save_image(input, '%s/real_samples.png' % self.save_path, normalize=True)
                    except Exception as e:
                        print(e)

        try:
            loss = self.monitor_train.getvalues()
            self.log_loss_train.update(loss)
        except Exception as e:
            print("Error while logging loss")
            print(e)

    def test(self, stage, epoch, dataloader):
        self.monitor_test.reset()
        data_iter = iter(dataloader)

        # switch to eval mode
        # self.modelG[0].eval()
        self.modelG[0].zero_grad()
        # self.modelG[1].eval()
        self.modelG[1].zero_grad()
        # self.modelD[0].eval()
        self.modelD[0].zero_grad()
        # self.modelD[1].eval()
        self.modelD[1].zero_grad()
        # self.Encoder.eval()
        self.Encoder.zero_grad()
        self.t_zero = Variable(torch.zeros(1))

        epoch_score_D0 = torch.Tensor([]).cuda()
        epoch_score_D0_G0 = torch.Tensor([]).cuda()
        epoch_score_D0_G1 = torch.Tensor([]).cuda()
        epoch_ssim_score = 0.0
        epoch_psnr_score = 0.0
        epoch_disc_acc = 0.0
        num_batches = len(dataloader)

        i = 0
        while i < len(dataloader):
            ############################
            # Evaluate Network
            ############################
            acc = 0.0
            # Get real data
            input = data_iter.next()[0]
            i += 1
            batch_size = input.size(0)
            self.test_input.data.resize_(input.size()).copy_(input)

            # Generate fake data
            self.epsilon.data.resize_(batch_size, self.nz).normal_(0, 1)
            noise_mu, noise_logvar = self.Encoder(self.test_input)
            noise_sigma = torch.exp(torch.mul(noise_logvar, 0.5))
            latents = noise_mu + torch.mul(noise_sigma, self.epsilon)
            while(len(latents.size()) < len(self.test_input.size())):
                latents = latents.unsqueeze(-1)

            fake_G0 = self.modelG[0](latents)
            fake_G1 = self.modelG[1](latents)
            score_D0 = self.modelD[0](self.test_input, self.extra_layer, self.extra_layer_gamma)
            score_D0_G0 = self.modelD[0](fake_G0, self.extra_layer, self.extra_layer_gamma)
            score_D0_G1 = self.modelD[0](fake_G1, self.extra_layer, self.extra_layer_gamma)
            ssim_score, psnr_score = 0.0, 0.0
            data_range = input.max() - input.min()

            acc += torch.sum((score_D0 > 0).float())
            acc += torch.sum((score_D0_G0 <= 0).float())
            disc_acc = float(acc)*50/batch_size

            if self.args.net_type != 'gmm':
                compare_real = input.permute(0,2,3,1)
                compare_fake = fake_G0.permute(0,2,3,1)
                for j in range(batch_size):
                    ssim_score += compare_ssim(compare_real[j,...].cpu().numpy(), compare_fake[j,...].data.cpu().numpy(), data_range=data_range, multichannel=True)
                    psnr_score += compare_psnr(compare_real[j,...].cpu().numpy(), compare_fake[j,...].data.cpu().numpy(), data_range=data_range)

            epoch_score_D0 = torch.cat((epoch_score_D0, score_D0.data))
            epoch_score_D0_G0 = torch.cat((epoch_score_D0_G0, score_D0_G0.data))
            epoch_score_D0_G1 = torch.cat((epoch_score_D0_G1, score_D0_G1.data))
            epoch_ssim_score += ssim_score/batch_size
            epoch_psnr_score += psnr_score/batch_size
            epoch_disc_acc += disc_acc


            # Bookkeeping
            test_scores = {}
            test_scores['Test_Score_D0'] = score_D0.median().item()
            test_scores['Test_Score_D0_G0'] = score_D0_G0.median().item()
            test_scores['Test_Score_D0_G1'] = score_D0_G1.median().item()
            test_scores['SSIM'] = ssim_score/batch_size
            test_scores['PSNR'] = psnr_score/batch_size
            test_scores['Test_Disc_Acc'] = disc_acc
            self.monitor_test.update(test_scores, batch_size)
            print('Test: [%d/%d][%d/%d] Score_D0: %.3f Score_D0_G0: %.3f Score_D0_G1: %.3f SSIM: %.3f PSNR: %.3f Disc_Acc: %.3f'
                    % (epoch, self.stage_epochs[stage], i, len(dataloader), score_D0.median().item(), score_D0_G0.median().item(), score_D0_G1.median().item(), ssim_score/batch_size, psnr_score/batch_size, disc_acc))

            # if (i % int(len(dataloader)*0.25)) == 0:
            #     test_scores['Test_Real'] = self.test_input.data.cpu()
            #     test_scores['Test_Fakes_Encoder'] = fake_G0.data.cpu()
            #     self.visualizer_test.update(test_scores)

            if i == len(dataloader)-2:
                try:
                    fake_encoder_z = fake_G0
                    if len(self.input.size()) < 3:
                        fig = plt.figure()
                        plt.hist(input.squeeze().cpu().numpy(), bins=60)
                        fig.savefig("{}/stage_{}_epoch_{}_real.png".format(self.save_path, stage, i), dpi=fig.dpi)
                        fig = plt.figure()
                        plt.hist(fake_encoder_z.squeeze().data.cpu().numpy(), bins=60)
                        fig.savefig("{}/stage_{}_epoch_{}_fake_enc.png".format(self.save_path, stage, i), dpi=fig.dpi)
                        fig = plt.figure()
                        plt.hist(fake_normal_z.squeeze().data.cpu().numpy(), bins=60)
                        fig.savefig("{}/stage_{}_epoch_{}_fake_norm.png".format(self.save_path, stage, i), dpi=fig.dpi)
                    else:
                        vutils.save_image(fake_encoder_z.data, '%s/val_fake_samples_stage_%03d_encoder_z.png' % (self.save_path, stage), normalize=True)
                        vutils.save_image(input, '%s/val_real_samples.png' % self.save_path, normalize=True)
                except Exception as e:
                    print(e)

        avg_mean_scores_D0 = epoch_score_D0.median()
        avg_mean_scores_D0_G0 = epoch_score_D0_G0.median()
        avg_mean_scores_D0_G1 = epoch_score_D0_G1.median()

        test_scores = {}
        test_scores['Test_Score_D0'] = avg_mean_scores_D0
        test_scores['Test_Score_D0_G0'] = avg_mean_scores_D0_G0
        test_scores['Test_Score_D0_G1'] = avg_mean_scores_D0_G1
        test_scores['SSIM'] = epoch_ssim_score/(num_batches)
        test_scores['PSNR'] = epoch_psnr_score/(num_batches)
        test_scores['Test_Disc_Acc'] = epoch_disc_acc / (num_batches)
        test_scores['Test_Real'] = self.test_input.data.cpu()
        test_scores['Test_Fakes_Encoder'] = fake_G0.data.cpu()
        self.visualizer_test.update(test_scores)

        # Check for Discriminator Saturation
        if avg_mean_scores_D0_G0 > avg_mean_scores_D0 - (avg_mean_scores_D0 - avg_mean_scores_D0_G1)/4:
            self.update_opt_flag = True
        else:
            self.update_opt_flag = False

        # if self.gp and stage == 1:
        #     self.margin = np.percentile(epoch_score_D0.cpu().numpy(), 50) - np.percentile(epoch_score_D0_G0.cpu().numpy(), 50)

        try:
            loss = self.monitor_test.getvalues()
            self.log_loss_test.update(loss)
        except Exception as e:
            print("Error while logging test loss")
            print(e)
        if stage == 1 and epoch_disc_acc / (num_batches) >= 80: #96
            return True
        else:
            return False

    def get_model_norm(self, model):
        norm = torch.zeros(1)
        param_list = list(model.main.parameters())
        for l in param_list:
            norm += torch.norm(l.data)
        return norm

    def update_opt_disc(self):
        # Increase Discriminator Capacity When Generator Becomes Strong
        if self.add_capacity:
            self.extra_layer = min(self.extra_layer + 1, 5)
            print("One layer added to the Discriminator")
        elif self.add_clamp:
            self.clamp_upper += 0.0002
            self.clamp_lower -= 0.0002
            self.lr_dis *= 1.003
            self.lr_gen *= 1.003
            if self.lr_dis > 10*self.lr_vae:
                self.lr_dis = 10*self.lr_vae
            if self.lr_gen > 10*self.lr_vae:
                self.lr_gen = 10*self.lr_vae
            self.optimizerD = self.initialize_optimizer(self.modelD[0], lr=self.lr_dis, optim_method=self.optim_method)
            self.optimizerG = self.initialize_optimizer(self.modelG[0], lr=self.lr_gen, optim_method=self.optim_method)
            print("Clamping increased to: Upper {} Lower {} LR_Disc {} LR_Gen {}".format(self.clamp_upper, self.clamp_lower, self.lr_dis, self.lr_gen))

    def optimize_discriminator(self, stage, epoch, dataloader):
        ############################
        # Optimize Discriminator Network
        ############################
        data_iter = iter(dataloader)
        self.len_dataset = int(len(dataloader) * self.dataset_fraction)
        self.modelD[0].train()
        self.modelG[0].eval()
        i = 0
        avg_mean_scores_D0 = 0.0
        avg_mean_scores_D0_G0 = 0.0
        avg_mean_scores_D0_G1 = 0.0

        while i < self.len_dataset:
            acc = 0.0
            i += 1

            # clamp parameters to a cube
            if not self.gp:
                for p in self.modelD[0].parameters():
                    p.data.clamp_(self.clamp_lower, self.clamp_upper)

            input = data_iter.next()[0]
            batch_size = input.size(0)
            self.input.data.resize_(input.size()).copy_(input)

            self.modelD[0].zero_grad()
            self.modelG[0].zero_grad()

            # train with real
            if self.add_noise:
                self.epsilon.data.resize_(self.input.size()).normal_(0, self.noise_var)
                dis_input = self.input + self.epsilon
                scores_D0 = self.modelD[0](dis_input, self.extra_layer, self.extra_layer_gamma)
            else:
                scores_D0 = self.modelD[0](self.input, self.extra_layer, self.extra_layer_gamma)
            # if self.add_noise:
            #     self.epsilon.data.resize_(self.input.size()).normal_(0, self.noise_var)
            #     dis_input = self.input + self.epsilon
            #     scores_D1 = self.modelD[1](dis_input, 0)
            # else:
            #     scores_D1 = self.modelD[1](self.input, 0)
            acc += torch.sum((scores_D0 > self.acc_margin).float())

            # train with fake
            self.epsilon.data.resize_(batch_size, self.nz).normal_(0, 1)
            noise_mu, noise_logvar = self.Encoder(self.input)
            noise_sigma = torch.exp(torch.mul(noise_logvar, 0.5))
            latents = noise_mu + torch.mul(noise_sigma, self.epsilon)
            latents = latents.detach()
            while(len(latents.size()) < len(input.size())):
                latents = latents.unsqueeze(-1)
            latents = latents.detach()

            fake_G0 = self.modelG[0](latents).detach()
            fake_G0.requires_grad = True
            if self.add_noise:
                self.epsilon.data.resize_(self.input.size()).normal_(0, self.noise_var)
                dis_input = fake_G0 + self.epsilon
                scores_D0_G0 = self.modelD[0](dis_input, self.extra_layer, self.extra_layer_gamma)
            else:
                scores_D0_G0 = self.modelD[0](fake_G0, self.extra_layer, self.extra_layer_gamma)
            fake_G1 = self.modelG[1](latents).detach()
            # if self.add_noise:
            #     self.epsilon.data.resize_(self.input.size()).normal_(0, self.noise_var)
            #     dis_input = fake_G1 + self.epsilon
            #     scores_D1_G1 = self.modelD[1](dis_input, 0)
            # else:
            #     scores_D1_G1 = self.modelD[1](fake_G1, 0)
            fake_G1.requires_grad = True
            if self.add_noise:
                self.epsilon.data.resize_(self.input.size()).normal_(0, self.noise_var)
                dis_input = fake_G1 + self.epsilon
                scores_D0_G1 = self.modelD[0](dis_input, self.extra_layer, self.extra_layer_gamma)
            else:
                scores_D0_G1 = self.modelD[0](fake_G1, self.extra_layer, self.extra_layer_gamma)
            acc += torch.sum((scores_D0_G0 <= 0).float())
            disc_acc = float(acc)*50 / batch_size

            # Compute loss and do backward()
            # disc_abs_diff = torch.pow(scores_D0 - self.marker_high, 2).mean() + torch.pow(scores_D0_G1 - self.marker_low, 2).mean()
            # disc_abs_diff = torch.norm(scores_D0 - self.marker_high) + torch.norm(scores_D0_G1 - self.marker_low)
            # disc_abs_diff = torch.abs(scores_D0 - self.marker_high).mean() + torch.abs(scores_D0_G1 - self.marker_low).mean()
            disc_abs_diff = self.criterion.hinge_loss(scores_D0, 1, self.marker_high)
            disc_abs_diff += self.criterion.hinge_loss(scores_D0_G1, -1, -self.marker_low)
            errD0 = self.criterion.rankerD([scores_D0, scores_D0_G0, scores_D0_G1])
            net_error = 100*errD0 + self.disc_diff_weight*disc_abs_diff

            if self.gp:
                gradient_penalty = self.calc_gradient_penalty(self.input, fake_G0, batch_size)
                net_error += gradient_penalty
            else:
                gradient_penalty = self.t_zero

            try:
                net_error.backward()
            except Exception as e:
                print(e)

            self.optimizerD.step()
            self.extra_layer_gamma = min(self.extra_layer_gamma + 1/(2*self.len_dataset), 1)

            mean_scores_D0 = scores_D0.median().item()
            mean_scores_D0_G0 = scores_D0_G0.median().item()
            mean_scores_D0_G1 = scores_D0_G1.median().item()
            # avg_mean_scores_D0 = (avg_mean_scores_D0*(i-1) + mean_scores_D0) / i
            # avg_mean_scores_D0_G0 = (avg_mean_scores_D0_G0*(i-1) + mean_scores_D0_G0) / i
            # avg_mean_scores_D0_G1 = (avg_mean_scores_D0_G1*(i-1) + mean_scores_D0_G1) / i
            avg_mean_scores_D0 = avg_mean_scores_D0 * 0.2 + mean_scores_D0 * 0.8
            avg_mean_scores_D0_G0 = avg_mean_scores_D0_G0 * 0.2 + mean_scores_D0_G0 * 0.8
            avg_mean_scores_D0_G1 = avg_mean_scores_D0_G1 * 0.2 + mean_scores_D0_G1 * 0.8

            if avg_mean_scores_D0_G0 < (avg_mean_scores_D0 + avg_mean_scores_D0_G1)/2 and disc_abs_diff.item() < 5:
                return False

            # Bookkeeping
            losses_train = {}
            losses_train['Loss_D0'] = errD0.item()
            losses_train['Loss_G0'] = 0
            losses_train['MSE'] = 0
            losses_train['KLD'] = 0
            losses_train['Corr_Loss'] = 0
            losses_train['Mean'] = noise_mu.mean().item()
            losses_train['Sigma'] = noise_sigma.mean().item()
            losses_train['Score_D0'] = mean_scores_D0
            losses_train['Score_D0_G0'] = mean_scores_D0_G0
            losses_train['Score_D0_G1'] = mean_scores_D0_G1
            losses_train['Score_D0_Normal_G0'] = self.t_zero.item()
            losses_train['Score_D0_Normal_G1'] = self.t_zero.item()
            losses_train['Disc_Diff'] = disc_abs_diff.item()
            losses_train['Disc_Acc'] = disc_acc
            losses_train['Clamp'] = self.clamp_upper
            self.monitor_train.update(losses_train, batch_size)
            print('Stage %d: [%d/%d][%d/%d] Loss_D: %.3f Loss_G: %.3f Score_D0: %.3f Score_D0_G0: %.3f Score_D0_G1: %.3f Disc_Diff %.3f Disc_Acc %.3f Extra_Gamma %.3f'
                % (stage, epoch, self.stage_epochs[stage], i, self.len_dataset, errD0.item(), 0, mean_scores_D0, mean_scores_D0_G0, mean_scores_D0_G1, disc_abs_diff.item(), disc_acc, self.extra_layer_gamma))

            if i % self.plot_update_interval == 0:
                # Compute MSE
                loss_mse = self.criterion.diff_loss(fake_G0, self.input, type=self.vae_loss_type)
                losses_train['MSE'] = loss_mse.item()
                losses_train['netD_norm_w'] = self.get_model_norm(self.modelD[0])[0]
                losses_train['Learning_Rate_E'] = self.optimizerE.param_groups[0]['lr']
                losses_train['Learning_Rate_G'] = self.optimizerG.param_groups[0]['lr']
                losses_train['Learning_Rate_D'] = self.optimizerD.param_groups[0]['lr']
                losses_train['Gradient_Penalty'] = gradient_penalty.item()
                losses_train['Real'] = self.input.data.cpu()
                losses_train['Fakes_Encoder'] = fake_G0.data.cpu()
                losses_train['Fakes_Previous'] = fake_G1.data.cpu()
                losses_train['Fakes_Normal'] = self.modelG[0](self.fixed_noise).data.cpu()
                # losses_train['Corr_Output'] = torch.zeros(fake_G0.size())
                self.visualizer_train.update(losses_train)

            if disc_abs_diff.item() < 4:
                self.disc_diff_weight = self.disc_diff_weight_init/1000
            elif disc_abs_diff.item() < 10:
                self.disc_diff_weight = self.disc_diff_weight_init/10
            else:
                self.disc_diff_weight = self.disc_diff_weight_init

        print('Stage %d: [%d/%d] Avg_Scores_D0 %.3f Avg_Scores_D0_G0 %.3f Avg_Scores_D0_G1 %.3f' % (stage, epoch, self.stage_epochs[stage], avg_mean_scores_D0, avg_mean_scores_D0_G0, avg_mean_scores_D0_G1))
        return True

    def compute_markers(self, dataloader):
        data_iter = iter(dataloader)
        self.len_dataset = int(len(dataloader) * self.dataset_fraction)
        # self.modelD[0].eval()
        # self.modelG[0].eval()
        i = 0
        all_scores_D1 = torch.Tensor([]).cuda()
        all_scores_D1_G1 = torch.Tensor([]).cuda()

        while i < self.len_dataset:
            i += 1
            input = data_iter.next()[0]
            batch_size = input.size(0)
            self.input.data.resize_(input.size()).copy_(input)

            if self.add_noise:
                self.epsilon.data.resize_(self.input.size()).normal_(0, self.noise_var)
                dis_input = self.input + self.epsilon
                scores_D1 = self.modelD[1](dis_input, 0)
            else:
                scores_D1 = self.modelD[1](self.input, 0)

            # train with fake
            self.epsilon.data.resize_(batch_size, self.nz).normal_(0, 1)
            noise_mu, noise_logvar = self.Encoder(self.input)
            noise_sigma = torch.exp(torch.mul(noise_logvar, 0.5))
            latents = noise_mu + torch.mul(noise_sigma, self.epsilon)
            latents = latents.detach()
            while(len(latents.size()) < len(input.size())):
                latents = latents.unsqueeze(-1)
            latents = latents.detach()

            fake_G1 = self.modelG[1](latents).detach()
            if self.add_noise:
                self.epsilon.data.resize_(self.input.size()).normal_(0, self.noise_var)
                dis_input = fake_G1 + self.epsilon
                scores_D1_G1 = self.modelD[1](dis_input, 0)
            else:
                scores_D1_G1 = self.modelD[1](fake_G1, 0)

            all_scores_D1 = torch.cat((all_scores_D1, scores_D1.data), dim=0)
            all_scores_D1_G1 = torch.cat((all_scores_D1_G1, scores_D1_G1.data), dim=0)
        return all_scores_D1.median(), all_scores_D1_G1.median()
