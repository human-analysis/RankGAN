# train.py

import torch
import torchvision
import torchvision.utils as vutils
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.autograd as autograd
import copy
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim, compare_psnr
import time
import os
import numpy as np
import plugins

class Trainer():
    def __init__(self, args, modelD, modelG, Encoder, criterion, prevD, prevG):

        self.args = args
        self.modelD = [modelD for i in range(2)]
        self.modelG = [modelG for i in range(2)]
        self.Encoder = Encoder
        self.prevD = prevD
        self.prevG = prevG
        self.criterion = criterion
        self.plot_update_interval = args.plot_update_interval

        self.port = args.port
        self.env = args.env
        self.result_path = args.result_path
        self.save_path = args.save
        self.log_path = args.logs
        self.dataset_fraction = args.dataset_fraction
        self.len_dataset = 0

        self.cuda = args.cuda
        self.nepochs = args.nepochs
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
        self.acc_margin = args.margin
        self.disc_diff_weight_init = args.disc_diff_weight
        self.disc_diff_weight = self.disc_diff_weight_init
        self.num_stages = args.num_stages
        self.counter_disc_opt = 0
        self.adaptive_iter = args.adaptive_iter

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
        self.input = Variable(torch.FloatTensor(self.batch_size,self.nchannels,self.resolution_high,self.resolution_wide), requires_grad=True)
        self.test_input = Variable(torch.FloatTensor(self.batch_size,self.nchannels,self.resolution_high,self.resolution_wide), volatile=True)
        self.fixed_noise = Variable(torch.FloatTensor(self.batch_size, self.nz).normal_(0, 1), volatile=True)
        self.epsilon = Variable(torch.randn(self.batch_size, self.nz), requires_grad=False)

        if args.cuda:
            self.input = self.input.cuda()
            self.test_input = self.test_input.cuda()
            self.fixed_noise = self.fixed_noise.cuda()
            self.epsilon = self.epsilon.cuda()

        # Initialize optimizer
        self.optimizerE = self.initialize_optimizer(self.Encoder, lr=self.lr_vae, optim_method='Adam')
        self.optimizerG = self.initialize_optimizer(self.modelG[0], lr=self.lr_vae, optim_method='Adam')
        self.optimizerD = self.initialize_optimizer(self.modelD[0], lr=self.lr_dis, optim_method='Adam', weight_decay=100)
        # self.schedulerE = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerE, factor=self.lr_decay, patience=self.scheduler_patience, min_lr=1e-3*self.lr_vae)
        # self.schedulerG = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, factor=self.lr_decay, patience=self.scheduler_patience, min_lr=1e-3*self.lr_vae)
        # self.schedulerD = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, factor=self.lr_decay, patience=self.scheduler_patience, min_lr=1e-3*self.lr_vae)
        # Automatic scheduler
        self.schedulerE = plugins.AutomaticLRScheduler(self.optimizerE, maxlen=self.scheduler_maxlen, factor=self.lr_decay, patience=self.scheduler_patience)
        self.schedulerG = plugins.AutomaticLRScheduler(self.optimizerG, maxlen=self.scheduler_maxlen, factor=self.lr_decay, patience=self.scheduler_patience)
        self.schedulerD = plugins.AutomaticLRScheduler(self.optimizerD, maxlen=self.scheduler_maxlen, factor=self.lr_decay, patience=self.scheduler_patience)

        # logging training
        self.log_loss_train = plugins.Logger(args.logs, 'TrainLogger.txt')
        self.params_loss_train = ['Loss_D0', 'Loss_G0', 'MSE', 'KLD', 'Corr_Loss', 'Mean', 'Sigma', 'Score_D0', 'Score_D0_G0', 'Score_D0_G1', 'Score_D0_Normal_G0', 'Score_D0_Normal_G1', 'Disc_Diff', 'Disc_Acc', 'Clamp']
        self.log_loss_train.register(self.params_loss_train)

        self.log_loss_test = plugins.Logger(args.logs, 'TestLogger.txt')
        self.params_loss_test = ['SSIM', 'PSNR', 'Test_Score_D0', 'Test_Score_D0_G0', 'Test_Score_D0_G1', 'Test_Disc_Acc']
        self.log_loss_test.register(self.params_loss_test)

        # monitor training
        self.monitor_train = plugins.Monitor()
        self.params_monitor_train = ['Loss_D0', 'Loss_G0', 'MSE', 'KLD', 'Corr_Loss', 'Mean', 'Sigma', 'Score_D0', 'Score_D0_G0', 'Score_D0_G1', 'Score_D0_Normal_G0', 'Score_D0_Normal_G1', 'Disc_Diff', 'Disc_Acc', 'Clamp']
        self.monitor_train.register(self.params_monitor_train)

        self.monitor_test = plugins.Monitor()
        self.params_monitor_test = ['SSIM', 'PSNR', 'Test_Score_D0', 'Test_Score_D0_G0', 'Test_Score_D0_G1', 'Test_Disc_Acc']
        self.monitor_test.register(self.params_monitor_test)

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
        self.visualizer_train = plugins.HourGlassVisualizer(port=self.port, env=self.env, title='Train')
        self.params_visualizer_train = {
        'Loss_D0':{'dtype':'scalar', 'vtype':'plot', 'win': 'loss_gan', 'layout': {'windows': ['Loss_D0', 'Loss_G0'], 'id': 0}},
        'Loss_G0':{'dtype':'scalar','vtype':'plot', 'win': 'loss_gan', 'layout': {'windows': ['Loss_D0', 'Loss_G0'], 'id': 1}},
        'MSE':{'dtype':'scalar','vtype':'plot', 'win': 'enc_losses', 'layout': {'windows': ['MSE', 'KLD', 'Corr_Loss'], 'id': 0}},
        'KLD':{'dtype':'scalar','vtype':'plot', 'win': 'enc_losses', 'layout': {'windows': ['MSE', 'KLD', 'Corr_Loss'], 'id': 1}},
        'Corr_Loss':{'dtype':'scalar','vtype':'plot', 'win': 'enc_losses', 'layout': {'windows': ['MSE', 'KLD', 'Corr_Loss'], 'id': 2}},
        'Mean':{'dtype':'scalar','vtype':'plot', 'win': 'norm_params', 'layout': {'windows': ['Mean', 'Sigma'], 'id': 0}},
        'Sigma':{'dtype':'scalar','vtype':'plot', 'win': 'norm_params', 'layout': {'windows': ['Mean', 'Sigma'], 'id': 1}},
        'Score_D0':{'dtype':'scalar','vtype':'plot', 'win': 'loss_D0', 'layout': {'windows': ['Score_D0', 'Score_D0_G0', 'Score_D0_G1'], 'id': 0}},
        'Score_D0_G0':{'dtype':'scalar','vtype':'plot', 'win': 'loss_D0', 'layout': {'windows': ['Score_D0', 'Score_D0_G0', 'Score_D0_G1'], 'id': 1}},
        'Score_D0_G1':{'dtype':'scalar','vtype':'plot', 'win': 'loss_D0', 'layout': {'windows': ['Score_D0', 'Score_D0_G0', 'Score_D0_G1'], 'id': 2}},
        'netD_norm_w':{'dtype':'scalar','vtype':'plot', 'win': 'netd_norm'},
        'Score_D0_Normal_G0':{'dtype':'scalar','vtype':'plot', 'win': 'normal_losses', 'layout': {'windows': ['Score_D0_Normal_G0', 'Score_D0_Normal_G1'], 'id': 0}},
        'Score_D0_Normal_G1':{'dtype':'scalar','vtype':'plot', 'win': 'normal_losses', 'layout': {'windows': ['Score_D0_Normal_G0', 'Score_D0_Normal_G1'], 'id': 1}},
        'Learning_Rate_E':{'dtype':'scalar','vtype':'plot', 'win': 'lr_E'},
        'Learning_Rate_G':{'dtype':'scalar','vtype':'plot', 'win': 'lr_G'},
        'Learning_Rate_D':{'dtype':'scalar','vtype':'plot', 'win': 'lr_D'},
        'Gradient_Penalty':{'dtype':'scalar','vtype':'plot', 'win': 'gp'},
        'Disc_Diff':{'dtype':'scalar','vtype':'plot', 'win': 'disc_diff'},
        'Disc_Acc':{'dtype':'scalar','vtype':'plot', 'win': 'disc_acc'},
        'Clamp':{'dtype':'scalar','vtype':'plot', 'win': 'clamp'},

        'Real': {'dtype': output_dtype, 'vtype': output_vtype, 'win': 'real'},
        'Fakes_Encoder': {'dtype': output_dtype, 'vtype': output_vtype, 'win': 'fakes_enc'},
        'Fakes_Normal': {'dtype': output_dtype, 'vtype': output_vtype, 'win': 'fakes_normal'},
        'Fakes_Previous': {'dtype': output_dtype, 'vtype': output_vtype, 'win': 'fakes_prev'},
        # 'Corr_Output': {'dtype': output_dtype, 'vtype': output_vtype, 'win': 'corr'},
        }
        self.visualizer_train.register(self.params_visualizer_train)

        self.visualizer_test = plugins.HourGlassVisualizer(port=self.port, env=self.env, title='Test')
        self.params_visualizer_test = {
        'Test_Score_D0':{'dtype':'scalar', 'vtype':'plot', 'win':'test_disc_scores', 'layout':{'windows': ['Test_Score_D0', 'Test_Score_D0_G0', 'Test_Score_D0_G1'], 'id':0}},
        'Test_Score_D0_G0':{'dtype':'scalar', 'vtype':'plot', 'win':'test_disc_scores', 'layout':{'windows': ['Test_Score_D0', 'Test_Score_D0_G0', 'Test_Score_D0_G1'], 'id':1}},
        'Test_Score_D0_G1':{'dtype':'scalar', 'vtype':'plot', 'win':'test_disc_scores', 'layout':{'windows': ['Test_Score_D0', 'Test_Score_D0_G0', 'Test_Score_D0_G1'], 'id':2}},
        'SSIM':{'dtype':'scalar', 'vtype':'plot', 'win':'ssim_scores'},
        'PSNR':{'dtype':'scalar', 'vtype':'plot', 'win':'psnr_scores'},
        'Test_Real': {'dtype': output_dtype, 'vtype': output_vtype, 'win': 'real_test'},
        'Test_Fakes_Encoder': {'dtype': output_dtype, 'vtype': output_vtype, 'win': 'fakes_enc_test'},
        'Test_Disc_Acc': {'dtype':'scalar','vtype':'plot', 'win': 'test_disc_acc'},
        }
        self.visualizer_test.register(self.params_visualizer_test)
        # impulse = self.criterion.impulse[0,...].data.cpu().numpy()
        # print(impulse)
        # self.visualizer_train.viz.image(impulse, opts=dict(title='ground truth g'))
        # self.visualizer_train.viz.surf(X=impulse.tolist(), opts=dict(colormap='Hot', title='ground truth g'))

        # display training progress
        self.print_train = '[%d/%d][%d/%d] '
        for item in self.params_loss_train:
            self.print_train = self.print_train + item + " %.3f "

        self.print_test = '[%d/%d][%d/%d] '
        for item in self.params_loss_test:
            self.print_test = self.print_test + item + " %.3f "

        self.giterations = 0
        self.d_iter_init = args.d_iter
        self.d_iter = self.d_iter_init
        self.g_iter_init = args.g_iter
        self.g_iter = self.g_iter_init
        self.clamp_lower = args.clamp_lower
        self.clamp_upper = args.clamp_upper
        self.marker_high = args.margin
        self.marker_low = - args.margin
        print('Discriminator:', self.modelD[0])
        print('Generator:', self.modelG[0])
        print('Encoder:', self.Encoder)

        # define a zero tensor
        self.t_zero = Variable(torch.zeros(1), requires_grad=True)
        self.update_opt_flag = False  #False
        self.update_opt_counter = 0       # -2
        self.disc_optimize = args.disc_optimize
        self.add_capacity = args.add_capacity
        self.add_clamp = args.add_clamp
        self.add_noise = args.add_noise
        self.noise_var = args.noise_var
        self.extra_layer = 0
        self.extra_layer_gamma = 0
        self.gen_gamma = args.gen_gamma
        self.gp_norm = args.gp_norm
        self.rank_error_weight = args.rank_weight
        self.gogan_optim_set = False
        self.optimize_mse = args.optimize_mse
        self.weight_mse = args.weight_mse

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
            self.weight_vae, self.weight_gan = 0, self.weight_gan_final
            self.optimizerD = self.initialize_optimizer(self.modelD[0], lr=self.lr_vae, optim_method='Adam', weight_decay=self.args.stage1_weight_decay)
            self.schedulerD = plugins.AutomaticLRScheduler(self.optimizerD, maxlen=self.scheduler_maxlen, factor=self.lr_decay, patience=self.scheduler_patience)
            self.criterion.set_margin(self.margin)
            print("Margin: {}".format(self.margin))
        else:
            # if stage == 2 and ((not self.gp) or (self.gp and self.start_stage == stage)):
            if stage > self.start_stage:
                self.gen_gamma = 0
            if not self.gogan_optim_set:
                self.optimizerD = self.initialize_optimizer(self.modelD[0], lr=self.lr_dis, optim_method=self.optim_method)
                self.optimizerG = self.initialize_optimizer(self.modelG[0], lr=self.lr_gen, optim_method=self.optim_method)
                self.gogan_optim_set = True
            self.margin *= 2
            self.weight_vae, self.weight_gan = 0, self.weight_gan_final
            self.criterion.set_margin(self.margin)
            self.disc_diff_weight = self.disc_diff_weight_init

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
            self.marker_high, self.marker_low = self.compute_markers(dataloader)
            print("Marker High: {}, Marker Low: {}".format(self.marker_high, self.marker_low))
            print("Margin: {}".format(self.margin))
            self.margin = self.margin / 2.0

        print("Loss Weightage: VAE: {}, GAN: {}".format(self.weight_vae, self.weight_gan))

    def calc_gradient_penalty(self, real_data, fake_data, batch_size):
        alpha = torch.rand(batch_size, 1)
        while len(alpha.size()) < len(real_data.size()):
            alpha = alpha.unsqueeze(-1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda() if self.cuda else alpha

        interpolates = alpha * real_data.data + ((1 - alpha) * fake_data.data)

        if self.cuda:
            interpolates = interpolates.cuda()
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.modelD[0](interpolates, self.extra_layer, self.extra_layer_gamma)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda() if self.cuda else torch.ones(
                                      disc_interpolates.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gp_lambda
        return gradient_penalty

    def train(self, stage, epoch, dataloader):
        self.monitor_train.reset()
        data_iter = iter(dataloader)
        self.len_dataset = int(len(dataloader) * self.dataset_fraction)
        if epoch == 0:
            print("Length of Dataset: {}".format(self.len_dataset))

        if self.args.net_type == 'gmm':
            self.visualizer_train.reset('Real')
            self.visualizer_train.reset('Fakes_Encoder')
            self.visualizer_train.reset('Fakes_Normal')

        # Optimize Discriminator if needed
        # while (stage == 2 and epoch == 0 and self.update_opt_counter == -1) or self.update_opt_flag:
        #     if self.update_opt_counter % 5 == 0:
        #         self.update_opt_disc()
        #     print("Training Discriminator Till optimality")
        #     self.update_opt_flag = self.optimize_discriminator(stage, epoch, dataloader)
        #     self.update_opt_counter += 1

        if stage == 2 and self.disc_optimize:
            self.optimize_discriminator(stage, epoch, dataloader)
            self.disc_optimize = False
        if self.update_opt_flag:
            if self.update_opt_counter % 3 == 0:
                self.update_opt_disc()
            self.update_opt_counter += 1

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
                self.input.data.resize_(input.size()).copy_(input)

                # zero grad
                self.Encoder.zero_grad()
                self.modelG[0].zero_grad()

                # get latents
                self.epsilon.data.resize_(batch_size, self.nz).normal_(0, 1)
                noise_mu, noise_logvar = self.Encoder(self.input)
                noise_sigma = torch.exp(torch.mul(noise_logvar, 0.5))
                latents = noise_mu + torch.mul(noise_sigma, self.epsilon)
                while(len(latents.size()) < len(self.input.size())):
                    latents = latents.unsqueeze(-1)

                # compute vae loss
                fake = self.modelG[0](latents)
                loss_mse = self.criterion.diff_loss(fake, self.input, type=self.vae_loss_type)
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
                self.schedulerE.step(loss_kld.data[0])
                self.schedulerG.step(loss_mse.data[0])

                # Bookkeeping
                losses_train = {}
                losses_train['Loss_D0'] = 0
                losses_train['Loss_G0'] = 0
                losses_train['MSE'] = loss_mse.data[0]
                losses_train['KLD'] = loss_kld.data[0]
                losses_train['Corr_Loss'] = loss_correlation.data[0]
                losses_train['Mean'] = noise_mu.mean().data[0]
                losses_train['Sigma'] = noise_sigma.mean().data[0]
                losses_train['Score_D0'] = 0
                losses_train['Score_D0_G0'] = 0
                losses_train['Score_D0_G1'] = 0
                losses_train['Score_D0_Normal_G0'] = 0
                losses_train['Score_D0_Normal_G1'] = 0
                losses_train['Disc_Diff'] = 0
                losses_train['Disc_Acc'] = 0
                losses_train['Clamp'] = self.clamp_upper
                self.monitor_train.update(losses_train, batch_size)
                print('Stage %d: [%d/%d][%d/%d] Loss_D0: %.3f Loss_G0: %.3f MSE: %.3f KLD: %.3f Corr_Loss %.3f Score_D0: %.3f Score_D0_G0: %.3f Score_D0_G1: %.3f Disc_Acc %.3f'
                    % (stage, epoch, self.stage_epochs, i, self.len_dataset, 0, 0, loss_mse.data[0], loss_kld.data[0], loss_correlation.data[0], 0, 0, 0, 0))

                if i % self.plot_update_interval == 0:
                    losses_train['netD_norm_w'] = self.get_model_norm(self.modelD[0])
                    losses_train['Gradient_Penalty'] = 0
                    losses_train['Learning_Rate_E'] = self.optimizerE.param_groups[0]['lr']
                    losses_train['Learning_Rate_G'] = self.optimizerG.param_groups[0]['lr']
                    losses_train['Learning_Rate_D'] = self.lr_dis
                    losses_train['Real'] = self.input.data.cpu()
                    losses_train['Fakes_Encoder'] = fake.data.cpu()
                    losses_train['Fakes_Previous'] = torch.zeros(fake.size())
                    losses_train['Fakes_Normal'] = self.modelG[0](self.fixed_noise).data.cpu()
                    # losses_train['Corr_Output'] = corr.unsqueeze(1).data.cpu()
                    # losses_train['Corr_Output'] = torch.zeros(fake.size())

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
                acc = 0.0
                # clamp parameters to a cube
                if not self.gp:
                    for p in self.modelD[0].parameters():
                        p.data.clamp_(self.clamp_lower, self.clamp_upper)

                input = data_iter.next()[0]
                i += 1
                batch_size = input.size(0)
                self.input.data.resize_(input.size()).copy_(input)
                self.modelD[0].zero_grad()

                # train with real
                if self.add_noise:
                    self.epsilon.data.resize_(self.input.size()).normal_(0, self.noise_var)
                    dis_input = self.input + self.epsilon
                    scores_D0 = self.modelD[0](dis_input)
                else:
                    scores_D0 = self.modelD[0](self.input)
                acc += torch.sum((scores_D0 > self.acc_margin).float())

                # train with fake
                self.epsilon.data.resize_(batch_size, self.nz).normal_(0, 1)
                noise_mu, noise_logvar = self.Encoder(self.input)
                noise_sigma = torch.exp(torch.mul(noise_logvar, 0.5))
                latents = noise_mu + torch.mul(noise_sigma, self.epsilon)
                latents = latents.detach()
                while(len(latents.size()) < len(self.input.size())):
                    latents = latents.unsqueeze(-1)

                fake = self.modelG[0](latents).detach()
                if self.add_noise:
                    self.epsilon.data.resize_(self.input.size()).normal_(0, self.noise_var)
                    dis_input = fake + self.epsilon
                    scores_D0_G0 = self.modelD[0](dis_input)
                else:
                    scores_D0_G0 = self.modelD[0](fake)
                acc += torch.sum((scores_D0_G0 <= 0).float())
                disc_acc = float(acc)*50 / batch_size

                errD0 = self.criterion.hinge_loss(scores_D0, 1)
                errD0 += self.criterion.hinge_loss(scores_D0_G0, -1)
                errD0 /= 2
                net_error = errD0
                gradient_penalty = self.t_zero
                # errD0 = self.criterion.rankerD([scores_D0, scores_D0_G0])
                if self.gp:
                    gradient_penalty = self.calc_gradient_penalty(self.input, fake, batch_size)
                    net_error = errD0 + gradient_penalty

                try:
                    net_error.backward()
                except Exception as e:
                    print(e)
                self.optimizerD.step()
                self.schedulerD.step(errD0.data[0])

                # Bookkeeping
                losses_train = {}
                losses_train['Loss_D0'] = errD0.data[0]
                losses_train['Loss_G0'] = 0
                losses_train['MSE'] = 0
                losses_train['KLD'] = 0
                losses_train['Corr_Loss'] = 0
                losses_train['Mean'] = noise_mu.mean().data[0]
                losses_train['Sigma'] = noise_sigma.mean().data[0]
                losses_train['Score_D0'] = scores_D0.median().data[0]
                losses_train['Score_D0_G0'] = scores_D0_G0.median().data[0]
                losses_train['Score_D0_G1'] = 0
                losses_train['Score_D0_Normal_G0'] = 0
                losses_train['Score_D0_Normal_G1'] = 0
                losses_train['Disc_Diff'] = 0
                losses_train['Disc_Acc'] = disc_acc
                losses_train['Clamp'] = self.clamp_upper
                self.monitor_train.update(losses_train, batch_size)
                print('Stage %d: [%d/%d][%d/%d] Loss_D0: %.3f Loss_G0: %.3f MSE: %.3f KLD: %.3f Score_D0: %.3f Score_D0_G0: %.3f Score_D0_G1: %.3f Disc_Diff %.3f Disc_Acc %.3f'
                    % (stage, epoch, self.stage_epochs, i, self.len_dataset, errD0.data[0], 0, 0, 0, scores_D0.median().data[0], scores_D0_G0.median().data[0], 0, 0, disc_acc))

                if i % self.plot_update_interval == 0:
                    losses_train['netD_norm_w'] = self.get_model_norm(self.modelD[0])[0]
                    losses_train['Learning_Rate_E'] = self.optimizerE.param_groups[0]['lr']
                    losses_train['Learning_Rate_G'] = self.optimizerG.param_groups[0]['lr']
                    losses_train['Learning_Rate_D'] = self.optimizerD.param_groups[0]['lr']
                    losses_train['Gradient_Penalty'] = gradient_penalty.data[0]
                    losses_train['Real'] = self.input.data.cpu()
                    losses_train['Fakes_Encoder'] = fake.data.cpu()
                    losses_train['Fakes_Previous'] = torch.zeros(fake.size())
                    losses_train['Fakes_Normal'] = self.modelG[0](self.fixed_noise).data.cpu()
                    # losses_train['Corr_Output'] = torch.zeros(fake.size())
                    self.visualizer_train.update(losses_train)

        else:
            ############################
            # Train GAN
            ############################
            # if epoch == 1:
            #     for param_group in self.optimizerD.param_groups:
            #         param_group['lr'] = 2*self.lr_dis

            i = 0
            avg_mean_scores_D0 = 0.0
            avg_mean_scores_D0_G0 = 0.0
            avg_mean_scores_D0_G1 = 0.0
            while i < self.len_dataset:
                ############################
                # Update Discriminator Network
                ############################
                # iter_start = time.time()
                self.modelD[0].train()
                # self.modelG[0].eval()
                # if self.giterations < 10 or self.giterations % (self.len_dataset/5) == 0:
                if epoch !=0 and self.giterations % (self.len_dataset/5) == 0:
                    d_iterations = self.len_dataset/10
                else:
                    d_iterations = self.d_iter

                j=0
                lossD = 0
                lossG = 0
                while j < d_iterations and i < self.len_dataset:
                    lossG = 0
                    acc = 0.0
                    j += 1
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
                    #     scores_D1 = self.modelD[1](dis_input, self.extra_layer-1)
                    # else:
                    #     scores_D1 = self.modelD[1](self.input, self.extra_layer-1)
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
                    #     scores_D1_G1 = self.modelD[1](dis_input, self.extra_layer-1)
                    # else:
                    #     scores_D1_G1 = self.modelD[1](fake_G1, self.extra_layer-1)
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
                    net_error = self.rank_error_weight*errD0 + self.disc_diff_weight*disc_abs_diff

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

                    if self.optimize_mse:
                        self.modelG[0].zero_grad()
                        latents.requires_grad = True
                        fake_G0 = self.modelG[0](latents)
                        loss_mse = self.criterion.diff_loss(fake_G0, self.input, type=self.vae_loss_type)
                        mse_error = self.weight_mse * loss_mse
                        try:
                            mse_error.backward()
                            self.optimizerG.step()
                        except Exception as e:
                            print(e)
                    else:
                        loss_mse = self.t_zero

                    self.extra_layer_gamma = min(self.extra_layer_gamma + 1/(2*self.len_dataset), 1)

                    mean_scores_D0 = scores_D0.median().data[0]
                    mean_scores_D0_G0 = scores_D0_G0.median().data[0]
                    mean_scores_D0_G1 = scores_D0_G1.median().data[0]
                    # avg_mean_scores_D0 = (avg_mean_scores_D0*(i-1) + mean_scores_D0) / i
                    # avg_mean_scores_D0_G0 = (avg_mean_scores_D0_G0*(i-1) + mean_scores_D0_G0) / i
                    # avg_mean_scores_D0_G1 = (avg_mean_scores_D0_G1*(i-1) + mean_scores_D0_G1) / i
                    avg_mean_scores_D0 = avg_mean_scores_D0 * 0.2 + mean_scores_D0 * 0.8
                    avg_mean_scores_D0_G0 = avg_mean_scores_D0_G0 * 0.2 + mean_scores_D0_G0 * 0.8
                    avg_mean_scores_D0_G1 = avg_mean_scores_D0_G1 * 0.2 + mean_scores_D0_G1 * 0.8

                    if self.adaptive_iter:
                        if avg_mean_scores_D0_G0 > avg_mean_scores_D0 - (avg_mean_scores_D0 - avg_mean_scores_D0_G1)/3:
                            self.d_iter += 1
                        # elif avg_mean_scores_D0_G0 <= avg_mean_scores_D0 - (avg_mean_scores_D0 - avg_mean_scores_D0_G1)/2:
                        else:
                            self.d_iter = self.d_iter_init

                    lossD = errD0.data[0]
                    if j < d_iterations:
                        # Bookkeeping
                        losses_train = {}
                        losses_train['Loss_D0'] = lossD
                        losses_train['Loss_G0'] = lossG
                        losses_train['MSE'] = loss_mse.data[0]
                        losses_train['KLD'] = 0
                        losses_train['Corr_Loss'] = 0
                        losses_train['Mean'] = noise_mu.mean().data[0]
                        losses_train['Sigma'] = noise_sigma.mean().data[0]
                        losses_train['Score_D0'] = mean_scores_D0
                        losses_train['Score_D0_G0'] = mean_scores_D0_G0
                        losses_train['Score_D0_G1'] = mean_scores_D0_G1
                        losses_train['Score_D0_Normal_G0'] = self.t_zero.data[0]
                        losses_train['Score_D0_Normal_G1'] = self.t_zero.data[0]
                        losses_train['Disc_Diff'] = disc_abs_diff.data[0]
                        losses_train['Disc_Acc'] = disc_acc
                        losses_train['Clamp'] = self.clamp_upper
                        self.monitor_train.update(losses_train, batch_size)
                        print('Stage %d: [%d/%d][%d/%d] Loss_D: %.3f Loss_G: %.3f MSE %.3f Score_D0: %.3f Score_D0_G0: %.3f Score_D0_G1: %.3f Disc_Diff %.3f Disc_Acc %.3f Gamma %.3f'
                        % (stage, epoch, self.stage_epochs, i, self.len_dataset, lossD, lossG, loss_mse.data[0], mean_scores_D0, mean_scores_D0_G0, mean_scores_D0_G1, disc_abs_diff.data[0], disc_acc, self.gen_gamma))

                        if i % self.plot_update_interval == 0:
                            losses_train['netD_norm_w'] = self.get_model_norm(self.modelD[0])[0]
                            losses_train['Learning_Rate_E'] = self.optimizerE.param_groups[0]['lr']
                            losses_train['Learning_Rate_G'] = self.optimizerG.param_groups[0]['lr']
                            losses_train['Learning_Rate_D'] = self.optimizerD.param_groups[0]['lr']
                            losses_train['Gradient_Penalty'] = gradient_penalty.data[0]
                            losses_train['Real'] = self.input.data.cpu()
                            losses_train['Fakes_Encoder'] = fake_G0.data.cpu()
                            losses_train['Fakes_Previous'] = fake_G1.data.cpu()
                            losses_train['Fakes_Normal'] = self.modelG[0](self.fixed_noise).data.cpu()
                            # losses_train['Corr_Output'] = torch.zeros(fake_G0.size())
                            self.visualizer_train.update(losses_train)
                            torch.save(self.modelD[0].state_dict(), '%s/stage_%d_epoch_%d_netD.pth' % (self.save_path, stage, 3*int(epoch/3)))


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

                    if self.add_noise:
                        self.epsilon.data.resize_(self.input.size()).normal_(0, self.noise_var)
                        dis_input = self.input + self.epsilon
                        scores_D0 = self.modelD[0](dis_input, self.extra_layer, self.extra_layer_gamma)
                    else:
                        scores_D0 = self.modelD[0](self.input, self.extra_layer, self.extra_layer_gamma)
                    fake_G1 = self.modelG[1](latents)
                    if self.add_noise:
                        self.epsilon.data.resize_(self.input.size()).normal_(0, self.noise_var)
                        dis_input = fake_G1 + self.epsilon
                        scores_D0_G1 = self.modelD[0](dis_input, self.extra_layer, self.extra_layer_gamma)
                    else:
                        scores_D0_G1 = self.modelD[0](fake_G1, self.extra_layer, self.extra_layer_gamma)
                    latents.requires_grad = True
                    fake_G0 = self.modelG[0](latents)
                    if self.add_noise:
                        self.epsilon.data.resize_(self.input.size()).normal_(0, self.noise_var)
                        dis_input = fake_G0 + self.epsilon
                        scores_D0_G0 = self.modelD[0](dis_input, self.extra_layer, self.extra_layer_gamma)
                    else:
                        scores_D0_G0 = self.modelD[0](fake_G0, self.extra_layer, self.extra_layer_gamma)

                    self.gen_gamma = min(self.gen_gamma + 1/(2*self.len_dataset), 1)
                    errG0 = self.criterion.rankerG([scores_D0, scores_D0_G0, scores_D0_G1])
                    net_error = self.gen_gamma * errG0
                    try:
                        net_error.backward()
                    except Exception as e:
                        print(e)
                    self.optimizerG.step()
                    self.giterations += 1

                    mean_scores_D0 = scores_D0.median().data[0]
                    mean_scores_D0_G0 = scores_D0_G0.median().data[0]
                    mean_scores_D0_G1 = scores_D0_G1.median().data[0]

                    avg_mean_scores_D0 = avg_mean_scores_D0 * 0.2 + mean_scores_D0 * 0.8
                    avg_mean_scores_D0_G0 = avg_mean_scores_D0_G0 * 0.2 + mean_scores_D0_G0 * 0.8
                    avg_mean_scores_D0_G1 = avg_mean_scores_D0_G1 * 0.2 + mean_scores_D0_G1 * 0.8
                    if self.gen_gamma > 0.5 and self.adaptive_iter:
                        if avg_mean_scores_D0_G0 < avg_mean_scores_D0 - 2*(avg_mean_scores_D0 - avg_mean_scores_D0_G1)/3:
                            self.g_iter += 1
                        # elif avg_mean_scores_D0_G0 >= avg_mean_scores_D0 - (avg_mean_scores_D0 - avg_mean_scores_D0_G1)/2:
                        else:
                            self.g_iter = self.g_iter_init

                    lossG = errG0.data[0]
                    if j < self.g_iter:
                        i += 1
                        # Bookkeeping
                        losses_train = {}
                        losses_train['Loss_D0'] = lossD
                        losses_train['Loss_G0'] = lossG
                        losses_train['MSE'] = loss_mse.data[0]
                        losses_train['KLD'] = 0
                        losses_train['Corr_Loss'] = 0
                        losses_train['Mean'] = noise_mu.mean().data[0]
                        losses_train['Sigma'] = noise_sigma.mean().data[0]
                        losses_train['Score_D0'] = mean_scores_D0
                        losses_train['Score_D0_G0'] = mean_scores_D0_G0
                        losses_train['Score_D0_G1'] = mean_scores_D0_G1
                        losses_train['Score_D0_Normal_G0'] = self.t_zero.data[0]
                        losses_train['Score_D0_Normal_G1'] = self.t_zero.data[0]
                        losses_train['Disc_Diff'] = disc_abs_diff.data[0]
                        losses_train['Disc_Acc'] = disc_acc
                        losses_train['Clamp'] = self.clamp_upper
                        self.monitor_train.update(losses_train, batch_size)
                        print('Stage %d: [%d/%d][%d/%d] Loss_D: %.3f Loss_G: %.3f MSE %.3f Score_D0: %.3f Score_D0_G0: %.3f Score_D0_G1: %.3f Disc_Diff %.3f Disc_Acc %.3f Gamma %.3f'
                        % (stage, epoch, self.stage_epochs, i, self.len_dataset, lossD, lossG, loss_mse.data[0], mean_scores_D0, mean_scores_D0_G0, mean_scores_D0_G1, disc_abs_diff.data[0], disc_acc, self.gen_gamma))

                        if i % self.plot_update_interval == 0:
                            losses_train['netD_norm_w'] = self.get_model_norm(self.modelD[0])[0]
                            losses_train['Learning_Rate_E'] = self.optimizerE.param_groups[0]['lr']
                            losses_train['Learning_Rate_G'] = self.optimizerG.param_groups[0]['lr']
                            losses_train['Learning_Rate_D'] = self.optimizerD.param_groups[0]['lr']
                            losses_train['Gradient_Penalty'] = gradient_penalty.data[0]
                            losses_train['Real'] = self.input.data.cpu()
                            losses_train['Fakes_Encoder'] = fake_G0.data.cpu()
                            losses_train['Fakes_Previous'] = fake_G1.data.cpu()
                            losses_train['Fakes_Normal'] = self.modelG[0](self.fixed_noise).data.cpu()
                            # losses_train['Corr_Output'] = torch.zeros(fake_G0.size())
                            self.visualizer_train.update(losses_train)

                        input = data_iter.next()[0]
                        batch_size = input.size(0)
                        self.input.data.resize_(input.size()).copy_(input)
                        self.epsilon.data.resize_(batch_size, self.nz).normal_(0, 1)
                        noise_mu, noise_logvar = self.Encoder(self.input)
                        noise_sigma = torch.exp(torch.mul(noise_logvar, 0.5))
                        latents = noise_mu + torch.mul(noise_sigma, self.epsilon)
                        latents = latents.detach()
                        while(len(latents.size()) < len(input.size())):
                            latents = latents.unsqueeze(-1)
                        latents = latents.detach()
                        lossD = 0

                # Bookkeeping
                losses_train = {}
                losses_train['Loss_D0'] = lossD
                losses_train['Loss_G0'] = lossG
                losses_train['MSE'] = loss_mse.data[0]
                losses_train['KLD'] = 0
                losses_train['Corr_Loss'] = 0
                losses_train['Mean'] = noise_mu.mean().data[0]
                losses_train['Sigma'] = noise_sigma.mean().data[0]
                losses_train['Score_D0'] = mean_scores_D0
                losses_train['Score_D0_G0'] = mean_scores_D0_G0
                losses_train['Score_D0_G1'] = mean_scores_D0_G1
                losses_train['Score_D0_Normal_G0'] = self.t_zero.data[0]
                losses_train['Score_D0_Normal_G1'] = self.t_zero.data[0]
                losses_train['Disc_Diff'] = disc_abs_diff.data[0]
                losses_train['Disc_Acc'] = disc_acc
                losses_train['Clamp'] = self.clamp_upper
                self.monitor_train.update(losses_train, batch_size)

                print('Stage %d: [%d/%d][%d/%d] Loss_D: %.3f Loss_G: %.3f MSE %.3f Score_D0: %.3f Score_D0_G0: %.3f Score_D0_G1: %.3f Disc_Diff %.3f Disc_Acc %.3f Gamma %.3f'
                        % (stage, epoch, self.stage_epochs, i, self.len_dataset, lossD, lossG, loss_mse.data[0], mean_scores_D0, mean_scores_D0_G0, mean_scores_D0_G1, disc_abs_diff.data[0], disc_acc, self.gen_gamma))

                if i % self.plot_update_interval == 0:
                    # Compute MSE
                    losses_train['netD_norm_w'] = self.get_model_norm(self.modelD[0])[0]
                    losses_train['Learning_Rate_E'] = self.optimizerE.param_groups[0]['lr']
                    losses_train['Learning_Rate_G'] = self.optimizerG.param_groups[0]['lr']
                    losses_train['Learning_Rate_D'] = self.optimizerD.param_groups[0]['lr']
                    losses_train['Gradient_Penalty'] = gradient_penalty.data[0]
                    losses_train['Real'] = self.input.data.cpu()
                    losses_train['Fakes_Encoder'] = fake_G0.data.cpu()
                    losses_train['Fakes_Previous'] = fake_G1.data.cpu()
                    losses_train['Fakes_Normal'] = self.modelG[0](self.fixed_noise).data.cpu()
                    # losses_train['Corr_Output'] = torch.zeros(fake_G0.size())
                    self.visualizer_train.update(losses_train)

                if i % 250 == 0 or i == self.len_dataset:
                    try:
                        fake_normal_z = self.modelG[0](self.fixed_noise)
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
                            vutils.save_image(fake_normal_z.data, '%s/fake_samples_stage_%03d_epoch_%03d_normal_z.png' % (self.save_path, stage, epoch), normalize=True)
                            vutils.save_image(fake_encoder_z.data, '%s/fake_samples_stage_%03d_epoch_%03d_encoder_z.png' % (self.save_path, stage, epoch), normalize=True)
                            vutils.save_image(input, '%s/real_samples.png' % self.save_path, normalize=True)
                    except Exception as e:
                        print(e)
                if True:
                    self.disc_diff_weight = self.disc_diff_weight_init
                elif disc_abs_diff.data[0] < 5:
                    self.disc_diff_weight = (self.marker_high - self.marker_low)*0.2 #self.disc_diff_weight_init/1000
                elif disc_abs_diff.data[0] < 10:
                    self.disc_diff_weight = (self.marker_high - self.marker_low)*10 #self.disc_diff_weight_init/500
                else:
                    self.disc_diff_weight = self.disc_diff_weight_init

            print('Stage %d: [%d/%d] Avg_Scores_D0 %.3f Avg_Scores_D0_G0 %.3f Avg_Scores_D0_G1 %.3f' % (stage, epoch, self.stage_epochs, avg_mean_scores_D0, avg_mean_scores_D0_G0, avg_mean_scores_D0_G1))

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
            test_scores['Test_Score_D0'] = score_D0.median().data[0]
            test_scores['Test_Score_D0_G0'] = score_D0_G0.median().data[0]
            test_scores['Test_Score_D0_G1'] = score_D0_G1.median().data[0]
            test_scores['SSIM'] = ssim_score/batch_size
            test_scores['PSNR'] = psnr_score/batch_size
            test_scores['Test_Disc_Acc'] = disc_acc
            self.monitor_test.update(test_scores, batch_size)
            print('Test: [%d/%d][%d/%d] Score_D0: %.3f Score_D0_G0: %.3f Score_D0_G1: %.3f SSIM: %.3f PSNR: %.3f Disc_Acc: %.3f'
                    % (epoch, self.stage_epochs, i, len(dataloader), score_D0.median().data[0], score_D0_G0.median().data[0], score_D0_G1.median().data[0], ssim_score/batch_size, psnr_score/batch_size, disc_acc))

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

            mean_scores_D0 = scores_D0.median().data[0]
            mean_scores_D0_G0 = scores_D0_G0.median().data[0]
            mean_scores_D0_G1 = scores_D0_G1.median().data[0]
            # avg_mean_scores_D0 = (avg_mean_scores_D0*(i-1) + mean_scores_D0) / i
            # avg_mean_scores_D0_G0 = (avg_mean_scores_D0_G0*(i-1) + mean_scores_D0_G0) / i
            # avg_mean_scores_D0_G1 = (avg_mean_scores_D0_G1*(i-1) + mean_scores_D0_G1) / i
            avg_mean_scores_D0 = avg_mean_scores_D0 * 0.2 + mean_scores_D0 * 0.8
            avg_mean_scores_D0_G0 = avg_mean_scores_D0_G0 * 0.2 + mean_scores_D0_G0 * 0.8
            avg_mean_scores_D0_G1 = avg_mean_scores_D0_G1 * 0.2 + mean_scores_D0_G1 * 0.8

            if avg_mean_scores_D0_G0 < (avg_mean_scores_D0 + avg_mean_scores_D0_G1)/2 and disc_abs_diff.data[0] < 5:
                return False

            # Bookkeeping
            losses_train = {}
            losses_train['Loss_D0'] = errD0.data[0]
            losses_train['Loss_G0'] = 0
            losses_train['MSE'] = 0
            losses_train['KLD'] = 0
            losses_train['Corr_Loss'] = 0
            losses_train['Mean'] = noise_mu.mean().data[0]
            losses_train['Sigma'] = noise_sigma.mean().data[0]
            losses_train['Score_D0'] = mean_scores_D0
            losses_train['Score_D0_G0'] = mean_scores_D0_G0
            losses_train['Score_D0_G1'] = mean_scores_D0_G1
            losses_train['Score_D0_Normal_G0'] = self.t_zero.data[0]
            losses_train['Score_D0_Normal_G1'] = self.t_zero.data[0]
            losses_train['Disc_Diff'] = disc_abs_diff.data[0]
            losses_train['Disc_Acc'] = disc_acc
            losses_train['Clamp'] = self.clamp_upper
            self.monitor_train.update(losses_train, batch_size)
            print('Stage %d: [%d/%d][%d/%d] Loss_D: %.3f Loss_G: %.3f Score_D0: %.3f Score_D0_G0: %.3f Score_D0_G1: %.3f Disc_Diff %.3f Disc_Acc %.3f Extra_Gamma %.3f'
                % (stage, epoch, self.stage_epochs, i, self.len_dataset, errD0.data[0], 0, mean_scores_D0, mean_scores_D0_G0, mean_scores_D0_G1, disc_abs_diff.data[0], disc_acc, self.extra_layer_gamma))

            if i % self.plot_update_interval == 0:
                # Compute MSE
                loss_mse = self.criterion.diff_loss(fake_G0, self.input, type=self.vae_loss_type)
                losses_train['MSE'] = loss_mse.data[0]
                losses_train['netD_norm_w'] = self.get_model_norm(self.modelD[0])[0]
                losses_train['Learning_Rate_E'] = self.optimizerE.param_groups[0]['lr']
                losses_train['Learning_Rate_G'] = self.optimizerG.param_groups[0]['lr']
                losses_train['Learning_Rate_D'] = self.optimizerD.param_groups[0]['lr']
                losses_train['Gradient_Penalty'] = gradient_penalty.data[0]
                losses_train['Real'] = self.input.data.cpu()
                losses_train['Fakes_Encoder'] = fake_G0.data.cpu()
                losses_train['Fakes_Previous'] = fake_G1.data.cpu()
                losses_train['Fakes_Normal'] = self.modelG[0](self.fixed_noise).data.cpu()
                # losses_train['Corr_Output'] = torch.zeros(fake_G0.size())
                self.visualizer_train.update(losses_train)

            if disc_abs_diff.data[0] < 4:
                self.disc_diff_weight = self.disc_diff_weight_init/1000
            elif disc_abs_diff.data[0] < 10:
                self.disc_diff_weight = self.disc_diff_weight_init/10
            else:
                self.disc_diff_weight = self.disc_diff_weight_init

        print('Stage %d: [%d/%d] Avg_Scores_D0 %.3f Avg_Scores_D0_G0 %.3f Avg_Scores_D0_G1 %.3f' % (stage, epoch, self.stage_epochs, avg_mean_scores_D0, avg_mean_scores_D0_G0, avg_mean_scores_D0_G1))
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
