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

import plugins

class Trainer():
    def __init__(self, args, modelD, modelG, Encoder, criterion):

        self.args = args
        self.modelD = [modelD for i in range(2)]
        self.modelG = [modelG for i in range(2)]
        self.Encoder = Encoder
        self.criterion = criterion

        self.port = args.port
        self.env = args.env
        self.dir_save = args.save
        self.dataset_fraction = args.dataset_fraction
        self.len_dataset = 0

        self.cuda = args.cuda
        self.nepochs = args.nepochs
        self.stage_epochs = args.stage_epochs
        self.nchannels = args.nchannels
        self.batch_size = args.batch_size
        self.resolution_high = args.resolution_high
        self.resolution_wide = args.resolution_wide
        self.nz = args.nz
        self.gp = args.gp
        self.gp_lambda = args.gp_lambda
        self.scheduler_patience = args.scheduler_patience

        self.weight_gan_final = args.weight_gan_final
        self.weight_vae_init = args.weight_vae_init
        self.weight_kld = args.weight_kld
        self.margin = args.margin
        self.disc_diff_weight = args.disc_diff_weight
        self.num_stages = args.num_stages

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
        self.schedulerE = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerE, factor=self.lr_decay, min_lr=1e-3*self.lr_vae)
        self.schedulerG = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, factor=self.lr_decay, min_lr=1e-3*self.lr_vae)
        self.schedulerD = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, factor=self.lr_decay, min_lr=1e-3*self.lr_vae)

        # logging training
        self.log_loss_train = plugins.Logger(args.logs, 'TrainLogger.txt')
        self.params_loss_train = ['Loss_D0', 'Loss_G0', 'MSE', 'KLD', 'Score_D0', 'Score_D1', 'Score_D0_G0', 'Score_D0_G1', 'Score_D1_G1', 'Disc_Difference']
        self.log_loss_train.register(self.params_loss_train)

        self.log_loss_test = plugins.Logger(args.logs, 'TestLogger.txt')
        self.params_loss_test = ['SSIM', 'PSNR', 'Test_Score_D0', 'Test_Score_D0_G0', 'Test_Score_D0_G1', 'Test_Disc_Accuracy']
        self.log_loss_test.register(self.params_loss_test)

        # monitor training
        self.monitor_train = plugins.Monitor()
        self.params_monitor_train = ['Loss_D0', 'Loss_G0', 'MSE', 'KLD', 'Score_D0', 'Score_D1', 'Score_D0_G0', 'Score_D0_G1', 'Score_D1_G1', 'Disc_Difference']
        self.monitor_train.register(self.params_monitor_train)

        self.monitor_test = plugins.Monitor()
        self.params_monitor_test = ['SSIM', 'PSNR', 'Test_Score_D0', 'Test_Score_D0_G0', 'Test_Score_D0_G1', 'Test_Disc_Accuracy']
        self.monitor_test.register(self.params_monitor_test)

        # Define visualizer plot type for given dataset
        if args.net_type == 'gmm':
            if self.args.gmm_dim == 1:
                output_dtype, output_vtype = 'vector', 'histogram'
            elif self.args.gmm_dim == 2:
                output_dtype, output_vtype = 'heatmap', 'heatmap'
        else:
            output_dtype, output_vtype = 'images', 'images'
            self.fixed_noise = self.fixed_noise.unsqueeze(-1).unsqueeze(-1)

        # visualize training
        self.visualizer_train = plugins.HourGlassVisualizer(port=self.port, env=self.env, title='Train')
        self.params_visualizer_train = {
        'Loss_D0':{'dtype':'scalar', 'vtype':'plot', 'win': 'loss_gan', 'layout': {'windows': ['Loss_D0', 'Loss_G0'], 'id': 0}},
        'Loss_G0':{'dtype':'scalar','vtype':'plot', 'win': 'loss_gan', 'layout': {'windows': ['Loss_D0', 'Loss_G0'], 'id': 1}},
        'MSE':{'dtype':'scalar','vtype':'plot', 'win': 'enc_losses', 'layout': {'windows': ['MSE', 'KLD'], 'id': 0}},
        'KLD':{'dtype':'scalar','vtype':'plot', 'win': 'enc_losses', 'layout': {'windows': ['MSE', 'KLD'], 'id': 1}},
        'Mean':{'dtype':'scalar','vtype':'plot', 'win': 'norm_params', 'layout': {'windows': ['Mean', 'Sigma'], 'id': 0}},
        'Sigma':{'dtype':'scalar','vtype':'plot', 'win': 'norm_params', 'layout': {'windows': ['Mean', 'Sigma'], 'id': 1}},
        'Score_D0':{'dtype':'scalar','vtype':'plot', 'win': 'loss_D0', 'layout': {'windows': ['Score_D0', 'Score_D0_G0', 'Score_D0_G1'], 'id': 0}},
        'Score_D0_G0':{'dtype':'scalar','vtype':'plot', 'win': 'loss_D0', 'layout': {'windows': ['Score_D0', 'Score_D0_G0', 'Score_D0_G1'], 'id': 1}},
        'Score_D0_G1':{'dtype':'scalar','vtype':'plot', 'win': 'loss_D0', 'layout': {'windows': ['Score_D0', 'Score_D0_G0', 'Score_D0_G1'], 'id': 2}},
        'Score_D1':{'dtype':'scalar','vtype':'plot', 'win': 'loss_D1', 'layout': {'windows': ['Score_D1', 'Score_D1_G1'], 'id': 0}},
        'Score_D1_G1':{'dtype':'scalar','vtype':'plot', 'win': 'loss_D1', 'layout': {'windows': ['Score_D1', 'Score_D1_G1'], 'id': 1}},
        'netD_norm_w':{'dtype':'scalar','vtype':'plot', 'win': 'netd_norm'},
        'LR_E':{'dtype':'scalar','vtype':'plot', 'win': 'lr', 'layout': {'windows': ['LR_E', 'LR_G', 'LR_D'], 'id': 0}},
        'LR_G':{'dtype':'scalar','vtype':'plot', 'win': 'lr', 'layout': {'windows': ['LR_E', 'LR_G', 'LR_D'], 'id': 1}},
        'LR_D':{'dtype':'scalar','vtype':'plot', 'win': 'lr', 'layout': {'windows': ['LR_E', 'LR_G', 'LR_D'], 'id': 2}},
        'Gradient_Penalty':{'dtype':'scalar','vtype':'plot', 'win': 'gp'},
        'Disc_Difference':{'dtype':'scalar','vtype':'plot', 'win': 'disc_diff'},
        'Disc_Accuracy':{'dtype':'scalar','vtype':'plot', 'win': 'disc_acc'},
        'Real': {'dtype': output_dtype, 'vtype': output_vtype, 'win': 'real'},
        'Fakes_Encoder': {'dtype': output_dtype, 'vtype': output_vtype, 'win': 'fakes_enc'},
        'Fakes_Normal': {'dtype': output_dtype, 'vtype': output_vtype, 'win': 'fakes_normal'},
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
        'Test_Disc_Accuracy': {'dtype':'scalar','vtype':'plot', 'win': 'test_disc_acc'},
        }
        self.visualizer_test.register(self.params_visualizer_test)

        # display training progress
        self.print_train = '[%d/%d][%d/%d] '
        for item in self.params_loss_train:
            self.print_train = self.print_train + item + " %.3f "

        self.print_test = '[%d/%d][%d/%d] '
        for item in self.params_loss_test:
            self.print_test = self.print_test + item + " %.3f "

        self.giterations = 0
        self.d_iter = args.d_iter
        self.g_iter = args.g_iter
        self.clamp_lower = args.clamp_lower
        self.clamp_upper = args.clamp_upper
        print('Discriminator:', self.modelD[0])
        print('Generator:', self.modelG[0])
        print('Encoder:', self.Encoder)

        # define a zero tensor
        self.t_zero = Variable(torch.zeros(1))

    def initialize_optimizer(self, model, lr, optim_method='RMSprop', weight_decay=None):
        if weight_decay is None:
            weight_decay = self.weight_decay
        if optim_method == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(self.adam_beta1, self.adam_beta2), weight_decay=weight_decay)
        elif optim_method == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=self.momentum, weight_decay=weight_decay)
        else:
            raise(Exception("Unknown Optimization Method"))
        return optimizer

    def model_train(self):
        self.modelD[0].train()
        self.modelG[0].train()
        self.Encoder.train()

    def setup_stage(self, stage):
        if stage == 0:
            # VAE
            self.weight_vae, self.weight_gan = self.weight_vae_init, 0
            self.optimizerE = self.initialize_optimizer(self.Encoder, lr=self.lr_vae, optim_method='Adam')
            self.optimizerG = self.initialize_optimizer(self.modelG[0], lr=self.lr_vae, optim_method='Adam')
            self.schedulerE = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerE, factor=self.lr_decay, patience=self.scheduler_patience, min_lr=1e-3*self.lr_vae)
            self.schedulerG = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, factor=self.lr_decay, patience=self.scheduler_patience, min_lr=1e-3*self.lr_vae)
        elif stage == 1:
            # Margin GAN
            self.weight_vae, self.weight_gan = 0, self.weight_gan_final
            self.optimizerD = self.initialize_optimizer(self.modelD[0], lr=self.lr_vae, optim_method='RMSprop', weight_decay=self.args.stage1_weight_decay)
            self.optimizerG = self.initialize_optimizer(self.modelG[0], lr=self.lr_gen, optim_method='RMSprop')
            self.schedulerG = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, factor=self.lr_decay, patience=self.scheduler_patience, min_lr=1e-3*self.lr_vae)
            self.schedulerD = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, factor=self.lr_decay, patience=self.scheduler_patience, min_lr=1e-3*self.lr_vae)
            self.criterion.set_margin(self.margin)
            print("Margin: {}".format(self.margin))
        else:
            # GoGAN
            self.optimizerD = self.initialize_optimizer(self.modelD[0], lr=self.lr_dis, optim_method='RMSprop')
            self.optimizerG = self.initialize_optimizer(self.modelG[0], lr=self.lr_gen, optim_method='RMSprop')
            self.weight_vae, self.weight_gan = 0, self.weight_gan_final
            self.criterion.set_margin(self.margin)

            # Initialize previous stage models and set them to eval mode
            self.modelD[1] = copy.deepcopy(self.modelD[0])
            self.modelD[1].eval()
            self.modelG[1] = copy.deepcopy(self.modelG[0])
            self.modelG[1].eval()
            self.Encoder.eval()
            for p in self.modelD[1].parameters():
                p.requires_grad = False
            for p in self.modelG[1].parameters():
                p.requires_grad = False
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
        disc_interpolates = self.modelD[0](interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).cuda() if self.cuda else torch.ones(
                                      disc_interpolates.size()), create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gp_lambda
        return gradient_penalty

    def train(self, stage, epoch, dataloader):
        self.monitor_train.reset()
        data_iter = iter(dataloader)
        self.len_dataset = int(len(dataloader) * self.dataset_fraction)
        if epoch == 0:
            print("Length of Dataset: {}".format(self.len_dataset))

        if epoch % 2 == 0 and self.args.net_type == 'gmm':
            self.visualizer_train.reset('Real')
            self.visualizer_train.reset('Fakes_Encoder')
            self.visualizer_train.reset('Fakes_Normal')

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
                loss_kld = self.criterion.kl_divergence(noise_mu, noise_logvar)
                net_error = self.weight_vae * loss_mse + self.weight_kld * loss_kld
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
                losses_train['Score_D0'] = 0
                losses_train['Score_D1'] = 0
                losses_train['Score_D0_G0'] = 0
                losses_train['Score_D0_G1'] = 0
                losses_train['Score_D1_G1'] = 0
                losses_train['Disc_Difference'] = 0
                self.monitor_train.update(losses_train, batch_size)
                print('Stage %d: [%d/%d][%d/%d] Loss_D0: %.3f Loss_G0: %.3f MSE: %.3f KLD: %.3f Score_D0: %.3f Score_D0_G0: %.3f Score_D0_G1: %.3f'
                    % (stage, epoch, self.stage_epochs, i, self.len_dataset, 0, 0, loss_mse.data[0], loss_kld.data[0], 0, 0, 0))

                if i % 30 == 0:
                    losses_train['Mean'] = noise_mu.mean().data[0]
                    losses_train['Sigma'] = noise_sigma.mean().data[0]
                    losses_train['Disc_Accuracy'] = 0
                    losses_train['netD_norm_w'] = self.get_model_norm(self.modelD[0])
                    losses_train['Gradient_Penalty'] = 0
                    losses_train['LR_E'] = self.optimizerE.param_groups[0]['lr']
                    losses_train['LR_G'] = self.optimizerG.param_groups[0]['lr']
                    losses_train['LR_D'] = self.lr_dis
                    losses_train['Real'] = self.input.data.cpu()
                    losses_train['Fakes_Encoder'] = fake.data.cpu()
                    losses_train['Fakes_Normal'] = self.modelG[0](self.fixed_noise).data.cpu()

                    self.visualizer_train.update(losses_train)

                if i % 250 == 0:
                    try:
                        fake_normal_z = self.modelG[0](self.fixed_noise)
                        fake_encoder_z = self.modelG[0](latents)
                        if len(self.input.size()) < 3:
                            fig = plt.figure()
                            plt.hist(self.input.squeeze().data.cpu().numpy(), bins=60)
                            fig.savefig("{}/stage_{}_epoch_{}_real.png".format(self.dir_save, stage, i), dpi=fig.dpi)
                            fig = plt.figure()
                            plt.hist(fake_encoder_z.squeeze().data.cpu().numpy(), bins=60)
                            fig.savefig("{}/stage_{}_epoch_{}_fake_enc.png".format(self.dir_save, stage, i), dpi=fig.dpi)
                            fig = plt.figure()
                            plt.hist(fake_normal_z.squeeze().data.cpu().numpy(), bins=60)
                            fig.savefig("{}/stage_{}_epoch_{}_fake_norm.png".format(self.dir_save, stage, i), dpi=fig.dpi)
                        else:
                            vutils.save_image(fake_normal_z.data, '%s/fake_samples_stage_%03d_normal_z.png' % (self.args.save, stage), normalize=True)
                            vutils.save_image(fake_encoder_z.data, '%s/fake_samples_stage_%03d_encoder_z.png' % (self.args.save, stage), normalize=True)
                            vutils.save_image(input, '%s/real_samples.png' % self.args.save, normalize=True)
                    except Exception as e:
                        print(e)
        elif stage == 1:
            ############################
            # Train Margin GAN
            ############################
            i = 0
            while i < self.len_dataset:
                ############################
                # Update Discriminator Network
                ############################
                acc = 0.0
                self.modelD[0].train()
                self.modelG[0].eval()
                if epoch < 3:
                    d_iterations = 1000
                elif self.giterations % (self.len_dataset/5) == 0:
                    d_iterations = 100
                else:
                    d_iterations = self.d_iter

                j=0
                while j < d_iterations and i < self.len_dataset:
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

                    # train with real
                    scores_D0 = self.modelD[0](self.input)
                    acc += torch.sum((scores_D0 > self.margin/2).float())

                    # train with fake
                    self.epsilon.data.resize_(batch_size, self.nz).normal_(0, 1)
                    noise_mu, noise_logvar = self.Encoder(self.input)
                    noise_sigma = torch.exp(torch.mul(noise_logvar, 0.5))
                    latents = noise_mu + torch.mul(noise_sigma, self.epsilon)
                    latents = latents.detach()
                    while(len(latents.size()) < len(self.input.size())):
                        latents = latents.unsqueeze(-1)

                    fake_G0 = self.modelG[0](latents).detach()
                    fake_G0.requires_grad = True
                    scores_D0_G0 = self.modelD[0](fake_G0)
                    acc += torch.sum((scores_D0_G0 <= self.margin/2).float())
                    disc_acc = float(acc)*50 / batch_size

                    # Compute loss and do backward()
                    errD0 = self.criterion.hinge_loss(scores_D0, 1)
                    errD0 += self.criterion.hinge_loss(scores_D0_G0, -1)
                    errD0 /= 2

                    if self.gp:
                        gradient_penalty = self.calc_gradient_penalty(self.input, fake_G0, batch_size)
                        net_error = errD0 + gradient_penalty
                    else:
                        net_error = errD0
                        gradient_penalty = self.t_zero
                    try:
                        net_error.backward()
                    except Exception as e:
                        print(e)

                    self.optimizerD.step()

                ############################
                # Update Generator Network
                ############################
                self.modelD[0].eval()
                self.modelG[0].train()
                self.modelG[0].zero_grad()

                fake_G0 = self.modelG[0](latents)
                scores_D0_G0 = self.modelD[0](fake_G0)

                errG0 = self.criterion.generator_margin_loss(scores_D0_G0)
                try:
                    errG0.backward()
                except Exception as e:
                    print(e)
                self.optimizerG.step()
                self.giterations += 1

                # Bookkeeping
                losses_train = {}
                losses_train['Loss_D0'] = errD0.data[0]
                losses_train['Loss_G0'] = errG0.data[0]
                losses_train['MSE'] = 0
                losses_train['KLD'] = 0
                losses_train['Score_D0'] = scores_D0.mean().data[0]
                losses_train['Score_D1'] = 0
                losses_train['Score_D0_G0'] = scores_D0_G0.mean().data[0]
                losses_train['Score_D0_G1'] = 0
                losses_train['Score_D1_G1'] = 0
                losses_train['Disc_Difference'] = 0
                self.monitor_train.update(losses_train, batch_size)
                print('Stage %d: [%d/%d][%d/%d] Loss_D: %.3f Loss_G: %.3f Score_D0: %.3f Score_D0_G0: %.3f Score_D0_G1: %.3f Disc_Difference %.3f'
                    % (stage, epoch, self.stage_epochs, i, self.len_dataset, errD0.data[0], errG0.data[0], scores_D0.mean().data[0], scores_D0_G0.mean().data[0], 0, 0))

                if i % 30 == 0:
                    losses_train['Mean'] = noise_mu.mean().data[0]
                    losses_train['Sigma'] = noise_sigma.mean().data[0]
                    losses_train['Disc_Accuracy'] = disc_acc
                    losses_train['netD_norm_w'] = self.get_model_norm(self.modelD[0])
                    losses_train['Gradient_Penalty'] = gradient_penalty.data[0]
                    losses_train['LR_E'] = self.optimizerE.param_groups[0]['lr']
                    losses_train['LR_G'] = self.optimizerG.param_groups[0]['lr']
                    losses_train['LR_D'] = self.lr_dis
                    losses_train['Real'] = self.input.data.cpu()
                    losses_train['Fakes_Encoder'] = fake_G0.data.cpu()
                    losses_train['Fakes_Normal'] = self.modelG[0](self.fixed_noise).data.cpu()
                    self.visualizer_train.update(losses_train)

                if i % 250 == 0:
                    try:
                        fake_encoder_z = fake_G0
                        fake_normal_z = self.modelG[0](self.fixed_noise)
                        if len(self.input.size()) < 3:
                            fig = plt.figure()
                            plt.hist(input.squeeze().cpu().numpy(), bins=60)
                            fig.savefig("{}/stage_{}_epoch_{}_real.png".format(self.dir_save, stage, i), dpi=fig.dpi)
                            fig = plt.figure()
                            plt.hist(fake_encoder_z.squeeze().data.cpu().numpy(), bins=60)
                            fig.savefig("{}/stage_{}_epoch_{}_fake_enc.png".format(self.dir_save, stage, i), dpi=fig.dpi)
                            fig = plt.figure()
                            plt.hist(fake_normal_z.squeeze().data.cpu().numpy(), bins=60)
                            fig.savefig("{}/stage_{}_epoch_{}_fake_norm.png".format(self.dir_save, stage, i), dpi=fig.dpi)
                        else:
                            vutils.save_image(fake_normal_z.data, '%s/fake_samples_stage_%03d_normal_z.png' % (self.args.save, stage), normalize=True)
                            vutils.save_image(fake_encoder_z.data, '%s/fake_samples_stage_%03d_encoder_z.png' % (self.args.save, stage), normalize=True)
                            vutils.save_image(input, '%s/real_samples.png' % self.args.save, normalize=True)
                    except Exception as e:
                        print(e)

        else:
            ############################
            # Train GAN
            ############################
            i = 0
            while i < self.len_dataset:
                ############################
                # Update Discriminator Network
                ############################
                iter_start = time.time()
                self.modelD[0].train()
                self.modelG[0].eval()
                if self.giterations % (self.len_dataset/5) == 0:
                    d_iterations = 100
                else:
                    d_iterations = self.d_iter

                j=0
                while j < d_iterations and i < self.len_dataset:
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
                    scores_D0 = self.modelD[0](self.input)
                    scores_D1 = self.modelD[1](self.input)

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
                    scores_D0_G0 = self.modelD[0](fake_G0)
                    fake_G1 = self.modelG[1](latents).detach()
                    scores_D1_G1 = self.modelD[1](fake_G1)
                    fake_G1.requires_grad = True
                    scores_D0_G1 = self.modelD[0](fake_G1)

                    # Compute loss and do backward()
                    disc_abs_diff = torch.norm(scores_D0 - scores_D1) + torch.norm(scores_D0_G1 - scores_D1_G1)
                    errD0 = self.criterion.rankerD([scores_D0, scores_D0_G0])
                    net_error = errD0 + self.disc_diff_weight*disc_abs_diff

                    if self.gp:
                        gradient_penalty = self.calc_gradient_penalty(self.input, fake_G0, batch_size)
                        net_error += gradient_penalty

                    try:
                        net_error.backward()
                    except Exception as e:
                        print(e)

                    self.optimizerD.step()

                ############################
                # Update Generator Network
                ############################
                self.modelD[0].eval()
                self.modelG[0].train()
                self.modelG[0].zero_grad()
                self.modelD[0].zero_grad()

                scores_D0 = self.modelD[0](self.input)
                fake_G1 = self.modelG[1](latents)
                scores_D0_G1 = self.modelD[0](fake_G1)
                latents.requires_grad = True
                fake_G0 = self.modelG[0](latents)
                scores_D0_G0 = self.modelD[0](fake_G0)

                errG0 = self.criterion.rankerG([scores_D0, scores_D0_G0, scores_D0_G1])
                try:
                    errG0.backward()
                except Exception as e:
                        print(e)
                self.optimizerG.step()
                self.giterations += 1

                # Bookkeeping
                losses_train = {}
                losses_train['Loss_D0'] = errD0.data[0]
                losses_train['Loss_G0'] = errG0.data[0]
                losses_train['MSE'] = 0
                losses_train['KLD'] = 0
                losses_train['Mean'] = noise_mu.mean().data[0]
                losses_train['Sigma'] = noise_sigma.mean().data[0]
                losses_train['Score_D0'] = scores_D0.mean().data[0]
                losses_train['Score_D1'] = scores_D1.mean().data[0]
                losses_train['Score_D0_G0'] = scores_D0_G0.mean().data[0]
                losses_train['Score_D0_G1'] = scores_D0_G1.mean().data[0]
                losses_train['Score_D1_G1'] = scores_D1_G1.mean().data[0]
                losses_train['Score_D0_Normal_G0'] = self.t_zero.data[0]
                losses_train['Score_D0_Normal_G1'] = self.t_zero.data[0]
                losses_train['Disc_Difference'] = disc_abs_diff.data[0]
                self.monitor_train.update(losses_train, batch_size)
                print('Stage %d: [%d/%d][%d/%d] Loss_D: %.3f Loss_G: %.3f MSE: %.3f KLD: %.3f Score_D0: %.3f Score_D0_G0: %.3f Score_D0_G1: %.3f Disc_Difference %.3f'
                    % (stage, epoch, self.stage_epochs, i, self.len_dataset, errD0.data[0], errG0.data[0], 0, 0, scores_D0.mean().data[0], scores_D0_G0.mean().data[0], scores_D0_G1.mean().data[0], disc_abs_diff.data[0]))

                if i % 30 == 0:
                    # Compute MSE
                    diff = fake_G0 - self.input
                    loss_mse = (0.5*torch.mul(diff, diff)).mean(0).sum()
                    losses_train['Disc_Accuracy'] = 0
                    losses_train['MSE'] = loss_mse.data[0]
                    losses_train['netD_norm_w'] = self.get_model_norm(self.modelD[0])
                    losses_train['LR_E'] = self.optimizerE.param_groups[0]['lr']
                    losses_train['LR_G'] = self.optimizerG.param_groups[0]['lr']
                    losses_train['LR_D'] = self.optimizerD.param_groups[0]['lr']
                    losses_train['Gradient_Penalty'] = gradient_penalty.data[0]
                    losses_train['Real'] = self.input.data.cpu()
                    losses_train['Fakes_Encoder'] = fake_G0.data.cpu()
                    losses_train['Fakes_Normal'] = self.modelG[0](self.fixed_noise).data.cpu()
                    self.visualizer_train.update(losses_train)

                if i % 250 == 0:
                    try:
                        fake_normal_z = self.modelG[0](self.fixed_noise)
                        fake_encoder_z = fake_G0
                        if len(self.input.size()) < 3:
                            fig = plt.figure()
                            plt.hist(input.squeeze().cpu().numpy(), bins=60)
                            fig.savefig("{}/stage_{}_epoch_{}_real.png".format(self.dir_save, stage, i), dpi=fig.dpi)
                            fig = plt.figure()
                            plt.hist(fake_encoder_z.squeeze().data.cpu().numpy(), bins=60)
                            fig.savefig("{}/stage_{}_epoch_{}_fake_enc.png".format(self.dir_save, stage, i), dpi=fig.dpi)
                            fig = plt.figure()
                            plt.hist(fake_normal_z.squeeze().data.cpu().numpy(), bins=60)
                            fig.savefig("{}/stage_{}_epoch_{}_fake_norm.png".format(self.dir_save, stage, i), dpi=fig.dpi)
                        else:
                            vutils.save_image(fake_normal_z.data, '%s/fake_samples_stage_%03d_normal_z.png' % (self.args.save, stage), normalize=True)
                            vutils.save_image(fake_encoder_z.data, '%s/fake_samples_stage_%03d_encoder_z.png' % (self.args.save, stage), normalize=True)
                            vutils.save_image(input, '%s/real_samples.png' % self.args.save, normalize=True)
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
        self.modelG[0].eval()
        self.modelG[1].eval()
        self.modelD[0].eval()
        self.modelD[1].eval()
        self.Encoder.eval()
        self.t_zero = Variable(torch.zeros(1))

        epoch_score_D0 = 0.0
        epoch_score_D0_G0 = 0.0
        epoch_score_D0_G1 = 0.0
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
            score_D0 = self.modelD[0](self.test_input)
            score_D0_G0 = self.modelD[0](fake_G0)
            score_D0_G1 = self.modelD[0](fake_G1)
            ssim_score, psnr_score = 0.0, 0.0
            data_range = input.max() - input.min()

            acc += torch.sum((score_D0 > self.margin/2).float())
            acc += torch.sum((score_D0_G0 <= self.margin/2).float())
            disc_acc = float(acc)*50/batch_size

            if self.args.net_type != 'gmm':
                compare_real = input.permute(0,2,3,1)
                compare_fake = fake_G0.permute(0,2,3,1)
                for j in range(batch_size):
                    ssim_score += compare_ssim(compare_real[i,...].cpu().numpy(), compare_fake[i,...].data.cpu().numpy(), data_range=data_range, multichannel=True)
                    psnr_score += compare_psnr(compare_real[i,...].cpu().numpy(), compare_fake[i,...].data.cpu().numpy(), data_range=data_range)

            if i <= num_batches-1:
                epoch_score_D0 += score_D0.mean().data[0]
                epoch_score_D0_G0 += score_D0_G0.mean().data[0]
                epoch_score_D0_G1 += score_D0_G1.mean().data[0]
                epoch_ssim_score += ssim_score
                epoch_psnr_score += psnr_score
                epoch_disc_acc += disc_acc


            # Bookkeeping
            test_scores = {}
            test_scores['Test_Score_D0'] = score_D0.mean().data[0]
            test_scores['Test_Score_D0_G0'] = score_D0_G0.mean().data[0]
            test_scores['Test_Score_D0_G1'] = score_D0_G1.mean().data[0]
            test_scores['SSIM'] = ssim_score/batch_size
            test_scores['PSNR'] = psnr_score/batch_size
            test_scores['Test_Disc_Accuracy'] = disc_acc
            self.monitor_test.update(test_scores, batch_size)
            print('Test: [%d/%d][%d/%d] Score_D0: %.3f Score_D0_G0: %.3f Score_D0_G1: %.3f SSIM: %.3f PSNR: %.3f Disc_Accuracy: %.3f'
                    % (epoch, self.stage_epochs, i, len(dataloader), score_D0.mean().data[0], score_D0_G0.mean().data[0], score_D0_G1.mean().data[0], ssim_score, psnr_score, disc_acc))

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
                        fig.savefig("{}/stage_{}_epoch_{}_real.png".format(self.dir_save, stage, i), dpi=fig.dpi)
                        fig = plt.figure()
                        plt.hist(fake_encoder_z.squeeze().data.cpu().numpy(), bins=60)
                        fig.savefig("{}/stage_{}_epoch_{}_fake_enc.png".format(self.dir_save, stage, i), dpi=fig.dpi)
                        fig = plt.figure()
                        plt.hist(fake_normal_z.squeeze().data.cpu().numpy(), bins=60)
                        fig.savefig("{}/stage_{}_epoch_{}_fake_norm.png".format(self.dir_save, stage, i), dpi=fig.dpi)
                    else:
                        vutils.save_image(fake_encoder_z.data, '%s/val_fake_samples_stage_%03d_encoder_z.png' % (self.args.save, stage), normalize=True)
                        vutils.save_image(input, '%s/val_real_samples.png' % self.args.save, normalize=True)
                except Exception as e:
                    print(e)

        test_scores = {}
        test_scores['Test_Score_D0'] = epoch_score_D0 / (num_batches-1)
        test_scores['Test_Score_D0_G0'] = epoch_score_D0_G0 / (num_batches-1)
        test_scores['Test_Score_D0_G1'] = epoch_score_D0_G1 / (num_batches-1)
        test_scores['SSIM'] = epoch_ssim_score/(num_batches-1)
        test_scores['PSNR'] = epoch_psnr_score/(num_batches-1)
        test_scores['Test_Disc_Accuracy'] = epoch_disc_acc / (num_batches-1)
        test_scores['Test_Real'] = self.test_input.data.cpu()
        test_scores['Test_Fakes_Encoder'] = fake_G0.data.cpu()
        self.visualizer_test.update(test_scores)
        try:
            loss = self.monitor_test.getvalues()
            self.log_loss_test.update(loss)
        except Exception as e:
            print("Error while logging test loss")
            print(e)
        return epoch_disc_acc / (num_batches-1)

    def get_model_norm(self, model):
        norm = torch.zeros(1)
        param_list = list(model.main.parameters())
        for l in param_list:
            norm += torch.norm(l.data)
        return norm
