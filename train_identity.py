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
from models import incep_resnetV1, incep_resnetV1_multigpu
from models import Hopenet

class Trainer():
    def __init__(self, args, modelD, modelG, Encoder, criterion, prevD, prevG):

        self.args = args
        self.modelD = [modelD for i in range(2)]
        self.modelG = [modelG for i in range(2)]
        self.prevD = prevD
        self.prevG = prevG
        self.criterion = criterion
        self.plot_update_interval = args.plot_update_interval
        self.ngpu = args.ngpu
        self.cuda = args.cuda
        self.device = torch.device("cuda" if (self.cuda and torch.cuda.is_available()) else "cpu")
        self.facenet = incep_resnetV1_multigpu(3, 128, 0, self.ngpu)
        self.facenet = self.facenet.to(self.device)
        facenet_model = torch.load('models/facenet.pth')
        self.facenet.load_state_dict(facenet_model)
        self.identity_lambda = args.identity_lambda
        self.cos_sim = torch.nn.CosineSimilarity(dim=1)
        self.pose_bin = 66
        self.hopenet = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], self.pose_bin).to(self.device)
        hopenet_model = torch.load('models/hopenet_robust_alpha1.pkl')
        self.hopenet.load_state_dict(hopenet_model)
        self.pose_lambda = args.pose_lambda

        self.port = args.port
        self.env = args.env
        self.dir_save = args.save
        self.dataset_fraction = args.dataset_fraction
        self.len_dataset = 0
        self.wgan = args.wgan

        self.nepochs = args.nepochs
        self.stage_epochs = args.stage_epochs
        self.start_stage = args.start_stage
        self.nchannels = args.nchannels
        self.batch_size = args.batch_size
        self.resolution_high = args.resolution_high
        self.resolution_wide = args.resolution_wide
        self.nz = args.nz
        self.noise_z = self.nz - 128 - 3*66
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

        self.lr_dis = args.learning_rate_dis
        self.lr_gen = args.learning_rate_gen
        self.lr_decay = args.learning_rate_decay
        self.momentum = args.momentum
        self.adam_beta1 = args.adam_beta1
        self.adam_beta2 = args.adam_beta2
        self.weight_decay = args.weight_decay
        self.optim_method = args.optim_method

        # for classification
        self.test_input = torch.FloatTensor(self.batch_size,self.nchannels,self.resolution_high,self.resolution_wide).to(self.device)
        self.noise = torch.FloatTensor(self.batch_size, self.noise_z).normal_(0, 1).to(self.device)
        self.fixed_noise = torch.FloatTensor(self.batch_size, self.nz).normal_(0, 1).to(self.device)
        self.epsilon = torch.randn(self.batch_size, self.nz).to(self.device)

        # Initialize optimizer
        self.optimizerG = self.initialize_optimizer(self.modelG[0], lr=self.lr_gen, optim_method='RMSprop')
        self.optimizerD = self.initialize_optimizer(self.modelD[0], lr=self.lr_dis, optim_method='RMSprop', weight_decay=1e-2*self.lr_dis)
        self.schedulerG = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerG, factor=self.lr_decay, patience=self.scheduler_patience , min_lr=1e-3*self.lr_gen)
        self.schedulerD = optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD, factor=self.lr_decay, patience=self.scheduler_patience , min_lr=1e-3*self.lr_dis)

        # logging training
        self.log_loss_train = plugins.Logger(args.logs, 'TrainLogger.txt')
        self.params_loss_train = ['Loss_D0', 'Loss_G0', 'Score_D0', 'Score_D1', 'Score_D0_G0', 'Score_D0_G1', 'Score_D1_G1', 'Disc_Diff', 'Disc_Acc', 'Iden_Loss', 'Pose_Loss']
        self.log_loss_train.register(self.params_loss_train)

        # self.log_loss_test = plugins.Logger(args.logs, 'TestLogger.txt')
        # self.params_loss_test = ['SSIM', 'PSNR', 'Test_Score_D0', 'Test_Score_D0_G0', 'Test_Score_D0_G1', 'Test_Disc_Acc']
        # self.log_loss_test.register(self.params_loss_test)

        # monitor training
        self.monitor_train = plugins.Monitor()
        self.params_monitor_train = ['Loss_D0', 'Loss_G0', 'Score_D0', 'Score_D1', 'Score_D0_G0', 'Score_D0_G1', 'Score_D1_G1', 'Disc_Diff', 'Disc_Acc', 'Iden_Loss', 'Pose_Loss']
        self.monitor_train.register(self.params_monitor_train)

        # self.monitor_test = plugins.Monitor()
        # self.params_monitor_test = ['SSIM', 'PSNR', 'Test_Score_D0', 'Test_Score_D0_G0', 'Test_Score_D0_G1', 'Test_Disc_Acc']
        # self.monitor_test.register(self.params_monitor_test)

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
        'Score_D0':{'dtype':'scalar','vtype':'plot', 'win': 'loss_D0', 'layout': {'windows': ['Score_D0', 'Score_D0_G0', 'Score_D0_G1'], 'id': 0}},
        'Score_D0_G0':{'dtype':'scalar','vtype':'plot', 'win': 'loss_D0', 'layout': {'windows': ['Score_D0', 'Score_D0_G0', 'Score_D0_G1'], 'id': 1}},
        'Score_D0_G1':{'dtype':'scalar','vtype':'plot', 'win': 'loss_D0', 'layout': {'windows': ['Score_D0', 'Score_D0_G0', 'Score_D0_G1'], 'id': 2}},
        # 'Score_D1':{'dtype':'scalar','vtype':'plot', 'win': 'loss_D1', 'layout': {'windows': ['Score_D1', 'Score_D1_G1'], 'id': 0}},
        # 'Score_D1_G1':{'dtype':'scalar','vtype':'plot', 'win': 'loss_D1', 'layout': {'windows': ['Score_D1', 'Score_D1_G1'], 'id': 1}},
        'netD_norm_w':{'dtype':'scalar','vtype':'plot', 'win': 'netd_norm'},
        'Gradient_Penalty':{'dtype':'scalar','vtype':'plot', 'win': 'gp'},
        'Disc_Diff':{'dtype':'scalar','vtype':'plot', 'win': 'disc_diff'},
        'Disc_Acc':{'dtype':'scalar','vtype':'plot', 'win': 'disc_acc'},
        'Iden_Loss':{'dtype':'scalar','vtype':'plot', 'win': 'iden_loss'},
        'Pose_Loss':{'dtype':'scalar','vtype':'plot', 'win': 'pose_loss'},

        'Real': {'dtype': output_dtype, 'vtype': output_vtype, 'win': 'real'},
        'Fakes': {'dtype': output_dtype, 'vtype': output_vtype, 'win': 'fakes_normal'},
        'Fakes_Previous': {'dtype': output_dtype, 'vtype': output_vtype, 'win': 'fakes_prev'},
        }
        self.visualizer_train.register(self.params_visualizer_train)

        # self.visualizer_test = plugins.HourGlassVisualizer(port=self.port, env=self.env, title='Test')
        # self.params_visualizer_test = {
        # 'Test_Score_D0':{'dtype':'scalar', 'vtype':'plot', 'win':'test_disc_scores', 'layout':{'windows': ['Test_Score_D0', 'Test_Score_D0_G0', 'Test_Score_D0_G1'], 'id':0}},
        # 'Test_Score_D0_G0':{'dtype':'scalar', 'vtype':'plot', 'win':'test_disc_scores', 'layout':{'windows': ['Test_Score_D0', 'Test_Score_D0_G0', 'Test_Score_D0_G1'], 'id':1}},
        # 'Test_Score_D0_G1':{'dtype':'scalar', 'vtype':'plot', 'win':'test_disc_scores', 'layout':{'windows': ['Test_Score_D0', 'Test_Score_D0_G0', 'Test_Score_D0_G1'], 'id':2}},
        # 'SSIM':{'dtype':'scalar', 'vtype':'plot', 'win':'ssim_scores'},
        # 'PSNR':{'dtype':'scalar', 'vtype':'plot', 'win':'psnr_scores'},
        # 'Test_Real': {'dtype': output_dtype, 'vtype': output_vtype, 'win': 'real_test'},
        # 'Test_Fakes': {'dtype': output_dtype, 'vtype': output_vtype, 'win': 'fakes_enc_test'},
        # 'Test_Disc_Acc': {'dtype':'scalar','vtype':'plot', 'win': 'test_disc_acc'},
        # }
        # self.visualizer_test.register(self.params_visualizer_test)

        # display training progress
        self.print_train = '[%d/%d][%d/%d] '
        for item in self.params_loss_train:
            self.print_train = self.print_train + item + " %.3f "

        # self.print_test = '[%d/%d][%d/%d] '
        # for item in self.params_loss_test:
        #     self.print_test = self.print_test + item + " %.3f "

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

        # define a zero tensor
        self.t_zero = torch.zeros(1)
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

    def setup_stage(self, stage, dataloader=None):
        if stage == 0:
            self.optimizerD = self.initialize_optimizer(self.modelD[0], lr=self.lr_dis, optim_method=self.optim_method, weight_decay=1e-2*self.lr_dis)
            self.optimizerG = self.initialize_optimizer(self.modelG[0], lr=self.lr_gen, optim_method=self.optim_method)
            # self.margin *= 2
            self.acc_margin = 0
        else:
            if stage > self.start_stage:
                self.gen_gamma = 0
            if not self.gogan_optim_set:
                self.optimizerD = self.initialize_optimizer(self.modelD[0], lr=self.lr_dis, optim_method=self.optim_method, weight_decay=1e-2*self.lr_dis)
                self.optimizerG = self.initialize_optimizer(self.modelG[0], lr=self.lr_gen, optim_method=self.optim_method)
                self.gogan_optim_set = True
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

            # self.modelD[1].eval()
            # self.modelG[1].eval()
            for p in self.modelD[1].parameters():
                p.requires_grad = False
            for p in self.modelG[1].parameters():
                p.requires_grad = False

            self.marker_high, self.marker_low = self.compute_markers(dataloader)
            print("Marker High: {}, Marker Low: {}".format(self.marker_high, self.marker_low))
            self.acc_margin = (self.marker_high + self.marker_low)/2
            # print("Margin: {}".format(self.margin))
            # self.margin = self.margin / 2.0

    def calc_gradient_penalty(self, real_data, fake_data, batch_size):
        alpha = torch.rand(batch_size, 1)
        while len(alpha.size()) < len(real_data.size()):
            alpha = alpha.unsqueeze(-1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(self.device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates.requires_grad = True

        disc_interpolates = self.modelD[0](interpolates, self.extra_layer, self.extra_layer_gamma)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(batch_size, -1)
        # gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        gradient_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = ((gradient_norm -1) ** 2).mean()
        return gradient_penalty

    def train(self, stage, epoch, dataloader):
        self.monitor_train.reset()
        data_iter = iter(dataloader)
        self.len_dataset = int(len(dataloader) * self.dataset_fraction)
        if epoch == 0:
            print("Length of Dataset: {}".format(self.len_dataset))

        if stage == 0:
            ############################
            # Train Margin GAN/WGAN
            ############################
            i = 0
            avg_mean_scores_D0 = 0.0
            avg_mean_scores_D0_G0 = 0.0
            avg_mean_scores_D0_G1 = 0.0
            feat_loss = torch.zeros(1)
            pose_loss = torch.zeros(1)
            lossG = 0
            while i < self.len_dataset:
                ############################
                # Update Discriminator Network
                ############################
                self.modelD[0].train()
                # self.modelG[0].eval()
                if epoch !=0 and self.giterations % (self.len_dataset/5) == 0:
                    d_iterations = self.len_dataset/10
                else:
                    d_iterations = self.d_iter

                j=0
                lossD = 0
                while j < d_iterations and i < self.len_dataset:
                    # lossG = 0
                    acc = 0.0
                    j += 1
                    i += 1

                    # clamp parameters to a cube
                    if not self.gp:
                        for p in self.modelD[0].parameters():
                            p.data.clamp_(self.clamp_lower, self.clamp_upper)

                    input = data_iter.next()[0].to(self.device)
                    batch_size = input.size(0)

                    self.modelD[0].zero_grad()
                    self.modelG[0].zero_grad()

                    # train with real
                    if self.add_noise:
                        self.epsilon.data.resize_(input.size()).normal_(0, self.noise_var)
                        dis_input = self.input + self.epsilon
                        scores_D0 = self.modelD[0](dis_input, self.extra_layer, self.extra_layer_gamma)
                    else:
                        scores_D0 = self.modelD[0](input, self.extra_layer, self.extra_layer_gamma)
                    acc += torch.sum((scores_D0 > self.acc_margin).float())

                    # train with fake
                    input_x = torch.nn.functional.upsample(input, size=160, mode='bilinear')
                    input_iden = self.facenet(input_x)[0].detach()
                    input_iden = (input_iden - input_iden.mean(1, keepdim=True)) / input_iden.std(1, keepdim=True)
                    input_pose = self.hopenet(input_x)
                    input_pose = torch.cat(input_pose, dim=1).detach()
                    input_pose = (input_pose - input_pose.mean(1, keepdim=True)) / input_pose.std(1, keepdim=True)

                    self.noise.resize_(batch_size, self.noise_z, 1, 1).normal_(0, 1)
                    latent = torch.cat((self.noise, input_iden.unsqueeze(-1).unsqueeze(-1), input_pose.unsqueeze(-1).unsqueeze(-1)), 1)

                    fake_G0 = self.modelG[0](latent).detach()
                    if self.add_noise:
                        self.epsilon.data.resize_(self.input.size()).normal_(0, self.noise_var)
                        dis_input = fake_G0 + self.epsilon
                        scores_D0_G0 = self.modelD[0](dis_input, self.extra_layer, self.extra_layer_gamma)
                    else:
                        scores_D0_G0 = self.modelD[0](fake_G0, self.extra_layer, self.extra_layer_gamma)
                    acc += torch.sum((scores_D0_G0 <= 0).float())
                    disc_acc = float(acc)*50 / batch_size

                    self.noise.resize_(batch_size, self.nz, 1, 1).normal_(0, 1)
                    fake_G1 = self.modelG[0](self.noise).detach()
                    if self.add_noise:
                        self.epsilon.data.resize_(self.input.size()).normal_(0, self.noise_var)
                        dis_input = fake_G0 + self.epsilon
                        scores_D0_G1 = self.modelD[0](dis_input, self.extra_layer, self.extra_layer_gamma)
                    else:
                        scores_D0_G1 = self.modelD[0](fake_G1, self.extra_layer, self.extra_layer_gamma)
                    acc += torch.sum((scores_D0_G1 <= 0).float())
                    disc_acc = float(acc)*100 / (3*batch_size)

                    # Compute loss and do backward()
                    if self.wgan:
                        errD0 = ((scores_D0_G0 + scores_D0_G1)/2 - scores_D0).mean()
                    else:
                        errD0 = self.criterion.hinge_loss(scores_D0, 1)
                        errD0 += self.criterion.hinge_loss(scores_D0_G0, -1)
                        errD0 /= 2
                    # self.schedulerD.step(errD0.data[0])

                    if self.gp:
                        gradient_penalty = self.calc_gradient_penalty(input, fake_G0, batch_size)
                        net_error = errD0 + self.gp_lambda * gradient_penalty
                    else:
                        net_error = errD0
                        gradient_penalty = self.t_zero

                    try:
                        net_error.backward()
                    except Exception as e:
                        print(e)

                    self.optimizerD.step()
                    self.extra_layer_gamma = min(self.extra_layer_gamma + 1/(2*self.len_dataset), 1)

                    mean_scores_D0 = scores_D0.median().item()
                    mean_scores_D0_G0 = scores_D0_G0.median().item()

                    lossD = errD0.item()
                    if j < d_iterations:
                        # Bookkeeping
                        losses_train = {}
                        losses_train['Loss_D0'] = lossD
                        losses_train['Loss_G0'] = lossG
                        losses_train['Score_D0'] = mean_scores_D0
                        losses_train['Score_D0_G0'] = mean_scores_D0_G0
                        losses_train['Score_D0_G1'] = 0
                        losses_train['Disc_Diff'] = 0
                        losses_train['Disc_Acc'] = disc_acc
                        # losses_train['Clamp'] = self.clamp_upper
                        losses_train['Score_D0_G1'] = 0
                        losses_train['Iden_Loss'] = feat_loss.item()
                        losses_train['Pose_Loss'] = pose_loss.item()
                        # losses_train['Score_D1_G1'] = 0
                        # losses_train['Score_D1'] = 0
                        self.monitor_train.update(losses_train, batch_size)
                        print('Stage %d: [%d/%d][%d/%d] Loss_D: %.3f Loss_G: %.3f Score_D0: %.3f Score_D0_G0: %.3f Disc_Acc %.3f GP %.3f Iden_Loss %.3f Pose_Loss %.3f'
                        % (stage, epoch, self.stage_epochs[stage], i, self.len_dataset, lossD, lossG, mean_scores_D0, mean_scores_D0_G0, disc_acc, gradient_penalty.item(), feat_loss.item(), pose_loss.item()))

                        if i % self.plot_update_interval == 0:
                            # Compute MSE
                            losses_train['netD_norm_w'] = self.get_model_norm(self.modelD[0])[0]
                            losses_train['Gradient_Penalty'] = gradient_penalty.item()
                            losses_train['Real'] = input.cpu()
                            losses_train['Fakes'] = fake_G0.detach().cpu()
                            losses_train['Fakes_Previous'] = torch.zeros(fake_G0.size())
                            self.visualizer_train.update(losses_train)

                ############################
                # Update Generator Network
                ############################
                # self.modelD[0].eval()
                self.modelG[0].train()
                self.modelG[0].zero_grad()

                fake_G1 = self.modelG[0](self.noise)
                if self.add_noise:
                    self.epsilon.data.resize_(self.input.size()).normal_(0, self.noise_var)
                    dis_input = fake_G0 + self.epsilon
                    scores_D0_G1 = self.modelD[0](dis_input, self.extra_layer, self.extra_layer_gamma)
                else:
                    scores_D0_G1 = self.modelD[0](fake_G1, self.extra_layer, self.extra_layer_gamma)

                fake_G0 = self.modelG[0](latent)
                if self.add_noise:
                    self.epsilon.data.resize_(self.input.size()).normal_(0, self.noise_var)
                    dis_input = fake_G0 + self.epsilon
                    scores_D0_G0 = self.modelD[0](dis_input, self.extra_layer, self.extra_layer_gamma)
                else:
                    scores_D0_G0 = self.modelD[0](fake_G0, self.extra_layer, self.extra_layer_gamma)

                fake_x = torch.nn.functional.upsample(fake_G0, size=160, mode='bilinear')
                fake_iden = self.facenet(fake_x)[0]
                fake_iden = (fake_iden - fake_iden.mean(1, keepdim=True)) / fake_iden.std(1, keepdim=True)
                # feat_loss = torch.norm(input_iden - fake_iden, dim=1).mean()
                feat_loss = 1 - self.cos_sim(input_iden, fake_iden).mean()
                fake_pose = self.hopenet(fake_x)
                fake_pose = torch.cat(fake_pose, dim=1)
                fake_pose = (fake_pose - fake_pose.mean(1, keepdim=True)) / fake_pose.std(1, keepdim=True)
                pose_loss = torch.norm(input_pose - fake_pose, dim=1).mean()

                lossG = (- scores_D0_G0.mean() - scores_D0_G1.mean())/2
                if self.wgan:
                    if epoch < 0:
                        errG0 = lossG
                    else:
                        errG0 = lossG + self.identity_lambda*feat_loss + self.pose_lambda*pose_loss
                else:
                    errG0 = self.criterion.hinge_loss(scores_D0_G0, 1) + self.identity_lambda*feat_loss + self.pose_lambda*pose_loss

                try:
                    errG0.backward()
                except Exception as e:
                    print(e)
                self.optimizerG.step()
                self.giterations += 1

                # Bookkeeping
                losses_train = {}
                losses_train['Loss_D0'] = lossD
                losses_train['Loss_G0'] = lossG.item()
                losses_train['Score_D0'] = mean_scores_D0
                # losses_train['Score_D1'] = 0
                losses_train['Score_D0_G0'] = mean_scores_D0_G0
                losses_train['Score_D0_G1'] = 0
                # losses_train['Score_D1_G1'] = 0
                losses_train['Disc_Diff'] = 0
                losses_train['Disc_Acc'] = disc_acc
                # losses_train['Clamp'] = self.clamp_upper
                losses_train['Iden_Loss'] = feat_loss.item()
                losses_train['Pose_Loss'] = pose_loss.item()
                self.monitor_train.update(losses_train, batch_size)
                print('Stage %d: [%d/%d][%d/%d] Loss_D: %.3f Loss_G: %.3f Score_D0: %.3f Score_D0_G0: %.3f Disc_Acc %.3f GP %.3f Iden_Loss %.3f Pose_Loss %.3f'
                % (stage, epoch, self.stage_epochs[stage], i, self.len_dataset, lossD, lossG.item(), mean_scores_D0, mean_scores_D0_G0, disc_acc, gradient_penalty.item(), feat_loss.item(), pose_loss.item()))

                if i % self.plot_update_interval == 0:
                    losses_train['netD_norm_w'] = self.get_model_norm(self.modelD[0])[0]
                    losses_train['Gradient_Penalty'] = gradient_penalty.item()
                    losses_train['Real'] = input.cpu()
                    losses_train['Fakes'] = fake_G0.detach().cpu()
                    losses_train['Fakes_Previous'] = torch.zeros(fake_G0.size())
                    self.visualizer_train.update(losses_train)

                if i % 250 == 0:
                    try:
                        fake_normal_z = self.modelG[0](self.fixed_noise)
                        if len(input.size()) < 3:
                            fig = plt.figure()
                            plt.hist(input.squeeze().cpu().numpy(), bins=60)
                            fig.savefig("{}/stage_{}_epoch_{}_real.png".format(self.dir_save, stage, i), dpi=fig.dpi)
                            fig = plt.figure()
                            plt.hist(fake_normal_z.squeeze().data.cpu().numpy(), bins=60)
                            fig.savefig("{}/stage_{}_epoch_{}_fake_norm.png".format(self.dir_save, stage, i), dpi=fig.dpi)
                        else:
                            vutils.save_image(fake_normal_z.data, '%s/fake_samples_stage_%03d_epoch_%03d_normal_z.png' % (self.args.save, stage, epoch), normalize=True)
                            vutils.save_image(input, '%s/real_samples.png' % self.args.save, normalize=True)
                    except Exception as e:
                        print(e)

                if epoch == self.stage_epochs[stage]-1:
                    self.margin = (scores_D0.mean() - scores_D0_G0.mean()).data

        else:
            ############################
            # Train GAN
            ############################
            i = 0
            avg_mean_scores_D0 = 0.0
            avg_mean_scores_D0_G0 = 0.0
            avg_mean_scores_D0_G1 = 0.0
            feat_loss = torch.zeros(1)
            pose_loss = torch.zeros(1)
            lossD = torch.zeros(1)
            lossG = torch.zeros(1)
            while i < self.len_dataset:
                ############################
                # Update Discriminator Network
                ############################
                self.modelD[0].train()
                # self.modelG[0].eval()
                if epoch !=0 and self.giterations % (self.len_dataset/5) == 0:
                    d_iterations = self.len_dataset/10
                else:
                    d_iterations = self.d_iter

                j=0
                while j < d_iterations and i < self.len_dataset:
                    acc = 0.0
                    j += 1
                    i += 1

                    # clamp parameters to a cube
                    if not self.gp:
                        for p in self.modelD[0].parameters():
                            p.data.clamp_(self.clamp_lower, self.clamp_upper)

                    input = data_iter.next()[0].to(self.device)
                    batch_size = input.size(0)

                    self.modelD[0].zero_grad()
                    self.modelG[0].zero_grad()

                    # train with real
                    if self.add_noise:
                        self.epsilon.data.resize_(input.size()).normal_(0, self.noise_var)
                        dis_input = input + self.epsilon
                        scores_D0 = self.modelD[0](dis_input, self.extra_layer, self.extra_layer_gamma)
                    else:
                        scores_D0 = self.modelD[0](input, self.extra_layer, self.extra_layer_gamma)
                    acc += torch.sum((scores_D0 > self.acc_margin).float())

                    # train with conditioned fake
                    with torch.no_grad():
                        input_x = torch.nn.functional.upsample(input, size=160, mode='bilinear')
                        input_iden = self.facenet(input_x)[0].detach()
                        input_iden = (input_iden - input_iden.mean(1, keepdim=True)) / input_iden.std(1, keepdim=True)
                        input_pose = self.hopenet(input_x)
                        input_pose = torch.cat(input_pose, dim=1).detach()
                        input_pose = (input_pose - input_pose.mean(1, keepdim=True)) / input_pose.std(1, keepdim=True)

                    self.noise.resize_(batch_size, self.noise_z, 1, 1).normal_(0, 1)
                    latent = torch.cat((self.noise, input_iden.unsqueeze(-1).unsqueeze(-1), input_pose.unsqueeze(-1).unsqueeze(-1)), 1)
                    with torch.no_grad():
                        fake_G0 = self.modelG[0](latent).detach()
                    if self.add_noise:
                        self.epsilon.data.resize_(input.size()).normal_(0, self.noise_var)
                        dis_input = fake_G0 + self.epsilon
                        scores_D0_G0 = self.modelD[0](dis_input, self.extra_layer, self.extra_layer_gamma)
                    else:
                        scores_D0_G0 = self.modelD[0](fake_G0, self.extra_layer, self.extra_layer_gamma)
                    acc += torch.sum((scores_D0_G0 <= self.acc_margin).float())

                    with torch.no_grad():
                        fake_G1 = self.modelG[1](latent).detach()
                    if self.add_noise:
                        self.epsilon.data.resize_(input.size()).normal_(0, self.noise_var)
                        dis_input = fake_G1 + self.epsilon
                        scores_D0_G1 = self.modelD[0](dis_input, self.extra_layer, self.extra_layer_gamma)
                    else:
                        scores_D0_G1 = self.modelD[0](fake_G1, self.extra_layer, self.extra_layer_gamma)

                    # # train with fake from random
                    # self.noise.data.resize_(batch_size, self.nz, 1, 1).normal_(0, 1)
                    # with torch.no_grad():
                    #     fake_G0_random = self.modelG[0](self.noise).detach()
                    # fake_G0_random.requires_grad = True
                    # if self.add_noise:
                    #     self.epsilon.data.resize_(input.size()).normal_(0, self.noise_var)
                    #     dis_input = fake_G0_random + self.epsilon
                    #     scores_D0_G0_random = self.modelD[0](dis_input, self.extra_layer, self.extra_layer_gamma)
                    # else:
                    #     scores_D0_G0_random = self.modelD[0](fake_G0_random, self.extra_layer, self.extra_layer_gamma)

                    # with torch.no_grad():
                    #     fake_G1_random = self.modelG[1](self.noise).detach()
                    # fake_G1_random.requires_grad = True
                    # if self.add_noise:
                    #     self.epsilon.data.resize_(input.size()).normal_(0, self.noise_var)
                    #     dis_input = fake_G1_random + self.epsilon
                    #     scores_D0_G1_random = self.modelD[0](dis_input, self.extra_layer, self.extra_layer_gamma)
                    # else:
                    #     scores_D0_G1_random = self.modelD[0](fake_G1_random, self.extra_layer, self.extra_layer_gamma)
                    # acc += torch.sum((scores_D0_G0_random <= self.acc_margin).float())
                    disc_acc = float(acc)*100 / (2*batch_size)

                    # Compute loss and do backward()
                    disc_abs_diff = self.criterion.hinge_loss(scores_D0, 1, self.marker_high)
                    disc_abs_diff += self.criterion.hinge_loss(scores_D0_G1, -1, -self.marker_low)
                    errD0 = self.criterion.rankerD([scores_D0, scores_D0_G0, scores_D0_G1])
                    # errD0 += self.criterion.rankerD([scores_D0, scores_D0_G0_random, scores_D0_G1_random])
                    # net_error = self.rank_error_weight*(errD0) + self.disc_diff_weight*disc_abs_diff
                    net_error = ((scores_D0_G1 + scores_D0_G0)/2 - scores_D0).mean() + errD0

                    if self.gp:
                        gradient_penalty = self.calc_gradient_penalty(input, fake_G0, batch_size)
                        net_error += self.gp_lambda * gradient_penalty
                    else:
                        gradient_penalty = self.t_zero

                    net_error.backward()
                    self.optimizerD.step()
                    self.extra_layer_gamma = min(self.extra_layer_gamma + 1/(2*self.len_dataset), 1)

                    mean_scores_D0 = scores_D0.median().item()
                    mean_scores_D0_G0 = scores_D0_G0.median().item()
                    mean_scores_D0_G1 = scores_D0_G1.median().item()
                    avg_mean_scores_D0 = avg_mean_scores_D0 * 0.2 + mean_scores_D0 * 0.8
                    avg_mean_scores_D0_G0 = avg_mean_scores_D0_G0 * 0.2 + mean_scores_D0_G0 * 0.8
                    avg_mean_scores_D0_G1 = avg_mean_scores_D0_G1 * 0.2 + mean_scores_D0_G1 * 0.8
                    # if self.adaptive_iter:
                    #     if avg_mean_scores_D0_G0 > avg_mean_scores_D0 - (avg_mean_scores_D0 - avg_mean_scores_D0_G1)/3:
                    #         self.d_iter += 1
                    #     else:
                    #         self.d_iter = self.d_iter_init

                    lossD = errD0.item()#/2
                    if j < d_iterations:
                        # Bookkeeping
                        losses_train = {}
                        losses_train['Loss_D0'] = lossD
                        losses_train['Loss_G0'] = lossG
                        losses_train['Score_D0'] = mean_scores_D0
                        losses_train['Score_D0_G0'] = mean_scores_D0_G0
                        losses_train['Score_D0_G1'] = mean_scores_D0_G1
                        losses_train['Disc_Diff'] = disc_abs_diff.item()
                        losses_train['Disc_Acc'] = disc_acc
                        losses_train['Iden_Loss'] = feat_loss.item()
                        losses_train['Pose_Loss'] = pose_loss.item()
                        # losses_train['Clamp'] = self.clamp_upper
                        self.monitor_train.update(losses_train, batch_size)
                        print('Stage %d: [%d/%d][%d/%d] Loss_D: %.3f Loss_G: %.3f Score_D0: %.3f Score_D0_G0: %.3f Score_D0_G1: %.3f Disc_Acc %.3f Disc_Diff %.3f GP %.3f Iden_Loss %.3f Pose_Loss %.3f'
                        % (stage, epoch, self.stage_epochs[stage], i, self.len_dataset, lossD, lossG, mean_scores_D0, mean_scores_D0_G0, mean_scores_D0_G1, disc_acc, disc_abs_diff.item(), gradient_penalty.item(), feat_loss.item(), pose_loss.item()))

                        if i % self.plot_update_interval == 0:
                            losses_train['netD_norm_w'] = self.get_model_norm(self.modelD[0])[0]
                            losses_train['Gradient_Penalty'] = gradient_penalty.item()
                            losses_train['Real'] = input.cpu()
                            losses_train['Fakes'] = fake_G0.detach().cpu()
                            losses_train['Fakes_Previous'] = fake_G1.detach().cpu()
                            self.visualizer_train.update(losses_train)

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
                    net_error = 0

                    # if self.add_noise:
                    #     self.epsilon.data.resize_(input.size()).normal_(0, self.noise_var)
                    #     dis_input = input + self.epsilon
                    #     scores_D0 = self.modelD[0](dis_input, self.extra_layer, self.extra_layer_gamma)
                    # else:
                    #     scores_D0 = self.modelD[0](input, self.extra_layer, self.extra_layer_gamma)

                    # fake_G0_random = self.modelG[0](self.noise)
                    # if self.add_noise:
                    #     self.epsilon.data.resize_(input.size()).normal_(0, self.noise_var)
                    #     dis_input = fake_G0_random + self.epsilon
                    #     scores_D0_G0_random = self.modelD[0](dis_input, self.extra_layer, self.extra_layer_gamma)
                    # else:
                    #     scores_D0_G0_random = self.modelD[0](fake_G0_random, self.extra_layer, self.extra_layer_gamma)

                    fake_G0 = self.modelG[0](latent)
                    if self.add_noise:
                        self.epsilon.data.resize_(input.size()).normal_(0, self.noise_var)
                        dis_input = fake_G0 + self.epsilon
                        scores_D0_G0 = self.modelD[0](dis_input, self.extra_layer, self.extra_layer_gamma)
                    else:
                        scores_D0_G0 = self.modelD[0](fake_G0, self.extra_layer, self.extra_layer_gamma)

                    fake_x = torch.nn.functional.upsample(fake_G0, size=160, mode='bilinear')
                    fake_iden = self.facenet(fake_x)[0]
                    fake_iden = (fake_iden - fake_iden.mean(1, keepdim=True)) / fake_iden.std(1, keepdim=True)
                    # feat_loss = torch.norm(input_iden - fake_iden, dim=1).mean()
                    feat_loss = 1 - self.cos_sim(input_iden, fake_iden).mean()
                    fake_pose = self.hopenet(fake_x)
                    fake_pose = torch.cat(fake_pose, dim=1)
                    fake_pose = (fake_pose - fake_pose.mean(1, keepdim=True)) / fake_pose.std(1, keepdim=True)
                    pose_loss = torch.norm(input_pose - fake_pose, dim=1).mean()

                    # compute error and backpropagate
                    self.gen_gamma = min(self.gen_gamma + 1/(2*self.len_dataset), 1)
                    # errG0 = self.criterion.rankerG([scores_D0, scores_D0_G0])
                    errG0 = - scores_D0_G0.mean()
                    # net_error = self.gen_gamma * errG0 + self.identity_lambda*feat_loss + self.pose_lambda*pose_loss# - scores_D0_G0_random.mean()
                    net_error = errG0 + self.identity_lambda*feat_loss + self.pose_lambda*pose_loss
                    net_error.backward()
                    self.optimizerG.step()
                    self.giterations += 1

                    mean_scores_D0 = scores_D0.median().item()
                    mean_scores_D0_G0 = scores_D0_G0.median().item()
                    mean_scores_D0_G1 = scores_D0_G1.median().item()
                    avg_mean_scores_D0 = avg_mean_scores_D0 * 0.2 + mean_scores_D0 * 0.8
                    avg_mean_scores_D0_G0 = avg_mean_scores_D0_G0 * 0.2 + mean_scores_D0_G0 * 0.8
                    avg_mean_scores_D0_G1 = avg_mean_scores_D0_G1 * 0.2 + mean_scores_D0_G1 * 0.8

                    # if self.gen_gamma > 0.5 and self.adaptive_iter:
                    #     if avg_mean_scores_D0_G0 < avg_mean_scores_D0 - 2*(avg_mean_scores_D0 - avg_mean_scores_D0_G1)/3:
                    #         self.g_iter += 1
                    #     else:
                    #         self.g_iter = self.g_iter_init

                    lossG = errG0.item()
                    if j < self.g_iter:
                        i += 1
                        # Bookkeeping
                        losses_train = {}
                        losses_train['Loss_D0'] = lossD
                        losses_train['Loss_G0'] = lossG
                        losses_train['Score_D0'] = mean_scores_D0
                        losses_train['Score_D0_G0'] = mean_scores_D0_G0
                        losses_train['Score_D0_G1'] = mean_scores_D0_G1
                        losses_train['Disc_Diff'] = disc_abs_diff.item()
                        losses_train['Disc_Acc'] = disc_acc
                        losses_train['Iden_Loss'] = feat_loss.item()
                        losses_train['Pose_Loss'] = pose_loss.item()
                        # losses_train['Clamp'] = self.clamp_upper
                        self.monitor_train.update(losses_train, batch_size)
                        print('Stage %d: [%d/%d][%d/%d] Loss_D: %.3f Loss_G: %.3f Score_D0: %.3f Score_D0_G0: %.3f Disc_Acc %.3f Disc_Diff %.3f GP %.3f Iden_Loss %.3f Pose_Loss %.3f'
                        % (stage, epoch, self.stage_epochs[stage], i, self.len_dataset, lossD, lossG, mean_scores_D0, mean_scores_D0_G0, disc_acc, disc_abs_diff.item(), gradient_penalty.item(), feat_loss.item(), pose_loss.item()))

                        if i % self.plot_update_interval == 0:
                            losses_train['netD_norm_w'] = self.get_model_norm(self.modelD[0])[0]
                            losses_train['Gradient_Penalty'] = gradient_penalty.item()
                            losses_train['Real'] = input.cpu()
                            losses_train['Fakes'] = fake_G0.detach().cpu()
                            losses_train['Fakes_Previous'] = fake_G1.detach().cpu()
                            self.visualizer_train.update(losses_train)

                        input = data_iter.next()[0].to(self.device)
                        batch_size = input.size(0)
                        with torch.no_grad():
                            input_x = torch.nn.functional.upsample(input, size=160, mode='bilinear')
                            input_iden = self.facenet(input_x)[0].detach()
                            input_iden = (input_iden - input_iden.mean(1, keepdim=True)) / input_iden.std(1, keepdim=True)
                            input_pose = self.hopenet(input_x)
                            input_pose = torch.cat(input_pose, dim=1).detach()
                            input_pose = (input_pose - input_pose.mean(1, keepdim=True)) / input_pose.std(1, keepdim=True)
                        self.noise.resize_(batch_size, self.noise_z, 1, 1).normal_(0, 1)
                        latent = torch.cat((self.noise, input_iden.unsqueeze(-1).unsqueeze(-1), input_pose.unsqueeze(-1).unsqueeze(-1)), 1)
                        self.noise.data.resize_(batch_size, self.nz, 1, 1).normal_(0, 1)

                # Bookkeeping
                losses_train = {}
                losses_train['Loss_D0'] = lossD
                losses_train['Loss_G0'] = lossG
                losses_train['Score_D0'] = mean_scores_D0
                losses_train['Score_D0_G0'] = mean_scores_D0_G0
                losses_train['Score_D0_G1'] = mean_scores_D0_G1
                losses_train['Disc_Diff'] = disc_abs_diff.item()
                losses_train['Disc_Acc'] = disc_acc
                losses_train['Iden_Loss'] = feat_loss.item()
                losses_train['Pose_Loss'] = pose_loss.item()
                # losses_train['Clamp'] = self.clamp_upper
                self.monitor_train.update(losses_train, batch_size)
                print('Stage %d: [%d/%d][%d/%d] Loss_D: %.3f Loss_G: %.3f Score_D0: %.3f Score_D0_G0: %.3f Score_D0_G1: %.3f Disc_Acc %.3f Disc_Diff %.3f GP %.3f Iden_Loss %.3f Pose_Loss %.3f'
                % (stage, epoch, self.stage_epochs[stage], i, self.len_dataset, lossD, lossG, mean_scores_D0, mean_scores_D0_G0, mean_scores_D0_G1, disc_acc, disc_abs_diff.item(), gradient_penalty.item(), feat_loss.item(), pose_loss.item()))

                if i % self.plot_update_interval == 0:
                    losses_train['netD_norm_w'] = self.get_model_norm(self.modelD[0])[0]
                    losses_train['Gradient_Penalty'] = gradient_penalty.item()
                    losses_train['Real'] = input.cpu()
                    losses_train['Fakes'] = fake_G0.detach().cpu()
                    losses_train['Fakes_Previous'] = fake_G1.detach().cpu()
                    self.visualizer_train.update(losses_train)

                if i % 250 == 0 or i == self.len_dataset:
                    fake_normal_z = self.modelG[0](self.fixed_noise)
                    fake_encoder_z = fake_G0
                    if len(input.size()) < 3:
                        fig = plt.figure()
                        plt.hist(input.squeeze().cpu().numpy(), bins=60)
                        fig.savefig("{}/stage_{}_epoch_{}_real.png".format(self.args.save, stage, i), dpi=fig.dpi)
                        fig = plt.figure()
                        plt.hist(fake_encoder_z.squeeze().data.cpu().numpy(), bins=60)
                        fig.savefig("{}/stage_{}_epoch_{}_fake_enc.png".format(self.args.save, stage, i), dpi=fig.dpi)
                        fig = plt.figure()
                        plt.hist(fake_normal_z.squeeze().data.cpu().numpy(), bins=60)
                        fig.savefig("{}/stage_{}_epoch_{}_fake_norm.png".format(self.args.save, stage, i), dpi=fig.dpi)
                    else:
                        vutils.save_image(fake_normal_z.data, '%s/fake_samples_stage_%03d_epoch_%03d_normal_z.png' % (self.args.save, stage, epoch), normalize=True)
                        vutils.save_image(fake_encoder_z.data, '%s/fake_samples_stage_%03d_epoch_%03d_encoder_z.png' % (self.args.save, stage, epoch), normalize=True)
                        vutils.save_image(input, '%s/real_samples.png' % self.args.save, normalize=True)

                if True:
                    self.disc_diff_weight = self.disc_diff_weight_init
                elif disc_abs_diff.item() < 5:
                    self.disc_diff_weight = (self.marker_high - self.marker_low)*0.2 #self.disc_diff_weight_init/1000
                elif disc_abs_diff.item() < 10:
                    self.disc_diff_weight = (self.marker_high - self.marker_low)*10 #self.disc_diff_weight_init/500
                else:
                    self.disc_diff_weight = self.disc_diff_weight_init

            print('Stage %d: [%d/%d] Avg_Scores_D0 %.3f Avg_Scores_D0_G0 %.3f Avg_Scores_D0_G1 %.3f' % (stage, epoch, self.stage_epochs[stage], avg_mean_scores_D0, avg_mean_scores_D0_G0, avg_mean_scores_D0_G1))

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
        # self.modelG[1].eval()
        # self.modelD[0].eval()
        # self.modelD[1].eval()
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
            self.noise.data.resize_(batch_size, self.nz, 1, 1).normal_(0, 1)
            fake_G0 = self.modelG[0](self.noise)
            fake_G1 = self.modelG[1](self.noise)
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
            test_scores['Test_Score_D0'] = score_D0.mean().item()
            test_scores['Test_Score_D0_G0'] = score_D0_G0.mean().item()
            test_scores['Test_Score_D0_G1'] = score_D0_G1.mean().item()
            test_scores['SSIM'] = ssim_score/batch_size
            test_scores['PSNR'] = psnr_score/batch_size
            test_scores['Test_Disc_Acc'] = disc_acc
            self.monitor_test.update(test_scores, batch_size)
            print('Test: [%d/%d][%d/%d] Score_D0: %.3f Score_D0_G0: %.3f Score_D0_G1: %.3f SSIM: %.3f PSNR: %.3f Disc_Acc: %.3f'
                    % (epoch, self.stage_epochs, i, len(dataloader), score_D0.mean().item(), score_D0_G0.mean().item(), score_D0_G1.mean().item(), ssim_score, psnr_score, disc_acc))

            if i == len(dataloader)-2:
                try:
                    fake_encoder_z = fake_G0
                    if len(self.input.size()) < 3:
                        fig = plt.figure()
                        plt.hist(input.squeeze().cpu().numpy(), bins=60)
                        fig.savefig("{}/stage_{}_epoch_{}_real.png".format(self.dir_save, stage, i), dpi=fig.dpi)
                        fig = plt.figure()
                        plt.hist(fake_normal_z.squeeze().data.cpu().numpy(), bins=60)
                        fig.savefig("{}/stage_{}_epoch_{}_fake_norm.png".format(self.dir_save, stage, i), dpi=fig.dpi)
                    else:
                        vutils.save_image(fake_encoder_z.data, '%s/val_fake_samples_stage_%03d_encoder_z.png' % (self.args.save, stage), normalize=True)
                        vutils.save_image(input, '%s/val_real_samples.png' % self.args.save, normalize=True)
                except Exception as e:
                    print(e)

        avg_mean_scores_D0 = epoch_score_D0.median()
        avg_mean_scores_D0_G0 = epoch_score_D0_G0.median()
        avg_mean_scores_D0_G1 = epoch_score_D0_G1.median()

        test_scores = {}
        test_scores['Test_Score_D0'] = avg_mean_scores_D0
        test_scores['Test_Score_D0_G0'] = avg_mean_scores_D0_G0
        test_scores['Test_Score_D0_G1'] = avg_mean_scores_D0_G1
        test_scores['SSIM'] = epoch_ssim_score/(num_batches-1)
        test_scores['PSNR'] = epoch_psnr_score/(num_batches-1)
        test_scores['Test_Disc_Acc'] = epoch_disc_acc / (num_batches-1)
        test_scores['Test_Real'] = self.test_input.data.cpu()
        test_scores['Test_Fakes'] = fake_G0.data.cpu()
        self.visualizer_test.update(test_scores)

        # Check for Discriminator Saturation
        if avg_mean_scores_D0_G0 > avg_mean_scores_D0 - (avg_mean_scores_D0 - avg_mean_scores_D0_G1)/4:
            self.update_opt_flag = True
        else:
            self.update_opt_flag = False

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
        self.monitor_train.reset()
        data_iter = iter(dataloader)
        self.len_dataset = int(len(dataloader) * self.dataset_fraction)
        if epoch == 0:
            print("Length of Dataset: {}".format(self.len_dataset))

        i = 0
        avg_mean_scores_D0 = 0.0
        avg_mean_scores_D0_G0 = 0.0
        avg_mean_scores_D0_G1 = 0.0
        feat_loss = torch.zeros(1)
        pose_loss = torch.zeros(1)
        lossG = 0
        while i < self.len_dataset:
            ############################
            # Update Discriminator Network
            ############################
            self.modelD[0].train()
            # self.modelG[0].eval()
            acc = 0.0
            i += 1

            # clamp parameters to a cube
            if not self.gp:
                for p in self.modelD[0].parameters():
                    p.data.clamp_(self.clamp_lower, self.clamp_upper)

            input = data_iter.next()[0].to(self.device)
            batch_size = input.size(0)

            self.modelD[0].zero_grad()
            self.modelG[0].zero_grad()

            # train with real
            if self.add_noise:
                self.epsilon.data.resize_(input.size()).normal_(0, self.noise_var)
                dis_input = self.input + self.epsilon
                scores_D0 = self.modelD[0](dis_input, self.extra_layer, self.extra_layer_gamma)
            else:
                scores_D0 = self.modelD[0](input, self.extra_layer, self.extra_layer_gamma)
            acc += torch.sum((scores_D0 > self.acc_margin).float())

            # train with fake
            input_x = torch.nn.functional.upsample(input, size=160, mode='bilinear')
            input_iden = self.facenet(input_x)[0].detach()
            input_iden = (input_iden - input_iden.mean(1, keepdim=True)) / input_iden.std(1, keepdim=True)
            input_pose = self.hopenet(input_x)
            input_pose = torch.cat(input_pose, dim=1).detach()
            input_pose = (input_pose - input_pose.mean(1, keepdim=True)) / input_pose.std(1, keepdim=True)

            self.noise.resize_(batch_size, self.noise_z, 1, 1).normal_(0, 1)
            latent = torch.cat((self.noise, input_iden.unsqueeze(-1).unsqueeze(-1), input_pose.unsqueeze(-1).unsqueeze(-1)), 1)

            fake_G0 = self.modelG[0](latent).detach()
            if self.add_noise:
                self.epsilon.data.resize_(self.input.size()).normal_(0, self.noise_var)
                dis_input = fake_G0 + self.epsilon
                scores_D0_G0 = self.modelD[0](dis_input, self.extra_layer, self.extra_layer_gamma)
            else:
                scores_D0_G0 = self.modelD[0](fake_G0, self.extra_layer, self.extra_layer_gamma)
            acc += torch.sum((scores_D0_G0 <= 0).float())
            disc_acc = float(acc)*50 / batch_size

            self.noise.resize_(batch_size, self.nz, 1, 1).normal_(0, 1)
            fake_G1 = self.modelG[0](self.noise).detach()
            if self.add_noise:
                self.epsilon.data.resize_(self.input.size()).normal_(0, self.noise_var)
                dis_input = fake_G0 + self.epsilon
                scores_D0_G1 = self.modelD[0](dis_input, self.extra_layer, self.extra_layer_gamma)
            else:
                scores_D0_G1 = self.modelD[0](fake_G1, self.extra_layer, self.extra_layer_gamma)
            acc += torch.sum((scores_D0_G1 <= 0).float())
            disc_acc = float(acc)*100 / (3*batch_size)

            # Compute loss and do backward()
            if self.wgan:
                errD0 = ((scores_D0_G0 + scores_D0_G1)/2 - scores_D0).mean()
            else:
                errD0 = self.criterion.hinge_loss(scores_D0, 1)
                errD0 += self.criterion.hinge_loss(scores_D0_G0, -1)
                errD0 /= 2
            # self.schedulerD.step(errD0.data[0])

            if self.gp:
                gradient_penalty = self.calc_gradient_penalty(input, fake_G0, batch_size)
                net_error = errD0 + self.gp_lambda * gradient_penalty
            else:
                net_error = errD0
                gradient_penalty = self.t_zero

            net_error.backward()
            self.optimizerD.step()
            self.extra_layer_gamma = min(self.extra_layer_gamma + 1/(2*self.len_dataset), 1)

            mean_scores_D0 = scores_D0.median().item()
            mean_scores_D0_G0 = scores_D0_G0.median().item()

            lossD = errD0.item()
            # Bookkeeping
            losses_train = {}
            losses_train['Loss_D0'] = lossD
            losses_train['Loss_G0'] = lossG
            losses_train['Score_D0'] = mean_scores_D0
            losses_train['Score_D0_G0'] = mean_scores_D0_G0
            losses_train['Score_D0_G1'] = 0
            losses_train['Disc_Diff'] = 0
            losses_train['Disc_Acc'] = disc_acc
            # losses_train['Clamp'] = self.clamp_upper
            losses_train['Score_D0_G1'] = 0
            losses_train['Iden_Loss'] = feat_loss.item()
            losses_train['Pose_Loss'] = pose_loss.item()
            # losses_train['Score_D1_G1'] = 0
            # losses_train['Score_D1'] = 0
            self.monitor_train.update(losses_train, batch_size)
            print('Stage %d: [%d/%d][%d/%d] Loss_D: %.3f Loss_G: %.3f Score_D0: %.3f Score_D0_G0: %.3f Disc_Acc %.3f GP %.3f Iden_Loss %.3f Pose_Loss %.3f'
            % (stage, epoch, self.stage_epochs[stage], i, self.len_dataset, lossD, lossG, mean_scores_D0, mean_scores_D0_G0, disc_acc, gradient_penalty.item(), feat_loss.item(), pose_loss.item()))

            if i % self.plot_update_interval == 0:
                # Compute MSE
                losses_train['netD_norm_w'] = self.get_model_norm(self.modelD[0])[0]
                losses_train['Gradient_Penalty'] = gradient_penalty.item()
                losses_train['Real'] = input.cpu()
                losses_train['Fakes'] = fake_G0.detach().cpu()
                losses_train['Fakes_Previous'] = torch.zeros(fake_G0.size())
                self.visualizer_train.update(losses_train)


    def compute_markers(self, dataloader):
        data_iter = iter(dataloader)
        self.len_dataset = int(len(dataloader) * self.dataset_fraction)
        # self.modelD[0].eval()
        # self.modelG[0].eval()
        i = 0
        all_scores_D1 = torch.Tensor([]).to(self.device)
        all_scores_D1_G1 = torch.Tensor([]).to(self.device)

        while i < self.len_dataset:
            i += 1
            input = data_iter.next()[0].to(self.device)
            batch_size = input.size(0)

            with torch.no_grad():
                if self.add_noise:
                    self.epsilon.data.resize_(input.size()).normal_(0, self.noise_var)
                    dis_input = input + self.epsilon
                    scores_D1 = self.modelD[1](dis_input, 0)
                else:
                    scores_D1 = self.modelD[1](input, 0)

                # train with fake
                input_x = torch.nn.functional.upsample(input, size=160, mode='bilinear')
                input_iden = self.facenet(input_x)[0].detach()
                input_iden = (input_iden - input_iden.mean(1, keepdim=True)) / input_iden.std(1, keepdim=True)
                input_pose = self.hopenet(input_x)
                input_pose = torch.cat(input_pose, dim=1).detach()
                input_pose = (input_pose - input_pose.mean(1, keepdim=True)) / input_pose.std(1, keepdim=True)

                self.noise.data.resize_(batch_size, self.noise_z, 1, 1).normal_(0, 1)
                latent = torch.cat((self.noise, input_iden.unsqueeze(-1).unsqueeze(-1), input_pose.unsqueeze(-1).unsqueeze(-1)), 1)

                fake_G1 = self.modelG[1](latent).detach()
                if self.add_noise:
                    self.epsilon.data.resize_(input.size()).normal_(0, self.noise_var)
                    dis_input = fake_G1 + self.epsilon
                    scores_D1_G1 = self.modelD[1](dis_input, 0)
                else:
                    scores_D1_G1 = self.modelD[1](fake_G1, 0)

            all_scores_D1 = torch.cat((all_scores_D1, scores_D1.data), dim=0)
            all_scores_D1_G1 = torch.cat((all_scores_D1_G1, scores_D1_G1.data), dim=0)
        return all_scores_D1.max().item(), all_scores_D1_G1.min().item()


