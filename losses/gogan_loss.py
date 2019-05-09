# gogan loss module

import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
# import pytorch_fft.fft.autograd as fft_autograd
# import pytorch_fft.fft as fft

class GoGANLoss(nn.Module):
    def __init__(self, args):
        super(GoGANLoss, self).__init__()
        self.is_cuda = args.cuda
        self.margin = args.margin
        self.batchsize = args.batch_size

        self.real_value = +1
        self.fake_value = -1
        self.epsilon = 10

        self.real_label = Variable(torch.zeros(self.batchsize).fill_(self.real_value))
        self.fake_label = Variable(torch.zeros(self.batchsize).fill_(self.fake_value))
        self.zeros = Variable(torch.zeros(self.batchsize))

        self.margin_D_zero = nn.MarginRankingLoss(margin=0)
        self.margin_G_zero = nn.MarginRankingLoss(margin=0)
        self.margin_D = nn.MarginRankingLoss(margin = self.margin)
        self.margin_G = nn.MarginRankingLoss(margin = self.margin)

        # Generate impulse for correlation loss
        # height, width = args.resolution_high, args.resolution_wide
        # sigma = args.correlation_sigma
        # self.fft = fft.fft2
        # self.ifft = fft.ifft2
        # self.fft_a = fft_autograd.Fft2d()
        # self.ifft_a = fft_autograd.Ifft2d()

        # g_x, g_y = np.meshgrid(range(round(-width/2), round(width/2)), range(round(-height/2), round(height/2)))
        # self.impulse = np.exp(-np.power(g_x, 2)/(2*sigma**2) - np.power(g_y, 2)/(2*sigma**2))
        # # self.impulse = self.impulse/np.sum(self.impulse)
        # self.impulse = self.impulse/self.impulse.max()
        # self.impulse = Variable(torch.FloatTensor(self.impulse)).cuda()
        # self.impulse = self.impulse.unsqueeze(0)
        # self.impulse = self.impulse.expand(self.batchsize, -1, -1).contiguous()
        # self.FI_re, self.FI_im = self.fft_a(self.impulse, Variable(torch.zeros(self.impulse.size()).contiguous()).cuda())

        # self.FI_re[:,0,0].data.fill_(0.0)
        # # self.FI_re[:,:,0].data.fill_(0.0)
        # self.FI_im[:,0,0].data.fill_(0.0)
        # # self.FI_im[:,:,0].data.fill_(0.0)

        if self.is_cuda:
            self.margin_D = self.margin_D.cuda()
            self.margin_G = self.margin_G.cuda()
            self.margin_D_zero = self.margin_D_zero.cuda()
            self.margin_G_zero = self.margin_G_zero.cuda()
            self.real_label = self.real_label.cuda()
            self.fake_label = self.fake_label.cuda()
            self.zeros = self.zeros.cuda()

    def set_margin(self, margin):
        self.margin_D = nn.MarginRankingLoss(margin=margin)
        self.margin_G = nn.MarginRankingLoss(margin=margin)

    def hinge_loss(self, input, label, margin=None):
        if margin is None:
            margin = self.margin
        self.zeros.data.resize_(input.size()).fill_(0)
        loss = torch.max(self.zeros, margin - label*input)
        return loss.mean()

    def rankerD(self, input):
        batchsize = input[0].size(0)
        # self.real_label.data.resize_(batchsize).fill_(self.real_value)
        self.fake_label.data.resize_(batchsize).fill_(self.fake_value)

        num = len(input)
        if num == 2:
            loss = self.margin_D(input[0], input[1], self.real_label)
        elif num == 3:
            loss = self.margin_D_zero(input[1], input[2], self.fake_label)
            # loss += self.margin_D(input[0], input[1], self.real_label)
        return loss

    def rankerG(self, input):
        batchsize = input[0].size(0)
        # self.real_label.data.resize_(batchsize).fill_(self.real_value)
        self.fake_label.data.resize_(batchsize).fill_(self.fake_value)

        # num = len(input)
        loss = self.margin_G_zero(input[0], input[1], self.fake_label)
            # loss += self.margin_G(input[1], input[2], self.real_label)
        return loss

    def generator(self, g_loss):
        g_loss = g_loss.mean(0).view(1)
        return g_loss

    def kl_divergence(self, z_mu, z_logvar):
        z_sigma = torch.exp(z_logvar * 0.5)
        kl_d = - 0.5 * (1 + z_logvar - z_mu.pow(2) - z_sigma.pow(2)).sum(1).mean(0)
        return kl_d

    def diff_loss(self, X, Y, type='l2'):
        if type == 'l1':
            diff = (X - Y).abs()
            loss = diff.mean(0).sum()
        elif type == 'l2':
            diff = X - Y
            loss = (0.5 * torch.mul(diff, diff)).mean(0).sum()
        return loss

    def model_norm_loss(self, model1, model2):
        loss = Variable(torch.zeros(1), requires_grad=True)
        if self.is_cuda:
            loss = loss.cuda()

        param_list1 = list(model1.main.parameters())
        param_list2 = list(model2.main.parameters())

        if len(param_list1) == 0 or len(param_list1) != len(param_list2):
            return loss

        for i in range(len(param_list1)):
            loss += torch.norm(param_list1[i] - param_list2[i])
        return loss

    # def correlation_loss(self, X, X_hat):
    #     if X.size(0) < self.batchsize:
    #         return Variable(torch.zeros(1)).cuda(), 0

    #     X_gray = 0.299 * X[:,0,...] + 0.587 * X[:,1,...] + 0.114 * X[:,2,...]
    #     X_hat_gray = 0.299 * X_hat[:,0,...] + 0.587 * X_hat[:,1,...] + 0.114 * X_hat[:,2,...]

    #     X_zero = Variable(torch.zeros(X_gray.size()))
    #     X_hat_zero = Variable(torch.zeros(X_gray.size()), requires_grad=True)
    #     if self.is_cuda:
    #         X_zero = X_zero.cuda()
    #         X_hat_zero = X_hat_zero.cuda()
    #     FX_re, FX_im = self.fft_a(X_gray, X_zero)

    #     # remove mean row and column frequencies
    #     FX_re[:,0,0].data.fill_(0.0)
    #     # FX_re[:,:,0].data.fill_(0.0)
    #     FX_im[:,0,0].data.fill_(0.0)
    #     # FX_im[:,:,0].data.fill_(0.0)

    #     # Compute FFT(H)
    #     # F(H) = (F*(X) o F(g)) / (F*(X)) o F(X) + eps)
    #     denominator = FX_re.pow(2) + FX_im.pow(2) + self.epsilon
    #     # print(denominator)

    #     FH_re = (FX_re*self.FI_re + FX_im*self.FI_im) / denominator
    #     FH_im = (FX_re*self.FI_im - FX_im*self.FI_re) / denominator

    #     # Compute FFT(X^)
    #     FX_hat_re, FX_hat_im = self.fft_a(X_hat_gray, X_hat_zero)

    #     # remove mean row and column frequencies
    #     # FX_hat_re[:,0,:].data.fill_(0.0)
    #     # FX_hat_re[:,:,0].data.fill_(0.0)
    #     # FX_hat_im[:,0,:].data.fill_(0.0)
    #     # FX_hat_im[:,:,0].data.fill_(0.0)

    #     # Compute F(H) o F(X^) - F(g)
    #     F_corr_re = FH_re*FX_hat_re - FH_im*FX_hat_im
    #     F_corr_im = FH_re*FX_hat_im + FH_im*FX_hat_re

    #     corr_re, corr_im = self.ifft_a(F_corr_re, F_corr_im)

    #     loss_re = F_corr_re - self.FI_re
    #     loss_im = F_corr_im - self.FI_im

    #     loss = loss_re.pow(2) + loss_im.pow(2)
    #     loss = 0.5*loss.mean(0).sum()

    #     return loss, corr_re

class RankOrderLoss(nn.Module):
    """docstring for RankOrderLoss"""
    def __init__(self, device):
        super(RankOrderLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss().to(device)

    def __call__(self, outputs, target):
        loss = self.loss(outputs, target)
        return loss



