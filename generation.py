from torch.autograd import Variable
# evaluate.py

import torch
import plugins
import torchvision
import numpy as np
import scipy
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import pyamg
import PIL.Image
from torchvision.utils import save_image
import warnings
warnings.filterwarnings("ignore")


class Generator():

    def __init__(self, args, netD, netG, netE):
        self.netD = netD
        self.netG = netG
        self.netE = netE

        self.args = args
        self.nchannels = args.nchannels
        self.resolution_high = args.resolution_high
        self.resolution_wide = args.resolution_wide
        self.nz = args.nz
        self.wcom = args.disc_loss_weight
        self.cuda = args.cuda
        self.citers = args.citers
        self.lr = args.learning_rate_vae
        self.momentum = args.momentum
        self.batch_size = args.batch_size
        self.use_encoder = args.use_encoder

        self.input = Variable(torch.FloatTensor(self.batch_size,self.nchannels,self.resolution_high,self.resolution_wide), volatile=True).cuda()
        self.epsilon = Variable(torch.randn(self.batch_size, self.nz), volatile=True).cuda()
        self.noise = Variable(torch.FloatTensor(self.batch_size, self.nz, 1, 1).normal_(0, 1), volatile=True)

        if args.cuda:
            self.input = self.input.cuda()
            self.epsilon = self.epsilon.cuda()
            self.noise = self.noise.cuda()

        self.log_eval_loss = plugins.Logger(args.logs, 'Generation.txt')
        self.params_eval_loss = ['Image', 'SSIM_1', 'SSIM_2', 'SSIM_3', 'SSIM_4', 'PSNR_1', 'PSNR_2', 'PSNR_3', 'PSNR_4', 'DiscScore_1', 'DiscScore_2', 'DiscScore_3', 'DiscScore_4']
        self.log_eval_loss.register(self.params_eval_loss)

        self.losses = {}
        self.log_eval_monitor = plugins.Monitor()
        self.params_eval_monitor = ['Image', 'SSIM_1', 'SSIM_2', 'SSIM_3', 'SSIM_4', 'PSNR_1', 'PSNR_2', 'PSNR_3', 'PSNR_4', 'DiscScore_1', 'DiscScore_2', 'DiscScore_3', 'DiscScore_4']
        self.log_eval_monitor.register(self.params_eval_monitor)

        self.print = '[%d/%d] '
        for item in self.params_eval_loss:
            self.print = self.print + item + " %.4f "

    def ssim(self, data1, data2):
        num = data1.size(0)
        nchannels = data1.size(1)

        data1 = data1.transpose(1, 2).transpose(2, 3)
        data2 = data2.transpose(1, 2).transpose(2, 3)

        score = 0

        if nchannels > 1:
            for i in range(num):
                img1 = data1[i].numpy()
                img2 = data2[i].numpy()

                range1 = img1.max() - img1.min()
                range2 = img2.max() - img2.min()
                range3 = max(range1, range2)
                score += ssim(img1, img2, dynamic_range=range3,
                              multichannel=True)
        else:
            for i in range(num):
                img1 = data1[i].numpy()
                img2 = data2[i].numpy()
                score += ssim(img1, img2, dynamic_range=self.range)
        return score/num

    def psnr(self, data1, data2):
        num = data1.size(0)
        nchannels = data1.size(1)

        data1 = data1.transpose(1, 2).transpose(2, 3)
        data2 = data2.transpose(1, 2).transpose(2, 3)

        score = 0

        for i in range(num):
            img1 = data1[i].numpy()
            img2 = data2[i].numpy()

            range1 = img1.max() - img1.min()
            range2 = img2.max() - img2.min()
            range3 = max(range1, range2)
            score += psnr(img1, img2, dynamic_range=range3)

        return score/num

    def generate(self, dataloader):
        data_iter = iter(dataloader)

        data_i = 0
        while data_i < len(dataloader):
            data_real = data_iter.next()[0]
            data_i += 1

            batch_size = data_real.size(0)
            self.input.data.resize_(data_real.size()).copy_(data_real)

            if self.use_encoder:
                self.epsilon.data.resize_(batch_size, self.nz).normal_(0, 1)
                noise_mu, noise_logvar = self.netE(self.input)
                noise_sigma = torch.exp(torch.mul(noise_logvar, 0.5))
                latents = noise_mu + torch.mul(noise_sigma, self.epsilon)
                latents = latents.unsqueeze(-1).unsqueeze(-1)
                self.noise.data.copy_(latents.data)
            else:
                self.noise.data.normal_(0, 1)

            self.output = [self.netG[i].forward(self.noise) for i in range(4)]
            ssims = [self.ssim(self.input.data.cpu(), self.output[i].data.cpu()) for i in range(4)]
            psnrs = [self.psnr(self.input.data.cpu(), self.output[i].data.cpu()) for i in range(4)]
            disc_scores = [self.netD(self.output[i]).mean(0).data[0] for i in range(4)]

            save_image(normalize(self.input.data[0].cpu()), "{}/{}_Original.png".format(self.args.save, data_i), padding=0, normalize=True)
            for k in range(4):
                save_image(normalize(self.output[k].data[0].cpu()), "{}/{}_Fake_Stage_{}.png".format(self.args.save, data_i, k+1), padding=0, normalize=True)

            self.losses['Image'] = float(data_i)
            for k in range(4):
                self.losses['SSIM_{}'.format(k+1)] = ssims[k]
                self.losses['PSNR_{}'.format(k+1)] = psnrs[k]
                self.losses['DiscScore_{}'.format(k+1)] = disc_scores[k]
            self.log_eval_monitor.update(self.losses, batch_size, keepsame=True)
            print(self.print % tuple([data_i, len(dataloader)] + [self.losses[key]
                                               for key in self.params_eval_monitor]))
            loss = self.log_eval_monitor.getvalues()
            self.log_eval_loss.update(loss)

    def generate_one(self, dataloader):
        data_iter = iter(dataloader)

        data_i = 0
        while data_i < len(dataloader):
            data_real = data_iter.next()[0]
            data_i += 1

            batch_size = data_real.size(0)
            self.input.data.resize_(data_real.size()).copy_(data_real)

            if self.use_encoder:
                self.epsilon.data.resize_(batch_size, self.nz).normal_(0, 1)
                noise_mu, noise_logvar = self.netE(self.input)
                noise_sigma = torch.exp(torch.mul(noise_logvar, 0.5))
                latents = noise_mu + torch.mul(noise_sigma, self.epsilon)
                latents = latents.unsqueeze(-1).unsqueeze(-1)
                self.noise.data.copy_(latents.data)
            else:
                self.noise.data.normal_(0, 1)

            self.output = self.netG.forward(self.noise)
            # ssims = [self.ssim(self.input.data.cpu(), self.output[i].data.cpu()) for i in range(4)]
            # psnrs = [self.psnr(self.input.data.cpu(), self.output[i].data.cpu()) for i in range(4)]
            # disc_scores = [self.netD(self.output[i]).mean(0).data[0] for i in range(4)]

            # save_image(normalize(self.input.data[0].cpu()), "{}/{}_Original.png".format(self.args.save, data_i), padding=0, normalize=True)
            # for k in range(4):
            save_image(normalize(self.output.data[0].cpu()), "{}/{}_wgan_fake.png".format(self.args.save, data_i), padding=0, normalize=True)

    def interpolate(self, dataloader):
        if self.use_encoder:
            data_iter = iter(dataloader)
            data_real = data_iter.next()[0]
            batch_size = data_real.size(0)

            self.input.data.resize_(data_real.size()).copy_(data_real)
            self.epsilon.data.resize_(batch_size, self.nz).normal_(0, 1)
            noise_mu, noise_logvar = self.netE(self.input)
            noise_sigma = torch.exp(torch.mul(noise_logvar, 0.5))
            latents = noise_mu + torch.mul(noise_sigma, self.epsilon)
            latents = latents.unsqueeze(-1).unsqueeze(-1)
            self.noise.data.copy_(latents.data)

            z1 = self.noise[0:3]
            z2 = self.noise[3:6]

            for i in range(8):
                z = z1 + (z2-z1)*i/7

                self.output = [self.netG[i].forward(z) for i in range(4)]
                for stage in range(4):
                    for anchor in range(3):
                        save_image(normalize(self.output[stage].data[anchor].cpu()), "{}/Montage_{}_Stage_{}_Z_{}.png".format(self.args.save, anchor+1, stage+1, i+1), padding=0, normalize=True)


        else:
            z1 = Variable(torch.FloatTensor(3, self.nz, 1, 1).normal_(0, 1), volatile=True).cuda()
            z2 = Variable(torch.FloatTensor(3, self.nz, 1, 1).normal_(0, 1), volatile=True).cuda()

            for i in range(8):
                z = z1 + (z2-z1)*i/7

                self.output = [self.netG[i].forward(z) for i in range(4)]
                for stage in range(4):
                    for anchor in range(3):
                        save_image(normalize(self.output[stage].data[anchor].cpu()), "{}/Montage_{}_Stage_{}_Z_{}.png".format(self.args.save, anchor+1, stage+1, i+1), padding=0, normalize=True)


def normalize(image):
    image = (image - image.min())
    return image / image.max()

