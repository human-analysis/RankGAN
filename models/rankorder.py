# gogan model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal, calculate_gain
import math

class Upsample(nn.Module):
    """docstring for Upsample"""
    def __init__(self, scale_factor):
        super(Upsample, self).__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear')

    def forward(self, x):
        x = self.up(x)
        return x

# Simple DCGAN
class DCGAN_DRank(nn.Module):
    def __init__(self, args, n_extra_layers=0):
        super(DCGAN_DRank, self).__init__()
        self.ngpu = args.ngpu
        nc = args.nchannels
        ndf = args.ndf
        isize = args.resolution_high
        nranks = args.nranks
        n_extra_layers = args.n_extra_layers

        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial_conv_{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial_relu_{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}_{1}_conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}_{1}_batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}_{1}_relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid_{0}-{1}_conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid_{0}_batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid_{0}_relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('penultimate_{0}-{1}_conv'.format(cndf, nranks*4),
                        nn.Conv2d(cndf, nranks*4, 4, 1, 0, bias=False))
        main.add_module('penultimate_{0}_relu'.format(nranks*4),
                        nn.LeakyReLU(0.2, inplace=True))
        self.final = nn.Linear(nranks*4, nranks-1)
        self.main = main


    def forward(self, input, extra_layer=0, extra_layer_gamma=0):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            inner = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            inner = inner.squeeze()
            output = nn.parallel.data_parallel(self.final, inner, range(self.ngpu))
        else:
            inner = self.main(input)
            inner = inner.squeeze()
            output = self.final(inner)

        # output = output.mean(0)
        return output

class DCGAN_GRank(nn.Module):
    def __init__(self, args, n_extra_layers=0):
        super(DCGAN_GRank, self).__init__()
        self.ngpu = args.ngpu
        isize = args.resolution_high
        nz = args.nz
        nc = args.nchannels
        ngf = args.ngf
        use_upsampling = args.use_upsampling
        n_extra_layers = args.n_extra_layers

        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial_{0}-{1}_convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial_{0}_batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial_{0}_relu'.format(cngf),
                        nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize//2:
            if use_upsampling:
                main.add_module('pyramid_{0}-{1}_upsample'.format(cngf, cngf//2),
                                Upsample(2))
                main.add_module('pyramid_{0}-{1}_convt'.format(cngf, cngf//2),
                                nn.Conv2d(cngf, cngf//2, 5, 1, 2, bias=False))
            else:
                main.add_module('pyramid_{0}-{1}_convt'.format(cngf, cngf//2),
                                nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid_{0}_batchnorm'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid_{0}_relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}_{1}_conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}_{1}_batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}_{1}_relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final_{0}-{1}_convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final_{0}_tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
