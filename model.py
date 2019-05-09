# model.py

import math
from torch import nn
import torch
import models
import losses
import numpy as np
import utils

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0] # number of rows
        fan_in = size[1] # number of columns
        variance = np.sqrt(2.0/(fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

class Model:

    def __init__(self, args):
        self.cuda = args.cuda
        self.device = torch.device("cuda" if (self.cuda and torch.cuda.is_available()) else "cpu")
        self.args = args
        self.net_type = args.net_type
        if self.net_type == 'dcgan_nvidia':
            self.models = (models.netD_nvidia_Generic_Cap(args), models.netG_nvidia_Generic(args), models.netE_Generic(args))
        elif self.net_type == 'dcgan_generic':
            self.models = (models.netD_Generic_Cap(args), models.netG_Generic(args), models.netE_Generic(args))
        elif self.net_type == 'dcgan':
            self.models = (models.DCGAN_D(args), models.DCGAN_G(args), models.netE_Generic(args))
        elif self.net_type == 'dcgan_rank':
            self.models = (models.DCGAN_DRank(args), models.DCGAN_GRank(args), models.netE_Generic(args))
        elif self.net_type == 'gmm':
            self.models = (models.GMM_D(args), models.GMM_G(args), models.GMM_E(args))
        elif self.net_type == 'openface':
            self.models = [models.netOpenFace(args)]
        elif self.net_type == 'lightcnn':
            self.models = [models.LightCNN_29Layers_v2(num_classes = 80013)]
        else:
            raise("Unknown network architecture")

        self.criterion = losses.GoGANLoss(args=self.args)

    def setup(self, checkpoints):
        if checkpoints.latest('resume') == None:
            for model in self.models:
                if 'NormConvBlock' not in model.__repr__():
                    try:
                        model.apply(weights_init)
                    except Exception as e:
                        print(e)

        # else:
        #     tmp = checkpoints.load(checkpoints['resume'])
        #     model.load_state_dict(tmp)

        for model in self.models:
            model = model.to(self.device)

        return self.models, self.criterion
