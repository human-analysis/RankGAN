# gogan model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal, calculate_gain
from torch.autograd import Variable
import math

def he_init(layer, nonlinearity='conv2d', param=None):
    nonlinearity = nonlinearity.lower()
    if nonlinearity not in ['linear', 'conv1d', 'conv2d', 'conv3d', 'relu', 'leaky_relu', 'sigmoid', 'tanh']:
        if not hasattr(layer, 'gain') or layer.gain is None:
            gain = 0  # default
        else:
            gain = layer.gain
    elif nonlinearity == 'leaky_relu':
        assert param is not None, 'Negative_slope(param) should be given.'
        gain = calculate_gain(nonlinearity, param)
    else:
        gain = calculate_gain(nonlinearity)
    kaiming_normal(layer.weight, a=gain)

class Upsample(nn.Module):
    """docstring for Upsample"""
    def __init__(self, scale_factor):
        super(Upsample, self).__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='nearest')

    def forward(self, x):
        x = self.up(x)
        return x

class PixelNormLayer(nn.Module):
    """
    Pixelwise feature vector normalization.
    """
    def __init__(self, eps=1e-8):
        super(PixelNormLayer, self).__init__()
        self.eps = eps

    def forward(self, x):
        nc_input = x.size(1)
        # return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
        return x * math.sqrt(nc_input) / (x.norm(dim=1, keepdim=True) + self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(eps = %s)' % (self.eps)

class MinibatchStatConcatLayer(nn.Module):
    """docstring for MinibatchStatConcatLayer"""
    def __init__(self):
        super(MinibatchStatConcatLayer, self).__init__()

    def forward(self, x):
        size = x.size()
        if size[0] == 1:
            std = x
        else:
            std = x.std(0, True, True)
        mean = std.mean(1, True).mean(2, True).mean(3, True)
        mean = mean.expand(size[0], 1, size[2], size[3])
        y = torch.cat((x, mean), dim=1)
        return y

class WScaleLayer(nn.Module):
    """
    Applies equalized learning rate to the preceding layer.
    """
    def __init__(self, incoming):
        super(WScaleLayer, self).__init__()
        self.incoming = incoming
        self.scale = (torch.mean(self.incoming.weight.data ** 2)) ** 0.5
        self.incoming.weight.data.copy_(self.incoming.weight.data / self.scale)
        self.bias = None
        if self.incoming.bias is not None:
            self.bias = self.incoming.bias
            self.incoming.bias = None

    def forward(self, x):
        # self.scale = (torch.mean(self.incoming.weight.data ** 2)) ** 0.5
        # self.incoming.weight.data.copy_(self.incoming.weight.data / self.scale)
        x = self.scale * x
        if self.bias is not None:
            x += self.bias.view(1, self.bias.size()[0], 1, 1)
        return x

    def __repr__(self):
        param_str = '(incoming = %s)' % (self.incoming.__class__.__name__)
        return self.__class__.__name__ + param_str

class LayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y

def NINLayer(in_channels, out_channels, nonlinearity, init, param=None,
            to_sequential=True, use_wscale=True):
    layers = []
    layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)]  # NINLayer in lasagne
    layers[-1].bias.data.fill_(0)
    he_init(layers[-1], init, param)  # init layers
    if use_wscale:
        layers += [WScaleLayer(layers[-1])]
    if not (nonlinearity == 'linear'):
        layers += [nonlinearity]
    if to_sequential:
        return nn.Sequential(*layers)
    else:
        return layers

class NormConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pixelnorm=True, activation=True):
        super(NormConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False)
        he_init(self.conv, 'leaky_relu', 0.2)
        self.wscale = WScaleLayer(self.conv)
        self.act = nn.LeakyReLU(0.2)
        self.norm = PixelNormLayer()
        self.pixelnorm = pixelnorm
        self.activate = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.wscale(x)
        if self.activate:
            x = self.act(x)
        if self.pixelnorm:
            x = self.norm(x)
        return x

class GenericConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, normalize=True, norm_type='batch', activation=True):
        super(GenericConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.act = nn.LeakyReLU(0.2)
        self.normalize = normalize
        try:
            self.norm = {'batch': nn.BatchNorm2d, 'lrn': nn.LocalResponseNorm, 'instance': nn.InstanceNorm2d, 'layer': LayerNorm}
        except Exception as e:
            self.norm = {'batch': nn.BatchNorm2d, 'instance': nn.InstanceNorm2d, 'layer': LayerNorm}
        layers = []
        layers.append(self.conv)
        if self.normalize:
            if norm_type == 'lrn':
                layers.append(nn.LocalResponseNorm(1))
            elif norm_type == 'pixel':
                layers.append(PixelNormLayer())
            else:
                layers.append(self.norm[norm_type](out_channels))
        if activation:
            layers.append(self.act)
        self.block = nn.Sequential(*layers)

    def forward(self, input):
        output = self.block(input)
        return output

# Generic Models With nVIDIA PixelNormalization and Weight Scaling
class netG_nvidia_Generic(nn.Module):
    def __init__(self, args):
        super(netG_nvidia_Generic, self).__init__()

        self.ngpu = args.ngpu
        self.nc = args.nchannels
        self.ngf = args.ngf
        self.nz = args.nz
        self.up = Upsample(scale_factor = 2)
        self.height = args.resolution_high
        self.width = args.resolution_wide
        self.extra_cap = args.extra_G_cap
        assert self.height == self.width, "Image height is not equal to width"

        count = int(self.height/2)
        layers = []
        layers.append(self.up)
        layers.append(NormConvBlock(self.nz, count*self.ngf, kernel_size=3, stride=1, padding=1, pixelnorm=True))
        if self.extra_cap:
            layers.append(NormConvBlock(count*self.ngf, count*self.ngf, kernel_size=3, stride=1, padding=1, pixelnorm=True))
        while count > 1:
            count = int(count/2)
            layers.append(self.up)
            layers.append(NormConvBlock(2*count*self.ngf, count*self.ngf, kernel_size=3, stride=1, padding=1, pixelnorm=True))
            if self.extra_cap:
                layers.append(NormConvBlock(count*self.ngf, count*self.ngf, kernel_size=3, stride=1, padding=1, pixelnorm=True))
        layers.append(self.toRGB(self.ngf))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def toRGB(self, input_channels):
        return NormConvBlock(input_channels, self.nc, kernel_size=1, stride=1, padding=0, pixelnorm=False, activation=False)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

# Generic DCGAN without ConvTranspose2D
class netE_nvidia_Generic(nn.Module):
    def __init__(self, args):
        super(netE_nvidia_Generic, self).__init__()

        self.ngpu = args.ngpu
        self.nc = args.nchannels
        self.ndf = args.ndf
        self.nz = args.nz
        self.down = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)
        self.height = args.resolution_high
        self.width = args.resolution_wide
        assert self.height == self.width, "Image height is not equal to width"

        count = 1
        layers = []
        layers.append(NormConvBlock(self.nc, self.ndf, kernel_size=3, padding=1, stride=1))
        layers.append(self.down)

        while count < self.height/4:
            layers.append(NormConvBlock(count*self.ndf, 2*count*self.ndf, kernel_size=3, stride=1, padding=1))
            layers.append(self.down)
            count *= 2

        layers.append(NormConvBlock(count*self.ndf, self.nz, kernel_size=2, stride=1, padding=0))
        self.main = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(self.nz, self.nz)
        self.fc_sigma = nn.Linear(self.nz, self.nz)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            hidden = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            hidden = self.main(input)

        hidden = hidden.view(-1, self.nz)

        if isinstance(hidden.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            z_mu = nn.parallel.data_parallel(self.fc_mu, hidden, range(self.ngpu))
            z_sigma = nn.parallel.data_parallel(self.fc_sigma, hidden, range(self.ngpu))
        else:
            z_mu = self.fc_mu(hidden)
            z_sigma = self.fc_sigma(hidden)
        return z_mu, z_sigma

# Generic DCGAN without ConvTranspose2D
class netD_nvidia_Generic(nn.Module):
    def __init__(self, args):
        super(netD_nvidia_Generic, self).__init__()

        self.ngpu = args.ngpu
        self.nc = args.nchannels
        self.ndf = args.ndf
        self.down = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)
        self.fullyconnected = nn.Linear(2048, 1)
        self.height = args.resolution_high
        self.width = args.resolution_wide
        self.extra_cap = args.extra_D_cap
        assert self.height == self.width, "Image height is not equal to width"

        count = 1
        layers = []
        layers.append(self.fromRGB(self.ndf))

        while count < self.height:
            layers.append(NormConvBlock(count*self.ndf, 2*count*self.ndf, kernel_size=3, stride=1, padding=1))
            if self.extra_cap:
                layers.append(NormConvBlock(2*count*self.ndf, 2*count*self.ndf, kernel_size=3, stride=1, padding=1))
            layers.append(self.down)
            count *= 2

        # layers.append(PixelNormLayer())
        self.main = nn.Sequential(*layers)

    def fromRGB(self, output_channels):
        return NormConvBlock(self.nc, output_channels, kernel_size=1, stride=1, padding=0, pixelnorm=False, activation=False)

    def forward(self, input, count=0):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            output = nn.parallel.data_parallel(self.fullyconnected, output.squeeze(), range(self.ngpu))
        else:
            output = self.main(input)
            output = self.fullyconnected(output.squeeze())

        return output.view(-1)


# Generic DCGAN without ConvTranspose2D
class netE_Generic(nn.Module):
    def __init__(self, args):
        super(netE_Generic, self).__init__()

        self.ngpu = args.ngpu
        self.nc = args.nchannels
        self.ndf = args.ndf
        self.nz = args.nz
        self.pixelnorm = PixelNormLayer()
        self.normalize = args.normalize
        self.norm_type = args.norm_type
        try:
            self.norm = {'batch': nn.BatchNorm2d, 'lrn': nn.LocalResponseNorm, 'instance': nn.InstanceNorm2d, 'layer': LayerNorm}
        except Exception as e:
            self.norm = {'batch': nn.BatchNorm2d, 'instance': nn.InstanceNorm2d, 'layer': LayerNorm}
        self.height = args.resolution_high
        self.width = args.resolution_wide
        assert self.height == self.width, "Image height is not equal to width"

        count = 1
        layers = []
        layers.append(nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False))
        if self.normalize:
            layers.append(nn.LeakyReLU(0.2))
        else:
            layers.append(nn.SELU())
        # layers.append(WScaleLayer(size=self.ndf))
        while count < self.height/4:
            layers.append(nn.Conv2d(count*self.ndf, 2*count*self.ndf, 4, 2, 1, bias=False))
            if self.normalize:
                if self.norm_type == 'lrn':
                    layers.append(nn.LocalResponseNorm(1))
                else:
                    layers.append(self.norm[self.norm_type](2*count*self.ndf))
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.SELU())
            # layers.append(WScaleLayer(size=2*count*self.ndf))
            # layers.append(self.pixelnorm)
            count *= 2
        layers.append(nn.Conv2d(count*self.ndf, self.nz, 2, 1, 0, bias=False))
        self.main = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(self.nz, self.nz)
        self.fc_sigma = nn.Linear(self.nz, self.nz)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            hidden = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            hidden = self.main(input)

        hidden = hidden.view(-1, self.nz)
        z_mu = self.fc_mu(hidden)
        z_sigma = self.fc_sigma(hidden)
        return z_mu, z_sigma

class netG_Generic_2(nn.Module):
    def __init__(self, args):
        super(netG_Generic, self).__init__()

        self.ngpu = args.ngpu
        self.nc = args.nchannels
        self.ngf = args.ngf
        self.nz = args.nz
        self.up = Upsample(scale_factor = 2)
        self.pixelnorm = PixelNormLayer()
        self.batch_norm = args.batch_norm
        self.height = args.resolution_high
        self.width = args.resolution_wide
        assert self.height == self.width, "Image height is not equal to width"

        count = int(self.height/4)
        layers = []
        layers.append(self.up)
        layers.append(nn.Conv2d(self.nz, count*self.ngf, 3, 1, 1, bias=False))
        if self.batch_norm:
            layers.append(nn.BatchNorm2d(count*self.ngf))
            layers.append(nn.ReLU())
        else:
            layers.append(nn.SELU())
        # layers.append(WScaleLayer(size=count*self.ngf))
        # layers.append(self.pixelnorm)
        while count > 1:
            count = int(count/2)
            layers.append(self.up),
            layers.append(nn.Conv2d(2*count*self.ngf, count*self.ngf, 3, 1, 1, bias=False))
            if self.batch_norm:
                layers.append(nn.BatchNorm2d(count*self.ngf))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.SELU())
            # layers.append(WScaleLayer(size=count*self.ngf))
            # layers.append(self.pixelnorm)
        layers.append(self.up),
        layers.append(nn.Conv2d(self.ngf, self.nc, 3, 1, 1, bias=False))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

# Generic DCGAN without ConvTranspose2D
class netD_Generic_2(nn.Module):
    def __init__(self, args):
        super(netD_Generic, self).__init__()

        self.ngpu = args.ngpu
        self.nc = args.nchannels
        self.ndf = args.ndf
        self.pixelnorm = PixelNormLayer()
        self.normalize = args.normalize
        self.norm_type = args.norm_type
        try:
            self.norm = {'batch': nn.BatchNorm2d, 'lrn': nn.LocalResponseNorm, 'instance': nn.InstanceNorm2d, 'layer': LayerNorm}
        except Exception as e:
            self.norm = {'batch': nn.BatchNorm2d, 'instance': nn.InstanceNorm2d, 'layer': LayerNorm}
        self.height = args.resolution_high
        self.width = args.resolution_wide
        self.extra_cap = args.extra_D_cap
        assert self.height == self.width, "Image height is not equal to width"

        count = 1
        layers = []
        layers.append(nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False))
        if self.normalize:
            layers.append(nn.LeakyReLU(0.2))
        else:
            layers.append(nn.SELU())
        # layers.append(WScaleLayer(size=self.ndf))
        while count < self.height/4:
            layers.append(nn.Conv2d(count*self.ndf, 2*count*self.ndf, 4, 2, 1, bias=False))
            if self.normalize:
                if self.norm_type == 'lrn':
                    layers.append(nn.LocalResponseNorm(1))
                else:
                    layers.append(self.norm[self.norm_type](2*count*self.ndf))
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.SELU())
            if self.extra_cap:
                layers.append(nn.Conv2d(2*count*self.ndf, 2*count*self.ndf, 3, 1, 1, bias=False))
                if self.normalize:
                    if self.norm_type == 'lrn':
                        layers.append(nn.LocalResponseNorm(1))
                    else:
                        layers.append(self.norm[self.norm_type](2*count*self.ndf))
                    layers.append(nn.LeakyReLU(0.2))
                else:
                    layers.append(nn.SELU())
            # layers.append(WScaleLayer(size=2*count*self.ndf))
            # layers.append(self.pixelnorm)
            count *= 2
        layers.append(nn.Conv2d(count*self.ndf, 1, 2, 1, 0, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1)

# GMM Model
class GMM_D(nn.Module):
    def __init__(self, args):
        super(GMM_D, self).__init__()
        self.ninput = args.gmm_dim
        self.nhidden = args.gmm_hidden
        self.nlayers = args.gmm_nlayers

        layers = []
        layers.append(nn.Linear(self.ninput, self.nhidden))
        layers.append(nn.SELU())

        for i in range(self.nlayers-2+2):
            layers.append(nn.Linear(self.nhidden, self.nhidden))
            layers.append(nn.SELU())

        layers.append(nn.Linear(self.nhidden, 1))
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        output = self.main(input).squeeze()
        return output

class GMM_G(nn.Module):
    def __init__(self, args):
        super(GMM_G, self).__init__()
        self.nlatent = args.nz
        self.nhidden = args.gmm_hidden
        self.nlayers = args.gmm_nlayers
        self.noutput = args.gmm_dim

        layers = []
        layers.append(nn.Linear(self.nlatent, self.nhidden))
        layers.append(nn.SELU())

        for i in range(self.nlayers-2+1):
            layers.append(nn.Linear(self.nhidden, self.nhidden))
            layers.append(nn.SELU())

        layers.append(nn.Linear(self.nhidden, self.noutput))
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        output = self.main(input)
        return output

class GMM_E(nn.Module):
    def __init__(self, args):
        super(GMM_E, self).__init__()
        self.ninput = args.gmm_dim
        self.nhidden = args.gmm_hidden
        self.nlayers = args.gmm_nlayers
        self.nlatent = args.nz

        self.linear_mu = nn.Linear(self.nlatent, self.nlatent)
        self.linear_sigma = nn.Linear(self.nlatent, self.nlatent)

        layers = []
        layers.append(nn.Linear(self.ninput, self.nhidden))
        layers.append(nn.SELU())

        for i in range(self.nlayers-2):
            layers.append(nn.Linear(self.nhidden, self.nhidden))
            layers.append(nn.SELU())

        layers.append(nn.Linear(self.nhidden, self.nlatent))
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        output = self.main(input)

        z_mu = self.linear_mu(output)
        z_sigma = self.linear_sigma(output)

        return z_mu, z_sigma

# # Generic DCGAN without ConvTranspose2D
class netG_Generic(nn.Module):
    def __init__(self, args):
        super(netG_Generic, self).__init__()

        self.ngpu = args.ngpu
        self.nc = args.nchannels
        self.ngf = args.ngf
        self.nz = args.nz
        self.up = Upsample(scale_factor = 2)
        self.height = args.resolution_high
        self.width = args.resolution_wide
        self.extra_cap = args.extra_G_cap
        self.normalize = args.normalize
        self.norm_type = args.norm_type
        assert self.height == self.width, "Image height is not equal to width"

        count = int(self.height/2)
        layers = []
        layers.append(self.up)
        layers.append(GenericConvBlock(self.nz, count*self.ngf, kernel_size=3, stride=1, padding=1, normalize=self.normalize, norm_type=self.norm_type))
        if self.extra_cap:
            layers.append(GenericConvBlock(count*self.ngf, count*self.ngf, kernel_size=3, stride=1, padding=1, normalize=self.normalize, norm_type=self.norm_type))
        while count > 1:
            count = int(count/2)
            layers.append(self.up)
            layers.append(GenericConvBlock(2*count*self.ngf, count*self.ngf, kernel_size=3, stride=1, padding=1, normalize=self.normalize, norm_type=self.norm_type))
            if self.extra_cap:
                layers.append(GenericConvBlock(count*self.ngf, count*self.ngf, kernel_size=3, stride=1, padding=1, normalize=self.normalize, norm_type=self.norm_type))
        layers.append(self.toRGB(self.ngf))
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def toRGB(self, input_channels):
        return GenericConvBlock(input_channels, self.nc, kernel_size=1, stride=1, padding=0, normalize=self.normalize, norm_type=self.norm_type, activation=False)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

# Generic DCGAN without ConvTranspose2D
class netD_Generic(nn.Module):
    def __init__(self, args):
        super(netD_Generic, self).__init__()

        self.ngpu = args.ngpu
        self.nc = args.nchannels
        self.ndf = args.ndf
        self.down = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)
        self.fullyconnected = nn.Linear(2048, 1)
        self.height = args.resolution_high
        self.width = args.resolution_wide
        self.extra_cap = args.extra_D_cap
        self.normalize = args.normalize
        self.norm_type = args.norm_type
        assert self.height == self.width, "Image height is not equal to width"

        count = 1
        layers = []
        layers.append(self.fromRGB(self.ndf))

        while count < self.height:
            layers.append(GenericConvBlock(count*self.ndf, 2*count*self.ndf, kernel_size=3, stride=1, padding=1, normalize=self.normalize, norm_type=self.norm_type))
            if self.extra_cap:
                layers.append(GenericConvBlock(2*count*self.ndf, 2*count*self.ndf, kernel_size=3, stride=1, padding=1, normalize=self.normalize, norm_type=self.norm_type))
            layers.append(self.down)
            count *= 2

        # layers.append(PixelNormLayer())
        self.main = nn.Sequential(*layers)

    def fromRGB(self, output_channels):
        return GenericConvBlock(self.nc, output_channels, kernel_size=1, stride=1, padding=0, normalize=self.normalize, norm_type=self.norm_type, activation=False)

    def forward(self, input, count=0):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            output = nn.parallel.data_parallel(self.fullyconnected, output.squeeze(), range(self.ngpu))
        else:
            output = self.main(input)
            output = self.fullyconnected(output.squeeze())

        return output.view(-1)

class netD_Generic_Cap(nn.Module):
    def __init__(self, args):
        super(netD_Generic_Cap, self).__init__()

        self.ngpu = args.ngpu
        self.nc = args.nchannels
        self.ndf = args.ndf
        self.down = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)
        self.linear1 = self.Linear(2048, 1024)
        self.final_layer = nn.Linear(1024, 1)
        self.height = args.resolution_high
        self.width = args.resolution_wide
        self.extra_cap = args.extra_D_cap
        self.normalize = args.normalize
        self.norm_type = args.norm_type
        assert self.height == self.width, "Image height is not equal to width"
        self.extra_layer = [self.Linear(1024, 1024).cuda() for i in range(5)]

        count = 1
        layers = []
        layers.append(self.fromRGB(self.ndf))

        while count < self.height:
            layers.append(GenericConvBlock(count*self.ndf, 2*count*self.ndf, kernel_size=3, stride=1, padding=1, normalize=self.normalize, norm_type=self.norm_type))
            if self.extra_cap:
                layers.append(GenericConvBlock(2*count*self.ndf, 2*count*self.ndf, kernel_size=3, stride=1, padding=1, normalize=self.normalize, norm_type=self.norm_type))
            layers.append(self.down)
            count *= 2

        # layers.append(PixelNormLayer())
        self.main = nn.Sequential(*layers)

    def fromRGB(self, output_channels):
        return GenericConvBlock(self.nc, output_channels, kernel_size=1, stride=1, padding=0, normalize=self.normalize, norm_type=self.norm_type, activation=False)

    def Linear(self, input_channels, output_channels, activation=True):
        layers = []
        layers.append(nn.Linear(input_channels, output_channels))
        layers[0].weight.data.copy_(torch.eye(output_channels, input_channels))
        if activation:
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, input, count=0, gamma=0):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            output = nn.parallel.data_parallel(self.linear1, output.squeeze(), range(self.ngpu))
            for i in range(count):
                output = nn.parallel.data_parallel(self.extra_layer[i], output, range(self.ngpu))
            output = nn.parallel.data_parallel(self.final_layer, output, range(self.ngpu))
        else:
            output = self.main(input)
            output = output.squeeze()
            output = self.linear1(output)
            for i in range(count):
                output = self.extra_layer[i](output)
            output = self.final_layer(output)

        return output.view(-1)

# Generic DCGAN without ConvTranspose2D
class netD_nvidia_Generic_Cap(nn.Module):
    def __init__(self, args):
        super(netD_nvidia_Generic_Cap, self).__init__()

        self.ngpu = args.ngpu
        self.nc = args.nchannels
        self.ndf = args.ndf
        self.down = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=False, count_include_pad=False)
        self.linear1 = self.Linear(2048, 1024)
        self.final_layer = nn.Linear(1024, 1)
        self.height = args.resolution_high
        self.width = args.resolution_wide
        self.extra_cap = args.extra_D_cap
        assert self.height == self.width, "Image height is not equal to width"
        self.extra_layer = [self.Linear(1024, 1024).cuda() for i in range(5)]

        count = 1
        layers = []
        layers.append(self.fromRGB(self.ndf))

        # while count < self.height:
        #     layers.append(NormConvBlock(count*self.ndf, 2*count*self.ndf, kernel_size=3, stride=1, padding=1))
        #     if self.extra_cap:
        #         layers.append(NormConvBlock(2*count*self.ndf, 2*count*self.ndf, kernel_size=3, stride=1, padding=1))
        #     layers.append(self.down)
        #     count *= 2

        while count < self.height:
            if count == self.height/2 and args.mini_batch_disc:
                layers.append(MinibatchStatConcatLayer())
                layers.append(NormConvBlock(count*self.ndf + 1, 2*count*self.ndf, kernel_size=3, stride=1, padding=1))
            else:
                layers.append(NormConvBlock(count*self.ndf, 2*count*self.ndf, kernel_size=3, stride=1, padding=1))
            if self.extra_cap:
                layers.append(NormConvBlock(2*count*self.ndf, 2*count*self.ndf, kernel_size=3, stride=1, padding=1))
            layers.append(self.down)
            count *= 2

        # layers.append(PixelNormLayer())
        self.main = nn.Sequential(*layers)

    def fromRGB(self, output_channels):
        return NormConvBlock(self.nc, output_channels, kernel_size=1, stride=1, padding=0, pixelnorm=False, activation=False)

    def Linear(self, input_channels, output_channels, activation=True):
        layers = []
        layers.append(nn.Linear(input_channels, output_channels))
        layers[0].weight.data.copy_(torch.eye(output_channels, input_channels))
        if activation:
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, input, count=0, gamma=1):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            output = nn.parallel.data_parallel(self.linear1, output.squeeze(), range(self.ngpu))
            for i in range(count-1):  # count-1
                output = nn.parallel.data_parallel(self.extra_layer[i], output, range(self.ngpu))
            if count-1 >= 0:
                new_layer_out = nn.parallel.data_parallel(self.extra_layer[count-1], output, range(self.ngpu))
                output = (1-gamma)*output + gamma*new_layer_out
            output = nn.parallel.data_parallel(self.final_layer, output, range(self.ngpu))
        else:
            output = self.main(input)
            output = output.squeeze()
            output = self.linear1(output)
            for i in range(count):
                output = self.extra_layer[i](output)
            output = self.final_layer(output)

        return output.view(-1)


# Simple DCGAN
class DCGAN_D(nn.Module):
    def __init__(self, args, n_extra_layers=0):
        super(DCGAN_D, self).__init__()
        self.ngpu = args.ngpu
        nc = args.nchannels
        ndf = args.ndf
        isize = args.resolution_high
        n_extra_layers = args.n_extra_layers

        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial.conv.{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial.relu.{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid.{0}-{1}.conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid.{0}.relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final.{0}-{1}.conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main


    def forward(self, input, extra_layer=0, extra_layer_gamma=0):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        # output = output.mean(0)
        return output.view(-1)

class DCGAN_G(nn.Module):
    def __init__(self, args, n_extra_layers=0):
        super(DCGAN_G, self).__init__()
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
        main.add_module('initial.{0}-{1}.convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial.{0}.batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial.{0}.relu'.format(cngf),
                        nn.ReLU(True))

        csize, cndf = 4, cngf
        while csize < isize//2:
            if use_upsampling:
                main.add_module('pyramid.{0}-{1}.upsample'.format(cngf, cngf//2),
                                Upsample(2))
                main.add_module('pyramid.{0}-{1}.convt'.format(cngf, cngf//2),
                                nn.Conv2d(cngf, cngf//2, 5, 1, 2, bias=False))
            else:
                main.add_module('pyramid.{0}-{1}.convt'.format(cngf, cngf//2),
                                nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid.{0}.relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final.{0}-{1}.convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final.{0}.tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
