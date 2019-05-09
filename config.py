# config.py

import os
import datetime
import argparse
import json
import configparser
import utils
import re
from ast import literal_eval as make_tuple

class Argument:
    def _init_arg(self, value):
        if isinstance(value, dict):
            return Argument(**value)
        else:
            return value

    def __init__(self,**kwargs):
        for k,v in kwargs.items():
            setattr(self, k, self._init_arg(v))

    def loads(args):
        pattern = re.compile('^\(.+\)')
        arg_values = make_tuple(args)
        for key, value in arg_values.items():
            if isinstance(value, str) and pattern.match(value):
                arg_values[key] = make_tuple(value)
        return Argument(**arg_values)

    def __getitem__(self,item):
        return self.__dict__.get(item,None)

    def __repr__(self):
        return "%s" % self.__dict__

def parse_args():
    result_path = "results/"
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_path = os.path.join(result_path, now)

    parser = argparse.ArgumentParser(description='Object Detection/Alignment')

    # the following two parameters can only be provided at the command line, because
    # there are some parameters that are relying on these values, such as save-dir.
    parser.add_argument("-c", "--config", "--args-file", dest="config_file", default="args.txt", help="Specify a config file", metavar="FILE")
    args, remaining_argv = parser.parse_known_args()

    parser = argparse.ArgumentParser(description='Your project title goes here')

    # ======================== Data Setings ============================================
    parser.add_argument('--dataset-test', type=str, default='CELEBA', metavar='', help='name of training dataset')
    parser.add_argument('--dataset-train', type=str, default='CELEBA', metavar='', help='name of training dataset')
    parser.add_argument('--split_test', type=float, default=None, metavar='', help='test split')
    parser.add_argument('--split_train', type=float, default=None, metavar='', help='train split')
    parser.add_argument('--dataroot', type=str, default=None, metavar='', help='path to the data')
    parser.add_argument('--result-path', type=str, default=result_path, help='path where to save results')
    parser.add_argument('--add-timestamp', type=utils.str2bool, default='Yes', metavar='', help='add timestamp to result path')
    parser.add_argument('--resume', type=str, default=None, metavar='', help='full path of models to resume training')
    parser.add_argument('--nclasses', type=int, default=None, metavar='', help='number of classes for classification')
    parser.add_argument('--input-filename-test', type=str, default=None, metavar='', help='input test filename for filelist and folderlist')
    parser.add_argument('--label-filename-test', type=str, default=None, metavar='', help='label test filename for filelist and folderlist')
    parser.add_argument('--input-filename-train', type=str, default=None, metavar='', help='input train filename for filelist and folderlist')
    parser.add_argument('--label-filename-train', type=str, default=None, metavar='', help='label train filename for filelist and folderlist')
    parser.add_argument('--loader-input', type=str, default=None, metavar='', help='input loader')
    parser.add_argument('--loader-label', type=str, default=None, metavar='', help='label loader')
    parser.add_argument('--prefetch', type=utils.str2bool, default='No', help='whether to prefetch data onto memory')

    # ======================== Network Model Setings ===================================
    parser.add_argument('--nchannels', type=int, default=3, metavar='', help='number of input channels')
    parser.add_argument('--resolution-high', type=int, default=64, metavar='', help='image resolution height')
    parser.add_argument('--resolution-wide', type=int, default=64, metavar='', help='image resolution width')
    parser.add_argument('--ndim', type=int, default=100, metavar='', help='number of feature dimensions')
    parser.add_argument('--nunits', type=int, default=None, metavar='', help='number of units in hidden layers')
    parser.add_argument('--dropout', type=float, default=None, metavar='', help='dropout parameter')
    parser.add_argument('--net-type', type=str, default='dcgan_nvidia', metavar='', help='type of network')
    parser.add_argument('--length-scale', type=float, default=None, metavar='', help='length scale')
    parser.add_argument('--tau', type=float, default=None, metavar='', help='Tau')
    parser.add_argument('--mini-batch-disc', action='store_true', default=True, help='enable minibatch discrimination in discriminator')

    # ======================== Training Settings =======================================
    parser.add_argument('--cuda', type=utils.str2bool, default=False, help='run on gpu')
    parser.add_argument('--ngpu', type=int, default=1, metavar='', help='number of gpus to use')
    parser.add_argument('--batch-size', type=int, default=32, metavar='', help='batch size for training')
    parser.add_argument('--nepochs', type=int, default=200, metavar='', help='number of epochs to train')
    parser.add_argument('--niters', type=int, default=None, metavar='', help='number of iterations at test time')
    parser.add_argument('--epoch-number', type=int, default=None, metavar='', help='epoch number')
    parser.add_argument('--nthreads', type=int, default=10, metavar='', help='number of threads for data loading')
    parser.add_argument('--manual-seed', type=int, default=101, metavar='', help='manual seed for randomness')
    parser.add_argument('--port', type=int, default=8097, metavar='', help='port for visualizing training at http://localhost:port')
    parser.add_argument('--env', type=str, default='main', help='visdom environment name')
    parser.add_argument('--same-env', type=utils.str2bool, default='No', metavar='', help='does not add date and time to the visdom environment name')
    parser.add_argument('--dataset-fraction', type=float, default=1, help='fraction of dataset to train (between 0-1)')
    parser.add_argument('--plot-update-interval', type=int, default=30, help='number of iterations per plot update')

    # ======================== Hyperparameter Setings ==================================
    parser.add_argument('--optim-method', type=str, default='Adam', metavar='', help='the optimization routine ')
    parser.add_argument('--learning-rate-vae', type=float, default=1e-4, metavar='', help='learning rate for vae')
    parser.add_argument('--learning-rate-dis', type=float, default=5e-5, metavar='', help='learning rate for discriminator')
    parser.add_argument('--learning-rate-gen', type=float, default=1e-7, metavar='', help='learning rate for generator')
    parser.add_argument('--learning-rate-decay', type=float, default=0.8, metavar='', help='learning rate decay')
    parser.add_argument('--momentum', type=float, default=0, metavar='', help='momentum')
    parser.add_argument('--weight-decay', type=float, default=0, metavar='', help='weight decay')
    parser.add_argument('--stage1-weight-decay', type=float, default=0.5, metavar='', help='stage 1 weight decay for hinge loss')
    parser.add_argument('--adam-beta1', type=float, default=0.5, metavar='', help='Beta 1 parameter for Adam')
    parser.add_argument('--adam-beta2', type=float, default=0.999, metavar='', help='Beta 2 parameter for Adam')
    parser.add_argument('--gp', type=utils.str2bool, default=True, help='use gradient penalty')
    parser.add_argument('--gp-lambda', type=float, default=10, help="gradient penalty lambda")
    parser.add_argument('--scheduler-patience', type=int, default=500, help='patience value for lr scheduler')
    parser.add_argument('--scheduler-maxlen', type=int, default=1000, help='patience value for lr scheduler')
    parser.add_argument('--identity-lambda', type=float, default=1, help='weight of identity loss')
    parser.add_argument('--pose-lambda', type=float, default=1, help='weight of pose loss')

    # ======================== GoGAN Setings ==================================
    parser.add_argument('--stage-epochs', type=utils.str2list, default=None, help='number of epochs per gogan stage')
    parser.add_argument('--num-stages', type=int, default=3, help='number of gogan stages')
    parser.add_argument('--margin', type=float, default=2.0, help='initial margin of gogan loss')
    parser.add_argument('--weight-gan-final', type=float, default=1.0, help='weight of discriminator loss')
    parser.add_argument('--weight-vae', type=float, default=1.0, help='weight of mse loss')
    parser.add_argument('--ngf', type=int, default=32)
    parser.add_argument('--ndf', type=int, default=32)
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--clamp-lower', type=float, default=-0.01, help='WGAN lower weight clip')
    parser.add_argument('--clamp-upper', type=float, default=0.01, help='WGAN upper weight clip')
    parser.add_argument('--d-iter', type=int, default=5, help='number of discriminator iterations per generation iteration')
    parser.add_argument('--g-iter', type=int, default=1, help='number of generator iterations per discriminator iteration')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--netE', default='', help="path to Encoder (to continue training)")
    parser.add_argument('--prevD', default='', help="path to prevD (to continue training)")
    parser.add_argument('--prevG', default='', help="path to prevG (to continue training)")
    parser.add_argument('--vae-loss-type', default='l2', help='type of vae loss l1 or l2')
    parser.add_argument('--disc-diff-weight', type=float, default=1.0, help="weightage of discriminator difference loss")
    parser.add_argument('--weight-kld', type=float, default=10.0, help='weightage of kl-divergence loss')
    parser.add_argument('--start-stage', type=int, default=0, help='starging stage (0/1/2/...)')
    parser.add_argument('--normalize', action='store_true', default=False, help='whether to have batch-norm')
    parser.add_argument('--gogan-type', type=str, default="vae", help="no_vae/vae_no_gen")
    parser.add_argument('--norm-type', type=str, default='batch', help="type of normalization to use in models")
    parser.add_argument('--wgan', type=utils.str2bool, default=False, help='whether to use wgan loss in first stage of GAN')
    parser.add_argument('--extra-D-cap', type=utils.str2bool, default=True, help='whether to add extra capacity to the discriminator')
    parser.add_argument('--extra-G-cap', type=utils.str2bool, default=True, help='whether to add extra capacity to the generator')
    parser.add_argument('--correlation-sigma', type=float, default=10.0, help='variance of impulse in correlation loss for VAE')
    parser.add_argument('--add-capacity', action='store_true', default=False, help='whether to add extra layer to the discriminator to increase capacity')
    parser.add_argument('--add-clamp', action='store_true', default=False, help='whether to change clamping in the discriminator to increase capacity')
    parser.add_argument('--disc-optimize', action='store_true', default=False, help='optimize discriminator before training gogan')
    parser.add_argument('--gen-gamma', type=float, default=0, help='curriculum learning gamma')
    parser.add_argument('--add-noise', action='store_true', default=False, help='adds noise to the discriminator input')
    parser.add_argument('--noise-var', type=float, default=0.1, help='std of noise to be added to GAN training')
    parser.add_argument('--gp-norm', action='store_true', default=False, help='penalizes sum of gradient squares')
    parser.add_argument('--rank-weight', type=float, default=1, help='weight of discriminator ranking loss')
    parser.add_argument('--adaptive-iter', action='store_false', default=True, help='enables adaptive iterations for discriminator and generator')
    parser.add_argument('--use-upsampling', type=utils.str2bool, default=False, help='use upsampling in dcgan')
    parser.add_argument('--optimize-mse', action='store_true', default=False, help='wheter to optimize mse during gogan training')
    parser.add_argument('--weight-mse', type=float, default=1, help='weight for mse loss during gogan training')
    parser.add_argument('--n-extra-layers', type=int, default=0, help='number of extra layers in DCGAN architecture')
    parser.add_argument('--nranks', type=int, default=1, help='number of rank orders in RankGAN')

    # ======================== Image Completion Setings ==================================
    parser.add_argument('--disc-loss-weight', type=float, default=0.1, help='weight for discriminator loss in image completion')
    parser.add_argument('--ssim-weight', type=float, default=1000, help='weight of ssim loss')
    parser.add_argument('--citers', type=int, default=100, help='number of iterations for image completion')
    parser.add_argument('--scale', type=float, default=0.2, help='mask scale for image completion')
    parser.add_argument('--use-encoder', type=utils.str2bool, default=True, help='whether to use encoder for image completion or not')
    parser.add_argument('--blend', action='store_true', default=False, help='enable poisson blending')
    parser.add_argument('--mask-type', type=str, default='central', help='mask type (central/periocular)')
    parser.add_argument('--netG1', default='', help="path to netG1 (to continue training)")
    parser.add_argument('--netG2', default='', help="path to netG2 (to continue training)")
    parser.add_argument('--netG3', default='', help="path to netG3 (to continue training)")
    parser.add_argument('--netG4', default='', help="path to netG4 (to continue training)")
    parser.add_argument('--netG5', default='', help="path to netG5 (to continue training)")
    parser.add_argument('--start-index', type=int, default=0, help="start index of images")
    parser.add_argument('--disc-type', type=str, default='wgan', help='discriminator loss type for image completion')

    # ======================== OpenFace Setings ==================================
    parser.add_argument('--model', type=str, default='', help="model path")
    parser.add_argument('--splits', type=int, default=1, help="number of splits for computing inception score")

    # ======================== GMM Setings ==================================
    parser.add_argument('--num-gaus', type=int, default=2, help='number of Gaussians in GMM')
    parser.add_argument('--gmm-dim', type=int, default=1, help='dimensionality of GMM')
    parser.add_argument('--num-samples', type=int, default=10000, help='number of GMM data samples to generate')
    parser.add_argument('--gmm-range', type=float, default=3.0, help='range of GMM')
    parser.add_argument('--gmm-hidden', type=int, default=8, help='dimensionality of hidden layers in GMM model')
    parser.add_argument('--gmm-nlayers', type=int, default=3, help='number of layers in GMM model')

    if os.path.exists(args.config_file):
        config = configparser.ConfigParser()
        config.read([args.config_file])
        defaults = dict(config.items("Arguments"))
        parser.set_defaults(**defaults)

    args = parser.parse_args(remaining_argv)

    # add date and time to the name of Visdom environment and the result
    if args.env is None:
        args.env = args.model_type
    if not args.same_env:
        args.env += '_' + now

    # add date and time to the result directory name
    if now not in args.result_path and args.add_timestamp:
        args.result_path = os.path.join(args.result_path, now)
    args.save_dir = os.path.join(args.result_path, 'Save')
    args.logs_dir = os.path.join(args.result_path, 'Logs')

    # refine tuple arguments: this section converts tuples that are
    #                         passed as string back to actual tuples.
    pattern = re.compile('^\(.+\)')

    for arg_name in vars(args):
        # print(arg, getattr(args, arg))
        arg_value = getattr(args, arg_name)
        if isinstance(arg_value, str) and pattern.match(arg_value):
            setattr(args, arg_name, make_tuple(arg_value))
            print(arg_name, arg_value)
        elif isinstance(arg_value, dict):
            dict_changed = False
            for key, value in arg_value.items():
                if isinstance(value, str) and pattern.match(value):
                    dict_changed = True
                    arg_value[key] = make_tuple(value)
            if dict_changed:
                setattr(args, arg_name, arg_value)

    return args
