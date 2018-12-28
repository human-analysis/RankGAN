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
import pytorch_ssim
import warnings
warnings.filterwarnings("ignore")
import time


class Evaluate():

    def __init__(self, args, netD, netG, netE):
        self.netD = netD
        self.netG = netG
        self.netE = netE
        for p in self.netD.parameters():
            p.requires_grad = False
        for p in self.netG.parameters():
            p.requires_grad = False

        self.args = args
        self.nchannels = args.nchannels
        self.resolution_high = args.resolution_high
        self.resolution_wide = args.resolution_wide
        self.nz = args.nz
        self.wcom = args.disc_loss_weight
        self.ssim_weight = args.ssim_weight
        self.cuda = args.cuda
        self.citers = args.citers
        self.lr = args.learning_rate_vae
        self.momentum = args.momentum
        self.batch_size = args.batch_size
        self.use_encoder = args.use_encoder
        self.start_index = args.start_index

        self.input = Variable(torch.FloatTensor(self.batch_size,self.nchannels,self.resolution_high,self.resolution_wide), requires_grad=False).cuda()
        self.input_occluded = Variable(torch.FloatTensor(self.batch_size,self.nchannels,self.resolution_high,self.resolution_wide), requires_grad=False).cuda()
        self.fake_out = Variable(torch.FloatTensor(self.batch_size,self.nchannels,self.resolution_high,self.resolution_wide), requires_grad=True).cuda()
        # self.recon_out = Variable(torch.FloatTensor(self.batch_size,self.nchannels,self.resolution_high,self.resolution_wide)).cuda()
        self.noise = torch.FloatTensor(self.batch_size, self.nz, 1, 1).normal_(0, 1).cuda()
        self.epsilon = Variable(torch.randn(self.batch_size, self.nz), requires_grad=False).cuda()
        self.noise = Variable(self.noise, requires_grad=True)

        self.optimizerC = optim.RMSprop([self.noise], lr=self.lr)
        self.scheduler = plugins.AutomaticLRScheduler(self.optimizerC, maxlen=500, factor=0.1, patience=self.args.scheduler_patience)
        self.log_eval_loss = plugins.Logger(args.logs, 'CompletionLog.txt')
        self.params_eval_loss = ['Image', 'Input_SSIM', 'Input_PSNR', 'SSIM', 'PSNR', 'C_Loss', 'P_Loss']
        self.log_eval_loss.register(self.params_eval_loss)

        # Create the mask
        # self.pmask = torch.ones(self.batch_size, self.nchannels, self.resolution_high, self.args.resolution_wide)
        # if args.mask_type == 'central':
        #     self.l = int(self.resolution_high*self.args.scale)
        #     self.u = int(self.resolution_wide*(1-self.args.scale))
        #     if self.l != self.u:
        #         self.pmask[:, :, 5+self.l:5+self.u, self.l:self.u] = 0.0
        # elif args.mask_type == 'periocular':
        #     self.pmask[:,:,int(0.4*self.resolution_high):,:] = 0.0
        #     self.pmask[:,:,:,:8] = 0.0
        #     self.pmask[:,:,:,56:] = 0.0
        # self.nmask = torch.add(-self.pmask, 1)
        self.pmask = torch.ones(self.resolution_high, self.args.resolution_wide)
        if args.mask_type == 'central':
            self.l = int(self.resolution_high*self.args.scale)
            self.u = int(self.resolution_wide*(1-self.args.scale))
            if self.l != self.u:
                self.pmask[5+self.l:5+self.u, self.l:self.u] = 0.0
        elif args.mask_type == 'periocular':
            self.pmask[int(0.4*self.resolution_high):,:] = 0.0
            # self.pmask[int(0.4*self.resolution_high):int(0.9*self.resolution_high),8:56] = 0.0
            if self.args.scale == 0.3:
                self.pmask[:,:8] = 0.0
                self.pmask[:,56:] = 0.0
        self.nmask = torch.add(-self.pmask, 1)
        self.non_mask_pixels = (self.nmask.view(-1) == 0).nonzero().squeeze()

        # create coefficient matrix
        self.num_pixels = self.resolution_high * self.resolution_wide
        A = scipy.sparse.identity(self.num_pixels, format='lil')
        for y in range(self.resolution_high):
            for x in range(self.resolution_wide):
                if self.nmask[y,x]:
                    index = x+y*self.resolution_wide
                    A[index, index] = 4
                    if index+1 < self.num_pixels:
                        A[index, index+1] = -1
                    if index-1 >= 0:
                        A[index, index-1] = -1
                    if index+self.resolution_wide < self.num_pixels:
                        A[index, index+self.resolution_wide] = -1
                    if index-self.resolution_wide >= 0:
                        A[index, index-self.resolution_wide] = -1
        A = torch.Tensor(A.toarray())
        # self.A = Variable(A)
        self.Ainv = Variable(torch.inverse(A))

        # Construct Poisson Matrix for Blending
        P = 4*torch.eye(self.num_pixels)
        diag = np.arange(0, self.num_pixels-1)
        P[diag, diag+1] = -1
        P[diag+1, diag] = -1
        diag = np.arange(self.resolution_high, self.num_pixels)
        P[diag - self.resolution_high, diag] = -1
        P[diag, diag - self.resolution_high] = -1
        self.P = Variable(P)

        self.criterion = nn.L1Loss()
        self.ssim_loss = pytorch_ssim.SSIM()
        if self.args.cuda == True:
            self.pmask = self.pmask.cuda()
            self.nmask = self.nmask.cuda()
            self.non_mask_pixels = self.non_mask_pixels.cuda()
            self.criterion = self.criterion.cuda()
            # self.A = self.A.cuda()
            self.Ainv = self.Ainv.cuda()
            self.P = self.P.cuda()

        self.pmask = Variable(self.pmask)
        self.nmask = Variable(self.nmask)
        self.blend = args.blend

        self.losses = {}

        self.log_eval_monitor = plugins.Monitor()
        self.params_eval_monitor = ['Image', 'Input_SSIM', 'Input_PSNR', 'SSIM', 'PSNR', 'C_Loss', 'P_Loss']
        self.log_eval_monitor.register(self.params_eval_monitor)

        self.print = '[%d/%d] [%d/%d]'
        for item in self.params_eval_loss:
            if item == 'Image':
                self.print = self.print + item + " %d "
            else:
                self.print = self.print + item + " %.4f "

        # self.visualizer = plugins.Visualizer(port=self.args.port, env=self.args.env, title='Image Completion')
        # self.visualizer_dict = {
        # 'Recon_SSIM': {'dtype':'scalar', 'vtype': 'plot', 'win': 'ssim'},
        # 'Recon_PSNR': {'dtype':'scalar', 'vtype': 'plot', 'win': 'psnr'},
        # 'Z_Norm': {'dtype':'scalar', 'vtype': 'plot', 'win': 'z_norm'},
        # 'Contextual_Loss': {'dtype':'scalar', 'vtype': 'plot', 'win': 'Contextual_Loss'},
        # 'Perceptual_Loss': {'dtype':'scalar', 'vtype': 'plot', 'win': 'Perceptual_Loss'},
        # # 'SSIM_Loss': {'dtype':'scalar', 'vtype': 'plot', 'win': 'SSIM_Loss'},
        # 'LR': {'dtype':'scalar', 'vtype': 'plot', 'win': 'lr'},
        # 'Original_Image': {'dtype':'images', 'vtype': 'images', 'win': 'input'},
        # 'Occluded_Image': {'dtype':'images', 'vtype': 'images', 'win': 'input_real'},
        # 'Fake_Image': {'dtype':'images', 'vtype': 'images', 'win': 'fake'},
        # 'Completed_Image': {'dtype':'images', 'vtype': 'images', 'win': 'completed'},
        # }
        # self.visualizer.register(self.visualizer_dict)

        self.c_loss = 0
        self.d_loss = 0
        self.disc_type = args.disc_type

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

    def loss(self, real, fake, mask):
        tmp1 = torch.mul(real, mask)
        tmp2 = torch.mul(fake, mask)
        loss = self.criterion(tmp2, tmp1)
        diff = tmp1 - tmp2
        # L2
        # loss = (0.5 * torch.mul(diff, diff)).mean(0).sum()
        # L1
        loss = torch.abs(diff).mean(0).sum()
        loss -= self.ssim_weight * self.ssim_loss(tmp1, tmp2)
        return loss

    def complete(self, dataloader):
        data_iter = iter(dataloader)

        data_i = 0
        while data_i < len(dataloader):
            self.optimizerC = optim.RMSprop([self.noise], lr=self.lr)
            self.scheduler = plugins.AutomaticLRScheduler(self.optimizerC, maxlen=500, factor=0.1, patience=self.args.scheduler_patience)

            data_real = data_iter.next()[0]
            data_i += 1
            if data_i < self.start_index:
                continue

            batch_size = data_real.size(0)
            self.input.data.resize_(data_real.size()).copy_(data_real)

            self.input_norm = 2*(torch.mul(0.5*(self.input+1), self.pmask))-1
            self.input_occluded = torch.mul(self.input, self.pmask)

            if self.use_encoder:
                self.epsilon.data.resize_(batch_size, self.nz).normal_(0, 1)
                noise_mu, noise_logvar = self.netE(self.input_occluded)
                noise_sigma = torch.exp(torch.mul(noise_logvar, 0.5))
                latents = noise_mu + torch.mul(noise_sigma, self.epsilon)
                latents = latents.unsqueeze(-1).unsqueeze(-1)
                self.noise.data.copy_(latents.data)
            else:
                self.noise.data.normal_(0, 1)

            # Save original images first
            for b in range(batch_size):
                save_image(normalize(self.input.data[b].cpu()), "{}/{}_Original.png".format(self.args.save, data_i*batch_size+b), padding=0, normalize=True)
                save_image(normalize(self.input_norm.data[b].cpu()), "{}/{}_Occluded.png".format(self.args.save, data_i*batch_size+b), padding=0, normalize=True)

            for j in range(self.citers):

                fake = self.netG.forward(self.noise)
                fake_out = fake.clone()
                # score = self.netD.forward(self.fake_out)
                score = self.netD.forward(fake)

                disc_loss = get_disc_loss(score, type=self.disc_type)
                Contextual_Loss = self.loss(self.input_occluded, fake, self.pmask)

                if not self.blend:
                    recon_out = torch.mul(self.input_norm, self.pmask) + torch.mul(self.fake_out, self.nmask)
                else:
                    recon_out = self.poisson_blending(self.input_norm, fake, self.nmask)
                disc_loss += get_disc_loss(self.netD.forward(recon_out), type=self.disc_type)
                # ssim_loss = - self.ssim_loss(self.input, fake)
                # Contextual_Loss = self.criterion(recon_out, self.input)
                # loss_vals = Parallel(n_jobs=3)(delayed(process_stage)(i) for i in zip(self.netG, self.netD, self.fake_out, [self.input_occluded]*3, [self.pmask]*3, self.noise, [self.loss]*3, [self.wcom]*3, self.optimizerC, self.scheduler))

                if j % 10 == 0 or j == self.citers-1:

                    # disc_loss -= self.netD.forward(recon_out).mean(0).view(1)
                    # ssims = self.ssim(self.input.data.cpu(), recon_out.data.cpu())
                    ssims = pytorch_ssim.ssim(self.input, recon_out).data[0]
                    psnrs = self.psnr(self.input.data.cpu(), recon_out.data.cpu())
                    # cl = self.loss(self.input, recon_out, self.pmask[0:batch_size]).data
                    cl = Contextual_Loss.data
                    dl = disc_loss.data
                    # cl = self.c_loss
                    # dl = self.d_loss
                    losses = {}
                    losses['Recon_SSIM'] = ssims
                    losses['Recon_PSNR'] = psnrs
                    losses['Z_Norm'] = self.noise.data.norm()
                    losses['Contextual_Loss'] = cl
                    losses['Perceptual_Loss'] = dl
                    # losses['SSIM_Loss'] = ssim_loss.data[0]
                    losses['LR'] = self.optimizerC.param_groups[0]['lr']
                    losses['Fake_Image'] = fake_out.data.cpu()
                    losses['Completed_Image'] = recon_out.data.cpu()
                    losses['Original_Image'] = self.input.data.cpu()
                    losses['Occluded_Image'] = self.input_norm.data.cpu()
                    # self.visualizer.update(losses)

                    self.losses['Image'] = float(data_i)
                    self.losses['Input_SSIM'] = self.ssim(self.input.data.cpu(), self.input_norm.data.cpu())
                    self.losses['Input_PSNR'] = self.psnr(self.input.data.cpu(), self.input_norm.data.cpu())
                    self.losses['SSIM'] = ssims
                    self.losses['PSNR'] = psnrs
                    self.losses['C_Loss'] = cl
                    self.losses['P_Loss'] = dl
                    self.log_eval_monitor.update(self.losses, batch_size, keepsame=True)
                    print(self.print % tuple([data_i, len(dataloader), j, self.citers] + [self.losses[key]
                                                       for key in self.params_eval_monitor]))
                    loss = self.log_eval_monitor.getvalues()
                    self.log_eval_loss.update(loss)

                loss = Contextual_Loss + self.wcom*disc_loss
                loss.backward()
                self.optimizerC.step()
                self.scheduler.step(disc_loss.data[0])
                self.optimizerC.zero_grad()

                if j % 250 == 0:
                # Save intermediate results
                    for b in range(batch_size):
                        save_image(normalize(fake_out.data[b].cpu()), "{}/{}_Fake_Stage_{}.png".format(self.args.save, data_i*batch_size+b, j), padding=0, normalize=True)
                        save_image(normalize(recon_out.data[b].cpu()), "{}/{}_Reconstructed_Stage_{}.png".format(self.args.save, data_i*batch_size+b, j), padding=0, normalize=True)

            # self.log_eval_image.update([data_real.clone(), self.input_occluded.data, self.recon_out.clone()])
            for b in range(batch_size):
                save_image(normalize(fake_out.data[b].cpu()), "{}/{}_Fake_Stage.png".format(self.args.save, data_i*batch_size+b), padding=0, normalize=True)
                save_image(normalize(recon_out.data[b].cpu()), "{}/{}_Reconstructed_Stage.png".format(self.args.save, data_i*batch_size+b), padding=0, normalize=True)

    def poisson_blending(self, target, source, mask, offset=(0,0)):
        # compute regions to be blended
        region_source = (
                max(-offset[0], 0),
                max(-offset[1], 0),
                min(target.shape[2]-offset[0], source.shape[2]),
                min(target.shape[3]-offset[1], source.shape[3]))
        region_target = (
                max(offset[0], 0),
                max(offset[1], 0),
                min(target.shape[2], source.shape[2]+offset[0]),
                min(target.shape[3], source.shape[3]+offset[1]))
        region_size = (region_source[2]-region_source[0], region_source[3]-region_source[1])
        num_pixels = int(np.prod(region_size))
        bsize = target.size(0)

        t = target[:, :, region_target[0]:region_target[2], region_target[1]:region_target[3]]
        s = source[:, :, region_source[0]:region_source[2], region_source[1]:region_source[3]]
        # t = t.transpose(0,1).transpose(1,2).contiguous().view(num_pixels, -1) # 4096 x 3
        # s = s.transpose(0,1).transpose(1,2).contiguous().view(num_pixels, -1) # 4096 x 3
        t = t.view(bsize, self.nchannels, -1).transpose(0, 2).contiguous().view(-1, self.nchannels*bsize) # 4096 x 3N
        s = s.view(bsize, self.nchannels, -1).transpose(0, 2).contiguous().view(-1, self.nchannels*bsize) # 4096 x 3N

        #create b
        b = torch.matmul(self.P, s)  # (4096 x 4096) x (4096 x 3N) = (4096 x 3N)
        b[self.non_mask_pixels] = t[self.non_mask_pixels]

        # # solve Ax=b
        # x = torch.gesv(b, self.A)[0]    # 4096 x 3N
        x = torch.matmul(self.Ainv, b) # 4096 x 3N

        # assign x to target image
        # x = x.transpose(0, 1).view(bsize, self.nchannels, region_size[0], region_size[1])
        x = x.transpose(0, 1).contiguous().view(self.nchannels, bsize, region_size[0], region_size[1]).transpose(0, 1)
        # x = x.transpose(0, 1).view(-1, region_size[0], region_size[1])
        # x.data.clamp_(-1, 1)
        # target[:, :, region_target[0]:region_target[2], region_target[1]:region_target[3]].data.copy_(x.data)

        return x


def blend(img_target, img_source, img_mask, offset=(0, 0)):
    # compute regions to be blended
    region_source = (
            max(-offset[0], 0),
            max(-offset[1], 0),
            min(img_target.shape[1]-offset[0], img_source.shape[1]),
            min(img_target.shape[2]-offset[1], img_source.shape[2]))
    region_target = (
            max(offset[0], 0),
            max(offset[1], 0),
            min(img_target.shape[1], img_source.shape[1]+offset[0]),
            min(img_target.shape[2], img_source.shape[2]+offset[1]))
    region_size = (region_source[2]-region_source[0], region_source[3]-region_source[1])
    # region_source = (offset[0], offset[0], offset[1], offset[1])
    # region_target = (offset[0], offset[0], offset[1], offset[1])
    # region_size = (region_source[2]-region_source[0], region_source[3]-region_source[1])

    # clip and normalize mask image
    img_mask = img_mask[region_source[0]:region_source[2], region_source[1]:region_source[3]]
    img_mask[img_mask==0] = False
    img_mask[img_mask!=False] = True

    # create coefficient matrix
    A = scipy.sparse.identity(np.prod(region_size), format='lil')
    for y in range(region_size[0]):
        for x in range(region_size[1]):
            if img_mask[y,x]:
                index = x+y*region_size[1]
                A[index, index] = 4
                if index+1 < np.prod(region_size):
                    A[index, index+1] = -1
                if index-1 >= 0:
                    A[index, index-1] = -1
                if index+region_size[1] < np.prod(region_size):
                    A[index, index+region_size[1]] = -1
                if index-region_size[1] >= 0:
                    A[index, index-region_size[1]] = -1
    A = A.tocsr()

    # create poisson matrix for b
    P = pyamg.gallery.poisson(img_mask.shape)

    # for each layer (ex. RGB)
    for num_layer in range(img_target.shape[0]):
        # get subimages
        t = img_target[num_layer, region_target[0]:region_target[2],region_target[1]:region_target[3]]
        s = img_source[num_layer, region_source[0]:region_source[2], region_source[1]:region_source[3]]
        t = t.flatten()
        s = s.flatten()

        # create b
        b = P * s
        for y in range(region_size[0]):
            for x in range(region_size[1]):
                if not img_mask[y,x]:
                    index = x+y*region_size[1]
                    b[index] = t[index]

        # solve Ax = b
        x = pyamg.solve(A,b,verb=False,tol=1e-10)

        # assign x to target image
        x = np.reshape(x, region_size)
        x[x>1] = 1
        x[x<-1] = -1
        x = np.array(x, img_target.dtype)
        img_target[num_layer, region_target[0]:region_target[2],region_target[1]:region_target[3]] = x

    return img_target

def unwrap_self(arg, **kwarg):
    return Evaluate.process_stage(*arg, **kwarg)


def process_stage(netG, netD, fake, real, mask, noise, loss, wcom, optimizer, scheduler):
    fake = netG.forward(noise)
    score = netD.forward(fake)
    Contextual_Loss = loss(real, fake, mask)
    disc_loss = -score.mean(0).view(1)
    loss = Contextual_Loss + wcom*disc_loss
    loss.backward()
    optimizer.step()
    scheduler.step(Contextual_Loss.data[0])
    return Contextual_Loss.data[0], disc_loss.data[0]

def normalize(image):
    image = (image - image.min())
    return image / image.max()

def get_disc_loss(scores, type='wgan'):
    if type == 'wgan':
        return -scores.mean(0).view(1)
    elif type == 'lsgan':
        diff = scores - 1
        loss = 0.5 * torch.pow(diff, 2).mean(0).sum()
        return loss
    else:
        raise("Disc loss should be specified")
