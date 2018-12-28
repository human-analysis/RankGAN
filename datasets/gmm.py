import torch
import numpy as np
import torch.utils.data as data

class GMM(data.Dataset):
    def __init__(self, args):
        self.num_gaus = args.num_gaus
        self.dim = args.gmm_dim
        self.range = args.gmm_range
        self.num_samples = args.num_samples

        self.mu = (2*np.random.rand(self.num_gaus, self.dim)-1)*self.range
        self.sigma = np.random.rand(self.num_gaus, self.dim)*self.range*0.1
        self.mix = np.random.rand(self.num_gaus)
        self.mix = self.mix/np.sum(self.mix)
        self.mix = self.mix.tolist()
        self.data = self.sample(self.num_samples)

    def sample(self, N):
        ind = np.random.choice(self.num_gaus, N, self.mix)
        samples = np.zeros((N,self.dim))
        for i in np.arange(N):
            samples[i] = np.random.multivariate_normal(self.mu[ind[i]], np.diag(self.sigma[ind[i]]), 1)
        return samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target) where target is the index of the target category.
            target contains the cropped objects, camera and vehicle parameters
        """
        sample = self.data[index]
        return sample, 1

    def __len__(self):
        return self.num_samples

class GMM_Ring(data.Dataset):
    def __init__(self, args):
        self.dim = args.gmm_dim
        self.num_samples = args.num_samples
        self.range = args.gmm_range
        self.rad = 1/np.sqrt(2)

        self.mu = np.array([[0,-1],[-self.rad,-self.rad],[-1,0],[-self.rad,self.rad],
                           [0,1],[self.rad,self.rad],[1,0],[self.rad,-self.rad]])*self.range
        self.num_gaus = self.mu.shape[0]
        self.sigma = np.ones((self.num_gaus, self.dim))*0.1
        self.mix = np.ones(self.num_gaus)
        self.mix = self.mix/np.sum(self.mix)
        self.mix = self.mix.tolist()
        # self.data = self.sample(self.num_samples)

    def sample(self, N):
        ind = np.random.choice(self.num_gaus, N, self.mix)
        samples = np.zeros((N,self.dim))
        for i in np.arange(N):
            samples[i] = np.random.multivariate_normal(self.mu[ind[i]], np.diag(self.sigma[ind[i]]), 1)
        return samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target) where target is the index of the target category.
            target contains the cropped objects, camera and vehicle parameters
        """
        # sample = self.data[index]
        sample = self.sample(1)[0]
        return sample, 1

    def __len__(self):
        return self.num_samples
