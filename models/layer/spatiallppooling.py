import torch.nn as nn


class SpatialLPPooling(nn.Module):
    def __init__(self, pnorm, kW, kH, dW, dH):
        super(SpatialLPPooling, self).__init__()
        self.pnorm = pnorm
        self.scale = kW * kH

        self.pool = nn.AvgPool2d((kH, kW), (dH, dW))

    def forward(self, x):
        x = x ** self.pnorm
        x = self.pool(x)
        x *= self.scale
        x = x ** (1.0/self.pnorm)
        return x