import torch
import torch.nn as nn
from torch.autograd import Variable

class Normalize(nn.Module):
    def __init__(self, p):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        if x.dim()==1:
            if self.p == float('inf'):
                norm = torch.max(x.abs())
            else:
                norm = torch.norm(x, self.p, 0)
            x.div_(norm.data[0])
            return x
        elif x.dim()==2:
            if self.p == float('inf'):
                x = x.abs()
                norm = Variable(torch.Tensor(x.size(0), 1))
                for i in range(norm.size(0)):
                    norm[i, 0] = torch.max(x[i])
            else:
                norm = torch.norm(x, self.p, 1)     # batch_size * 1

            for i in range(x.size(0)):
                x[i].div_(norm.data[i][0])
            return x
        else:
            raise RuntimeError("expected dim=1 or 2, got {}".format(x.dim()))