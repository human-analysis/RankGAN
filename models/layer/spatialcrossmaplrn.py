import torch
import torch.nn as nn
from torch.autograd import Variable


class SpatialCrossMapLRN(nn.Module):
    def __init__(self, size, alpha=0.0001, beta=0.75, k=1):
        super(SpatialCrossMapLRN, self).__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        # transfer to 4D
        assert x.dim() == 3 or x.dim() == 4, 'support only 3D or 4D input'
        is_batch = True
        if x.dim() == 3:
            x = torch.unsqueeze(x, 0)
            is_batch = False

        layer_square = x * x

        pad = int((self.size - 1) / 2 + 1)
        channels = x.size(1)
        if pad > channels:    # pad_crop: input range of first layer
            pad_crop = channels
        else:
            pad_crop = pad

        # sum for first layer
        list_sum = []
        current_sum = Variable(x.data.new(x.size(0), 1, x.size(2), x.size(3)))
        current_sum.data.zero_()
        for c in range(pad_crop):
            current_sum = current_sum + layer_square.select(1, c)
        list_sum.append(current_sum)

        # sum other layers by 'add + remove'
        for c in range(1, channels):
            current_sum = list_sum[c-1]    # copy previous sum
            # add
            if c < channels - pad + 1:
                index_next = c + pad - 1
                current_sum = current_sum + layer_square.select(1, index_next)
            # remove
            if c > pad:
                index_prev = c - pad
                current_sum = current_sum - layer_square.select(1, index_prev)
            list_sum.append(current_sum)
        layer_square_sum = torch.cat(list_sum, 1)

        # y = x / { (k + alpha / size * sum)^beta }
        layer_square_sum = layer_square_sum * self.alpha / self.size + self.k
        output = x * torch.pow(layer_square_sum, -self.beta)

        # recover 3D
        if not is_batch:
            output = torch.squeeze(output, 1)
        return output