import torch.nn as nn
from functools import reduce


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        # result is Variables list [Variable1, Variable2, ...]
        return map(self.lambda_func,self.forward_prepare(input))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        # result is a Variable
        return reduce(self.lambda_func,self.forward_prepare(input))