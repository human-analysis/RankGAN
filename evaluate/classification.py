# classification.py

import torch

class Classification():

    def __init__(self, topk=(1,)):
        self.topk = topk

    def forward(self, output, target):
        """Computes the precision@k for the specified values of k"""
        maxk = max(self.topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in self.topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class Logits_Classification:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, output, target):
        pred = (output > self.threshold).type('torch.LongTensor')
        correct = (pred == target.type('torch.LongTensor'))
        num_correct = correct.sum()
        return num_correct.item() / (correct.shape[0] * correct.shape[1])
