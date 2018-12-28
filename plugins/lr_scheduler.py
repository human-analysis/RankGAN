from collections import deque
import numpy as np
from scipy.stats import norm

class AutomaticLRScheduler:
    def __init__(self, optimizer=None, maxlen=None, patience=1000, factor=1):
        self.maxlen = maxlen                                            # n
        self.var_multiplier = 12 / (self.maxlen ** 3 - self.maxlen)     # 12 / (n^3 - n)
        self.loss_queue = deque(maxlen=maxlen)                          # y
        self.A = np.vstack([np.arange(maxlen), np.ones(maxlen)]).T      # y = mx + c = Ap, A = [[x 1]], p = [[m], [c]]
        self.p = np.array([0, 0])
        self.var = 0
        self.epsilon = 1e-8
        self.steps_without_decrease = 0
        self.patience = patience
        self.factor = factor
        self.optimizer = optimizer
        self.min_lr = optimizer.param_groups[0]['lr']*1e-2

    def step(self, loss):
        self.loss_queue.append(loss)
        if len(self.loss_queue) == self.maxlen:
            self.p = np.linalg.lstsq(self.A, np.array(self.loss_queue))[0]
            self.loss_pred = np.matmul(self.A, self.p)
            self.var = np.sum(np.power(np.array(self.loss_queue) - self.loss_pred, 2)) / (self.maxlen-2)

            prob_neg_slope = self.get_prob_neg_slope()
            if self.count_steps_without_decrease(prob_neg_slope) == self.patience:
                self.update_optimizer()
                self.reset_steps_without_decrease()

    def get_prob_neg_slope(self):
        if len(self.loss_queue) == self.maxlen:
            pdf = norm(loc=self.p[0], scale=np.sqrt(self.var*self.var_multiplier))
            return pdf.cdf(-self.epsilon)
        else:
            return 1

    def count_steps_without_decrease(self, prob_neg_slope):
        if prob_neg_slope < 0.51:
            self.steps_without_decrease += 1
        else:
            self.steps_without_decreas = 0
        return self.steps_without_decrease

    def reset_steps_without_decrease(self):
        self.steps_without_decrease = 0

    def update_optimizer(self):
        for param in self.optimizer.param_groups:
            new_lr = param['lr']*self.factor
            if new_lr > self.min_lr:
                param['lr'] = new_lr
            else:
                param['lr'] = self.min_lr
