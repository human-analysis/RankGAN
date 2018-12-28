# checkpoints.py

import os
import torch

class Checkpoints:

    def __init__(self,args):
        self.dir_save = args.save
        self.dir_load = args.resume
        # self.prevmodel = args.prevmodel
        self.prevmodel = None

        if os.path.isdir(self.dir_save) == False:
            os.makedirs(self.dir_save)

    def latest(self, name):
        output = {}
        if self.dir_load == None:
            output['resume'] = None
        else:
            output['resume'] = self.dir_load

        if (self.prevmodel != None):
            output['prevmodel'] = self.prevmodel
        else:
            output['prevmodel'] = None

        return output[name]

    def save(self, epoch, model, best):
        if best == True:
            output = {}
            num = len(model)
            for key, value in model[0].items():
                output[key] = value.state_dict()
            torch.save(output, '%s/model_%d_epoch_%d.pth' %
                       (self.dir_save, num, epoch))

    def load(self, filename):
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            model = torch.load(filename)
        else:
            print("=> no checkpoint found at '{}'".format(filename))

        return model
