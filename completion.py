#completion.py

import torch
import random
import torchvision
from model import Model
from dataloader import Dataloader
from checkpoints import Checkpoints
from evaluation import Evaluate
from generation import Generator
import os
import datetime
import utils
import copy
import config

# parse the arguments
args = config.parse_args()
random.seed(args.manual_seed)
torch.manual_seed(args.manual_seed)
args.save = os.path.join(args.result_path, 'save')
args.logs = os.path.join(args.result_path, 'logs')
utils.saveargs(args)

# initialize the checkpoint class
checkpoints = Checkpoints(args)

# Create Model
models = Model(args)
gogan_model, criterion = models.setup(checkpoints)
netD = gogan_model[0]
netG = gogan_model[1]
netE = gogan_model[2]

if args.netD is not '':
    checkpointD = checkpoints.load(args.netD)
    netD.load_state_dict(checkpointD)
if args.netG is not '':
    checkpointG = checkpoints.load(args.netG)
    netG.load_state_dict(checkpointG)
if args.netE is not '':
    checkpointE = checkpoints.load(args.netE)
    netE.load_state_dict(checkpointE)

# Data Loading
dataloader = Dataloader(args)
test_loader = dataloader.create("Test", shuffle=False)

# The trainer handles the training loop and evaluation on validation set
# evaluate = Evaluate(args, netD, netG, netE)
generator = Generator(args, netD, netG, netE)

# test for a single epoch
# test_loss = evaluate.complete(test_loader)
# loss = generator.generate_one(test_loader)
loss = generator.interpolate(test_loader)
