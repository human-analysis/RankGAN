# main.py

import sys, traceback
import torch
import random
import torchvision
from model import Model
from dataloader import Dataloader
from checkpoints import Checkpoints
# from train_rank_preset import Trainer
from train_rank_preset_inverted import Trainer
import utils
import time
import datetime
import copy
import os
import config


def main():
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
    rankgan_model, criterion = models.setup(checkpoints)
    modelD = rankgan_model[0]
    modelG = rankgan_model[1]
    Encoder = rankgan_model[2]
    prevD, prevG = None, None

    if args.netD is not '':
        checkpointD = checkpoints.load(args.netD)
        modelD.load_state_dict(checkpointD)
    if args.netG is not '':
        checkpointG = checkpoints.load(args.netG)
        modelG.load_state_dict(checkpointG)
    if args.netE is not '':
        checkpointE = checkpoints.load(args.netE)
        Encoder.load_state_dict(checkpointE)
    if args.prevD is not '':
        prevD = copy.deepcopy(modelD)
        checkpointDprev = checkpoints.load(args.prevD)
        prevD.load_state_dict(checkpointDprev)
    if args.prevG is not '':
        prevG = copy.deepcopy(modelG)
        checkpointGprev = checkpoints.load(args.prevG)
        prevG.load_state_dict(checkpointGprev)

    # Data Loading
    dataloader = Dataloader(args)
    loader_train = dataloader.create(flag="Train")
    loader_test = dataloader.create(flag="Test")

    # The trainer handles the training loop and evaluation on validation set
    trainer = Trainer(args, modelD, modelG, Encoder, criterion, prevD, prevG)

    # start training !!!
    num_stages = args.num_stages
    stage_epochs = args.stage_epochs
    for stage in range(args.start_stage, num_stages):

        # check whether ready to start new stage and if not, optimize discriminator
        # if stage > 2:
        #     print("Optimizing Discriminator")
        #     trainer.setup_stage(stage, loader_test)
        #     opt_disc_flag = True
        #     epoch = 0
        #     while opt_disc_flag:
        #         opt_disc_flag = trainer.optimize_discriminator(stage-1, epoch, loader_train)
        #         epoch += 1

        # setup trainer for the stage
        trainer.setup_stage(stage, loader_test)
        print("Training for Stage {}".format(stage))

        for epoch in range(stage_epochs[stage]):
            # train for a single epoch
            # cur_time = time.time()
            # if stage == 2:

            loss_train = trainer.train(stage, epoch, loader_train)
            # if stage > 0:
            #     disc_acc = trainer.test(stage, epoch, loader_test)
            # print("Time taken = {}".format(time.time() - cur_time))

            try:
                torch.save(modelD.state_dict(), '%s/stage_%d_netD.pth' % (args.save, stage))
                torch.save(modelG.state_dict(), '%s/stage_%d_netG.pth' % (args.save, stage))
                torch.save(Encoder.state_dict(), '%s/stage_%d_netE.pth' % (args.save, stage))
            except Exception as e:
                print(e)

            # if stage == 1:
            #     break

if __name__ == "__main__":
    utils.setup_graceful_exit()
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        # do not print stack trace when ctrl-c is pressed
        pass
    except Exception:
        traceback.print_exc(file=sys.stdout)
    finally:
        utils.cleanup()
