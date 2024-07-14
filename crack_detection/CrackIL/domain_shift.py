import os
import random
import time
import numpy as np
import torch
import math
import re
import torch
from metrics import Evaluator
from argparse import ArgumentParser

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data.dataset import CrackSource, CrackTarget
from model.erfnet_RA_parallel import Net as Net_RAP
from model.discriminator import Discriminator

from shutil import copyfile
from tensorboardX import SummaryWriter

def domain_shift(model1, model2, loader, task1, task2):

    model1.eval()
    model2.eval()
    shift = 0
    with torch.no_grad():
        for step, sample in enumerate(loader):
            # if step%10 == 0:
            #     print(step, '/', len(loader))
            images, name = sample['image'], sample['image_id']
            inputs = images.cuda()

            x1, _= model1(inputs, task1)
            x2, _ = model2(inputs, task2)
            shift+= torch.nn.MSELoss()(x1, x2)
    
    return shift


def main(args):

  

    target_dataset = CrackTarget(args = args, base_dir = args.target_dataset_path)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    model1 = Net_RAP(args.num_classes, args.nb_tasks-1, args.current_task-1)
    model2 = Net_RAP(args.num_classes, args.nb_tasks, args.current_task)

    model1 = torch.nn.DataParallel(model1).cuda()
    model2 = torch.nn.DataParallel(model2).cuda()
    
    if os.path.isfile(args.weight1):
        print("=> loading checkpoint 1 '{}'".format(args.weight1))
        checkpoint1 = torch.load(args.weight1)
        model1.load_state_dict(checkpoint1['model'])
        # model_old.load_state_dict(checkpoint['state_dict_old'])

        print("=> loaded checkpoint 1 '{}' (epoch {})"
                .format(args.weight1, checkpoint1['epoch']))
    else:
        print("=> no checkpoint 1 found at '{}'".format(args.weight1))

    if os.path.isfile(args.weight2):
        print("=> loading checkpoint 2 '{}'".format(args.weight2))
        checkpoint2 = torch.load(args.weight2)
        model2.load_state_dict(checkpoint2['state_dict'])
        # model_old.load_state_dict(checkpoint['state_dict_old'])

        print("=> loaded checkpoint 2 '{}' (epoch {})"
                .format(args.weight2, checkpoint2['epoch']))
    else:
        print("=> no checkpoint 2 found at '{}'".format(args.weight2))

    # eval(model, model_old, args, val_loader_old, 0)

    # eval(model, model_old, args, train_loader_old, 0)
   
    shift = domain_shift(model1, model2, target_loader, 0, 1)
    print(shift/len(target_dataset))
    # eval(model_old, args, target_loader, 0)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--batch_size_val', type=int, default=1, help='input val batch size')
    parser.add_argument('--source_dataset_path', type=str, default='/scratch/kushagra0301/CrackDataset', help='source dataset')
    parser.add_argument('--target_dataset_path', type=str, default='/scratch/kushagra0301/CustomCrackDetection', help='target dataset')
    parser.add_argument('--saved_model', type=str, default='/scratch/kushagra0301/Crack_IL_step_1/best_model.pth', help='saved model')
    parser.add_argument('--num_classes', type=int, default=[2,2], help='number of classes')
    parser.add_argument('--crop_size_height', type=int, default=512, help='crop size height')
    parser.add_argument('--crop_size_width', type=int, default=1024, help='crop size width')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--nb_tasks', type=int, default=2) 
    parser.add_argument('--current_task', type=int, default=1)
    parser.add_argument('--result_dir', type=str, default='/scratch/kushagra/crack_il_results', help='save directory')
    parser.add_argument('--weight1', type=str, default='/scratch/kushagra/weights', help='weight path directory')
    parser.add_argument('--weight2', type=str, default='/scratch/kushagra/weights', help='weight path directory')

    args = parser.parse_args()

    main(args)