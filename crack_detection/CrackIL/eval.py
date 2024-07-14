import os
import random
import time
import numpy as np
import torch
import math
import re
from metrics import Evaluator
from argparse import ArgumentParser

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

from data.dataset import CrackSourceTesting, CrackTarget
from model.erfnet_RA_parallel import Net as Net_RAP

from shutil import copyfile
from tensorboardX import SummaryWriter
import cv2
from metrics import Evaluator

def eval(model, args, loader, task):

    model.eval()
    evaluator = Evaluator(2)
    with torch.no_grad():
        for step, sample in enumerate(loader):
            images, name, targets = sample['image'], sample['image_id'], sample['label']
            image_size = sample['image_size']
            inputs = images.cuda()

            _, outputs = model(inputs, task)
            pred = outputs.data.max(1)[1].cpu().numpy() 
            evaluator.add_batch(targets.cpu().numpy(), pred)

            pred_image = np.transpose(pred, (1, 2, 0))
            pred_image = np.asarray(pred_image*255, dtype=np.uint8)
            # print(pred_image.shape)
            # print(name[0])
            pred_image = cv2.resize(pred_image, (448, 448))
            # cv2.imwrite(os.path.join(args.result_dir, name[0]+'.jpg'), pred_image)
        acc = evaluator.Pixel_Accuracy()
        acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        fwavacc = evaluator.Frequency_Weighted_Intersection_over_Union()

        print("Validateion: mIoU: {:.3f}, Acc: {:.3f}, Acc_class: {:.3f}, Fwavacc: {:.3f}".format(mIoU, acc, acc_class, fwavacc))



def main(args):

    # train_dataset_source = CrackSourceTesting(args = args, base_dir = args.source_dataset_path, split ='train')
    val_dataset_source = CrackSourceTesting(args = args, base_dir = args.source_dataset_path, split ='val')

    target_dataset = CrackTarget(args = args, base_dir = args.target_dataset_path)

    # train_loader_old = DataLoader( train_dataset_source, batch_size=args.batch_size_val, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader_old = DataLoader(val_dataset_source, batch_size=args.batch_size_val, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    target_loader = DataLoader(target_dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    # model = Net_RAP(args.num_classes, args.nb_tasks, args.current_task)
    model_old = Net_RAP(args.num_classes, args.nb_tasks-1, args.current_task-1)

    # model = torch.nn.DataParallel(model).cuda()
    model_old = torch.nn.DataParallel(model_old).cuda()
    
    if os.path.isfile(args.weight):
        print("=> loading checkpoint '{}'".format(args.weight))
        checkpoint = torch.load(args.weight)
        model_old.load_state_dict(checkpoint['state_dict'])
        # model_old.load_state_dict(checkpoint['state_dict_old'])

        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.weight, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.weight))

    # eval(model, model_old, args, val_loader_old, 0)

    # eval(model, model_old, args, train_loader_old, 0)
    os.makedirs(args.result_dir, exist_ok=True)
    # eval(model, args, train_loader_old, 0)
    eval(model_old, args, val_loader_old, 0)
    eval(model_old, args, target_loader, 0)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--batch_size_val', type=int, default=1, help='input val batch size')
    parser.add_argument('--source_dataset_path', type=str, default='/scratch/kushagra0301/CrackDataset', help='source dataset')
    parser.add_argument('--target_dataset_path', type=str, default='/scratch/kushagra0301/CustomCrackDetectionModified', help='target dataset')
    parser.add_argument('--saved_model', type=str, default='/scratch/kushagra0301/Crack_IL_step_1/best_model.pth', help='saved model')
    parser.add_argument('--num_classes', type=int, default=[2,2], help='number of classes')
    parser.add_argument('--crop_size_height', type=int, default=512, help='crop size height')
    parser.add_argument('--crop_size_width', type=int, default=1024, help='crop size width')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--nb_tasks', type=int, default=2) 
    parser.add_argument('--current_task', type=int, default=1)
    parser.add_argument('--result_dir', type=str, default='/scratch/kushagra/crack_il_results', help='save directory')
    parser.add_argument('--weight', type=str, default='/scratch/kushagra/weights', help='weight path directory')
    parser.add_argument('--dataset_avoided', type=str)

    args = parser.parse_args()

    main(args)