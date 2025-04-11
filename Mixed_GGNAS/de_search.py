import os
import torch
import numpy as np
from Mixed_GGNAS.utils.dice_coefficient_loss import multiclass_dice_coeff
from utils.my_dataset import DriveDataset
from utils.data_transform import get_transform
from utils.losses import criterion
from utils.metrics import *
from utils.myde import MYDE
import utils.distributed_utils as utils_




def main(args):

    crop_size = 0
    # using compute_mean_std.py
    mean = (0.457, 0.221, 0.064)
    std = (0.321, 0.167, 0.086)
    # mean = (0.485, 0.341, 0.628)
    # std = (0.231, 0.237, 0.193)
    batch_size = args.batch_size
    init_c = args.init_c

    train_dataset = DriveDataset(args.data_path,
                                 train=True,
                                 transforms=get_transform(train=True, crop_size=crop_size, mean=mean, std=std))


    val_dataset = DriveDataset(args.data_path,
                               train=False,
                               transforms=get_transform(train=False, crop_size=crop_size, mean=mean, std=std))


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=0,
                                               shuffle=True,
                                               pin_memory=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             num_workers=0,
                                             pin_memory=True,
                                             collate_fn=val_dataset.collate_fn)

    loss_fn = criterion()
    metric_fn = utils_

    de = MYDE(pop_size=args.pop_size, mutation_factor=0.5, crossover_prob=0.5, seed=42,
              train_dataloader=train_loader,val_dataloader=val_loader,
              loss_fn=loss_fn,metric_fn=metric_fn,device=args.device,
              init_c=init_c,img_size=(crop_size,crop_size))
    de.run(42,args.resume, args.pop_size)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="", help="")
    # exclude background 
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--pop_size", default=10, type=int)
    parser.add_argument("--device", default="cuda:0", help="training device")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=400, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=0.0015, type=float, help='initial learning rate')
    parser.add_argument('--wd', '--weight-decay', default=5e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--init_c', default=3, type=int, help='init channels')
    parser.add_argument('--resume', default=True, help='resume from checkpoint')
    parser.add_argument('--count', default=1, type=int, help='')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()

    return args



if __name__ == "__main__":

    args = parse_args()
    main(args)