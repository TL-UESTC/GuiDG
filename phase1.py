import random
import argparse
import torch.nn as nn
import shutil
import scipy as sp
import os

import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW, SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import clip
from utils import get_checkpoint_path, create_logger, accuracy, AverageMeter
from engine import get_dataset
from models import CustomCLIP
from torch.cuda.amp import autocast
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

 
def validate(decoder, val_loader, logger, device, epoch):
    top1_t = AverageMeter('Acc@1', ':6.2f')
    decoder.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            with autocast():
                images = images.to(device)
                target = target.to(device) 
                y, _, _ = decoder(images)
                acc_t, = accuracy(y, target, topk=(1,))
                top1_t.update(acc_t.item(), images.size(0))
    logger.info('Eval @ Epoch{} = {:.3f}'.format(epoch, top1_t.avg))
    return top1_t.avg


def train(train_loader, model, optimizer, lr_scheduler, args, device):
    model.train()
    for data in tqdm(train_loader):
        x, labels = data
        x, labels = x.to(device), labels.to(device)
        with autocast():
            y, _, _ = model(x)
            loss = F.cross_entropy(y, labels) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()


def print_args(args):
    strr = '\n'
    for k in vars(args):
        strr += '{}: {}\n'.format(k, getattr(args, k))
    return strr


def main(args):
    logger = create_logger(args.output, '123', file=f"{args.log}_shot{args.n_shot}_seed{args.seed}")
    args.logger = logger
    logger.info(print_args(args))
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    cudnn.benchmark = True
    
    clip_model, _ = clip.load(args.arch, device)
    
    train_loader, val_loader, test_loaders, train_class_names, template, indiv_domain_loader = get_dataset(args) 

    #  create model
    logger.info("=> using pre-trained model '{}'".format(args.arch))

    decoder = CustomCLIP(args.cfg, train_class_names, clip_model).cuda()

    if args.phase == "train":
        # define optimizer and lr scheduler
        optimizer = SGD(decoder.prompt_learner.parameters(), lr=args.lr)
        lr_scheduler = CosineAnnealingLR(optimizer, args.epochs * args.iters_per_epoch)
        
        # define temperature for training
        if args.temperature is None:
            args.temperature = clip_model.logit_scale.exp().item()

        best_val_acc1 = 0

        # start training
        for epoch in range(args.epochs):
            logger.info("Learning rate: {:.4e}".format(lr_scheduler.get_last_lr()[0]))
            
            # train for one epoch
            train(train_loader, decoder, optimizer, lr_scheduler, args, device)

            # evaluate all
            if epoch % 5 == 0:
                val_acc1 = validate(decoder, val_loader, logger, device, epoch=epoch)

                # remember best acc@1 and save checkpoint
                torch.save(decoder.prompt_learner.state_dict(), get_checkpoint_path(args, 'latest'))
                if val_acc1 > best_val_acc1:
                    logger.info('[Saving best]')
                    best_pth = 'best{}_{}'.format(args.n_shot, args.targets[0]) if args.data == 'ImageNet' else f'best{args.n_shot}'
                    shutil.copy(get_checkpoint_path(args, 'latest'), get_checkpoint_path(args, best_pth))
                    best_val_acc1 = val_acc1

        logger.info("Training completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument('--root', metavar='DIR', default='DomainBed/domainbed/data/',  help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='ImageNet')
    parser.add_argument('--task', default='phase1', type=str)
    parser.add_argument('--targets', nargs='+', type=int, default=[2], help='target domain(s) (DomainBed datasets only)')
    parser.add_argument('--n-shot', type=int, default=16)
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N', help='mini-batch size (default: 36)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=0.1, type=float, metavar='W', help='weight decay (default: 0.1)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=31, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=400, type=int, help='Number of iterations per epoch')
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
    parser.add_argument('--file', type=str, default=None, help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'], help="When phase is 'test', only test the model.")
    # parameters for CLIPood
    parser.add_argument('--temperature', type=float, default=None, help="Use CLIP's original temperature in default.")
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--method', type=str, default='coop')
    parser.add_argument('--log', type=str, default='log')
    parser.add_argument('--source', type=int, default=1, help='for imagenet-DG only')

    args = parser.parse_args()
    class_num_dict = {'OfficeHome': 65, 'DomainNet': 345, 'TerraIncognita': 10, 'PACS': 7, 'VLCS': 5, 'ImageNet':200}
    all_domain_dict = {'OfficeHome': list(range(4)), 'DomainNet': list(range(6)), 'TerraIncognita': list(range(4)), 'PACS': list(range(4)), 'VLCS': list(range(4))}
    all_domain_dict['ImageNet'] = [1,4,5] + args.targets
    args.class_num = class_num_dict[args.data]
    args.all_domains = all_domain_dict[args.data]
    torch.cuda.set_device(args.device)
    if args.data not in ['ImageNet']:
        args.output = 'log/{}/{}/{}/{}'.format(args.data, args.arch, args.task, args.targets)
    else:
        args.output = 'log/{}/{}/{}/[{}]'.format(args.data, args.arch, args.task, args.source)
    os.makedirs(args.output, exist_ok=True)

    coop_cfg = {}
    coop_cfg['N_CTX'] = 16  # number of context vectors
    coop_cfg['CSC'] = False  # class-specific context
    coop_cfg['CTX_INIT'] = ""  # initialization words
    coop_cfg['PREC'] = "fp16"  # fp16, fp32, amp
    coop_cfg['CLASS_TOKEN_POSITION'] = "end"  # 'middle' or 'end' or 'front'
    
    if args.method == 'coop':   # can add more Prompt-Tuning methods
        args.cfg = coop_cfg
    else:
        raise NotImplementedError()

    main(args)
