import random
import warnings
import argparse
import shutil
import scipy as sp
import os

import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

import clip
from utils import get_checkpoint_path, create_logger
from engine import *
from models import CustomCLIP, CrossModalAttn, Adapter
import itertools


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_args(args):
    strr = '\n'
    for k in vars(args):
        strr += '{}: {}\n'.format(k, getattr(args, k))
    return strr


def main(args):
    logger = create_logger(args.output, '123', file=f'{args.log}_{args.baseline}_shot{args.n_shot}_seed{args.seed}')
    args.logger = logger
    logger.info(print_args(args))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    cudnn.benchmark = True
    
    clip_model, _ = clip.load(args.arch, device)
    
    train_iter, val_loader, test_loaders, train_class_names, template, indiv_domain_loader = get_dataset(args) 

    # create model
    logger.info("=> using pre-trained model '{}'".format(args.arch))
    classifier = clip_model.visual
    classifier = classifier.to(device)
    clip.model.convert_weights(classifier)
    classifier.eval()
    decoder = CustomCLIP(args.cfg, train_class_names, clip_model).to(device)
    
    # obtain text features
    train_text_features = get_text_features(clip_model, template, train_class_names, device)
    pretrained_text_features = get_pretrained_text_features(decoder, args, device)
    pretrained_text_features['naive'] = train_text_features
    for test_loader in test_loaders:
        test_loader["text_features"] = get_text_features(clip_model, template, test_loader["class_names"], device)

    # define beta moving average 
    beta_dist = sp.stats.beta(args.beta, args.beta)
    total_iter = args.epochs * args.iters_per_epoch
    weight_func = lambda it: beta_dist.pdf((it + 0.5) / (total_iter + 1))

    bma_classifier = GeneralMovingAverage(classifier, weight_func)
    attn = CrossModalAttn(512, args.class_num, n_head=4).to(device)

    pretrained = {k: v.clone() for k, v in classifier.state_dict().items()}

    if args.phase == "train":
        # define optimizer and lr scheduler
        optimizer = AdamW(itertools.chain(classifier.parameters(), attn.parameters()), lr=args.lr, weight_decay=args.wd)
        lr_scheduler = CosineAnnealingLR(optimizer, args.epochs * args.iters_per_epoch)
        
        # define temperature for training
        if args.temperature is None:
            args.temperature = clip_model.logit_scale.exp().item()

        # evaluate zero-shot performance
        evaluate_all(classifier, val_loader, pretrained_text_features, test_loaders, args, logger, device, 'test', epoch='Start')

        # start training
        for epoch in range(args.epochs):
            logger.info("Learning rate: {:.4e}".format(lr_scheduler.get_last_lr()[0]))
            
            # train for one epoch
            train_resample(train_iter, classifier, bma_classifier, attn, pretrained_text_features, optimizer, lr_scheduler, args, device)

            if args.baseline != 'clipood':
                trained = {k: v.clone() for k, v in classifier.state_dict().items()}
                tmp_classifier = clip_model.visual.to(device)
                alpha_lis = [0.5]
                for alpha in alpha_lis:
                    args.alpha = alpha
                    tmp_classifier.load_state_dict(wise_ft(pretrained, trained, args.alpha))
                    logger.info(f'Epoch {epoch}, Alpha = {args.alpha}')
                    evaluate_all(tmp_classifier, val_loader, pretrained_text_features, test_loaders, args, logger, device, 'test', epoch, attn=attn)
            else:
                evaluate_all(bma_classifier, val_loader, pretrained_text_features, test_loaders, args, logger, device, 'test', epoch, attn=attn)

        logger.info("Training completed.")

    else:
        classifier.load_state_dict(torch.load(get_checkpoint_path(args, 'best')))
        logger.info("Evaluate best model:")
        evaluate_all(classifier, val_loader, pretrained_text_features, test_loaders, args, logger, device, 'test', 'End')

        bma_classifier.load_state_dict(torch.load(get_checkpoint_path(args, 'bma')))
        logger.info("Evaluate BMA model:")
        evaluate_all(bma_classifier, val_loader, pretrained_text_features, test_loaders, args, logger, device, 'test', 'End')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument('--root', metavar='DIR', default='DomainBed/domainbed/data/')
    parser.add_argument('-d', '--data', metavar='DATA', default='ImageNet')
    parser.add_argument('--task', default='phase2', type=str)
    parser.add_argument('--targets', nargs='+', type=int, default=[0], help='target domain(s) (DomainBed datasets only)')
    parser.add_argument('--n-shot', type=int, default=16)
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 36)')
    parser.add_argument('--lr', '--learning-rate', default=5e-6, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=0.1, type=float, metavar='W', help='weight decay (default: 0.1)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=400, type=int, help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training. ')
    parser.add_argument('--file', type=str, default=None, help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'], help="When phase is 'test', only test the model.")
    # parameters for CLIPood
    parser.add_argument('--temperature', type=float, default=None, help="Use CLIP's original temperature in default.")
    parser.add_argument('--lam', type=float, default=0.3)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--method', type=str, default='coop')
    parser.add_argument('--log', type=str, default='log')

    parser.add_argument('--baseline', type=str, default='clipood', choices=['clipood','ueo','erm'])
    parser.add_argument('--alpha', type=float, default=0.5)

    args = parser.parse_args()
    class_num_dict = {'OfficeHome': 65, 'DomainNet': 345, 'TerraIncognita': 10, 'PACS': 7, 'VLCS': 5, 'ImageNet':200, }
    all_domain_dict = {'OfficeHome': list(range(4)), 'DomainNet': list(range(6)), 'TerraIncognita': list(range(4)), 'PACS': list(range(4)), 'VLCS': list(range(4))}
    all_domain_dict['ImageNet'] = [1,4,5] + args.targets
    args.domain_names = [str(i) for i in all_domain_dict[args.data]]
    args.all_domains = all_domain_dict[args.data]
    args.class_num = class_num_dict[args.data] 
    if args.data == 'ImageNet' and args.targets[0] == 2:
        args.class_num = 1000
    torch.cuda.set_device(args.device)
    args.output = 'log/{}/{}/{}/{}'.format(args.data, args.arch, args.task, args.targets)
    os.makedirs(args.output, exist_ok=True)

    coop_cfg = {}
    coop_cfg['N_CTX'] = 16  # number of context vectors
    coop_cfg['CSC'] = False  # class-specific context
    coop_cfg['CTX_INIT'] = ""  # initialization  words
    coop_cfg['PREC'] = "fp16"  # fp16, fp32, amp
    coop_cfg['CLASS_TOKEN_POSITION'] = "end"  # 'middle' or 'end' or 'front'
    
    if args.method == 'coop':
        args.cfg = coop_cfg
    else:
        raise NotImplementedError()

    main(args)