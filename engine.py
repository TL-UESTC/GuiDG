import time
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

import clip

import converter_dassl, converter_domainbed
from utils import accuracy, AverageMeter, ProgressMeter, ForeverDataIterator
from tqdm import tqdm
import pickle
from models import ent_loss
import os
from pathlib import Path
from torch.cuda.amp import GradScaler, autocast



class GeneralMovingAverage(object):
    def __init__(self, model, weight_func):
        self.model = model
        self.weight_func = weight_func
        self.iter = 0
        self.weight = weight_func(self.iter)
        self.weight_sum = self.weight
        self.moving_avg = copy.deepcopy(model)
        for param in self.moving_avg.parameters():
            param.requires_grad = False

    def update(self):
        self.iter += 1
        self.weight = self.weight_func(self.iter)
        relative_weight = self.weight / self.weight_sum
        for moving_avg_param, param in zip(self.moving_avg.parameters(), self.model.parameters()):
            moving_avg_param.data = (moving_avg_param + relative_weight * param) / (1 + relative_weight)
        self.weight_sum += self.weight

    def __call__(self, x: torch.Tensor):
        return self.moving_avg(x)

    def train(self, mode=True):
        self.moving_avg.train(mode)

    def eval(self):
        self.train(False)

    def state_dict(self):
        return self.moving_avg.state_dict()

    def load_state_dict(self, state_dict):
        self.moving_avg.load_state_dict(state_dict)

    @property
    def module(self):
        return self.moving_avg.module


def get_dataset(args):
    indiv_domain_loader = None
    if args.task == 'phase1':
        if args.n_shot == 0:
            assert len(args.targets) == 1
            all_domains = args.all_domains
            input_targets = [i for i in all_domains if i not in args.targets]  
            train_datasets, val_datasets, test_datasets, class_names = converter_domainbed.get_domainbed_datasets(dataset_name=args.data, root=args.root, targets=input_targets, holdout=0.7)   
            save_root = os.path.join('splits', args.data)
            os.makedirs(save_root, exist_ok=True)
            with open(os.path.join(save_root, '{}_phase1.pkl'.format(args.targets[0])), 'wb+') as f:
                pickle.dump(train_datasets[0].keys, f)
            with open(os.path.join(save_root, '{}_phase2.pkl'.format(args.targets[0])), 'wb+') as f:
                pickle.dump(val_datasets[0].keys, f)
            train_class_names = class_names
            train_iter = DataLoader(ConcatDataset(train_datasets), batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
            val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
            indiv_domain_loader = {}
            test_loaders = [
                {
                    "name": "test",
                    "loader": DataLoader(ConcatDataset(test_datasets), batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                    "class_names": class_names
                }
            ]
        else:   #  few-shot 
            train_dataset, val_dataset, test_dataset, class_names = converter_dassl.get_dassl_datasets_fewshot_dg(args)
            train_class_names = class_names
            train_iter = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
            test_loaders = [
                {
                    "name": "test",
                    "loader": DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4),
                    "class_names": class_names
                },
            ]

        template = "a photo of a {}."

    elif args.task == 'phase2':  # phase 2
        if args.n_shot == 0:
            holdout_dict = {}
            all_domains = args.all_domains
            input_targets = [i for i in all_domains if i not in args.targets]
            for d in input_targets:
                pth = os.path.join('splits', args.data, '{}_phase2.pkl'.format(d))
                with open(pth, 'rb') as f:
                    holdout_dict[d] = pickle.load(f)
            train_datasets, test_datasets, class_names = converter_domainbed.get_domainbed_datasets(dataset_name=args.data, root=args.root, targets=args.targets, holdout=holdout_dict)
            train_class_names = class_names
            train_iter = converter_domainbed.get_forever_iter(train_datasets, args.batch_size, num_workers=args.workers)
            test_loaders = [
                {
                    "name": f"{i}",
                    "loader": DataLoader(d, batch_size=args.batch_size, shuffle=False, num_workers=args.workers),
                    "class_names": class_names
                }
                for i,d in enumerate(test_datasets)
            ]
        else:
            train_dataset, val_dataset, test_dataset, class_names = converter_dassl.get_dassl_datasets_fewshot_dg(args)
            train_class_names = class_names
            train_iter = converter_domainbed.get_forever_iter(train_dataset, args.batch_size, num_workers=args.workers)   # use 'val' split for phase2
            test_loaders = [{
                    "name": args.domain_names[i],
                    "loader": DataLoader(test_dataset[i], batch_size=args.batch_size*3, shuffle=False, num_workers=args.workers),
                    "class_names": class_names[i]
                } for i in range(len(test_dataset))
            ]
        val_loader = indiv_domain_loader = None
        template = "a photo of a {}."

    
    return train_iter, val_loader, test_loaders, train_class_names, template, indiv_domain_loader


def get_text_features(clip_model, template, class_names, device):
    with torch.no_grad():
        texts = torch.cat([clip.tokenize(template.format(c.replace("_", " "))) for c in class_names]).to(device)
        text_features = clip_model.encode_text(texts)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


def get_pretrained_text_features(decoder, args, device):
    pretrained_text_features = {}
    for d in args.all_domains:
        if d not in args.targets:
            with torch.no_grad():
                d_str = '[{}]'.format(d)
                if args.n_shot == 0:
                    pth_root = os.path.join('log', args.data, args.arch, 'phase1', d_str, 'best.pth') 
                else:
                    if args.data != 'ImageNet':
                        pth_root = os.path.join('log', args.data, args.arch, 'phase1', d_str, f'best{args.n_shot}.pth')
                    else:
                        pth_root = os.path.join('log', args.data, args.arch, 'phase1', d_str, f'best{args.n_shot}_{args.targets[0]}.pth')

                weight = torch.load(pth_root, map_location=device)
                decoder.prompt_learner.load_state_dict(weight)
                prompts = decoder.prompt_learner()
                tokenized_prompts = decoder.tokenized_prompts
                text_features = decoder.text_encoder(prompts, tokenized_prompts)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                pretrained_text_features[d] = text_features

    return pretrained_text_features


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


def normalize(x):
    return x - x.mean(1, keepdim=True)


def train_resample(train_iter, model, moving_avg_model, attn, text_features, optimizer, lr_scheduler, args, device):
    model.eval()
    attn.train()
    use_domains = [i for i in args.all_domains if i not in args.targets] 
    for i in tqdm(range(args.iters_per_epoch)):
        x, labels, domain_idx, cnt = [], [], [], 0
        for x_d, labels_d in next(train_iter):
            x.append(x_d)
            labels.append(labels_d)
            domain_idx += [use_domains[cnt]]*x_d.shape[0]
            cnt += 1
        domain_idx = torch.tensor(domain_idx)
        x, labels = torch.cat(x), torch.cat(labels)
        assert len(domain_idx) == labels.shape[0], 'Domain idx ({}) != labels ({})'.format(len(domain_idx), labels.shape[0])

        x, labels = x.to(device), labels.to(device)

        with autocast():
            f = model(x)
            f = f / f.norm(dim=-1, keepdim=True)

            loss = torch.tensor(0.0).cuda()
            text_feas = []
            # domain_specific
            for d in use_domains:
                text_feature = text_features[d]
                text_feas.append(text_feature)

            weights = attn(f, text_feas)        
            weights = weights.mean(0)     
            ys = []
            
            for i in range(len(use_domains)):
                f_cur = f.clone()
                if args.baseline == 'clipood':
                    f_cur -= args.lam * text_feas[i][labels]
                y = f_cur @ text_feas[i].T
                y = args.temperature * y
                ys.append(y)
                loss += F.cross_entropy(y, labels) * weights[i]
                if args.baseline == 'ueo':
                    loss += 0.01 * ent_loss(y)

            optimizer.zero_grad()
            loss.backward()
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
            lr_scheduler.step()

            moving_avg_model.update()


def validate_resample(val_loader, model, text_features, args, device, attn) -> float:
    if attn is not None:
        attn.eval()
    shift = 0
    use_domains = [i for i in args.all_domains if i not in args.targets] 
    top1_dict = {i:AverageMeter('Acc@1', ':6.2f') for i in use_domains}
    top1_dict['Ensemble_result'] = AverageMeter('Acc@1', ':6.2f')
    pred_dict = {i:None for i in use_domains}
    all_label, all_weight, cnt = None, None, 0

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            with autocast():
                images = images.to(device)
                target = target.to(device) - shift

                image_features = model(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                all_label = target.cpu() if all_label is None else torch.concat([all_label, target.cpu()])

                ys, ts = [], []
                for d in use_domains:
                    text_feature = text_features[d]
                    output_similarity = image_features @ text_feature.T
                    acc_t, = accuracy(output_similarity, target, topk=(1,))
                    top1_dict[d].update(acc_t.item(), images.size(0))
                    pred_dict[d] = output_similarity.cpu() if pred_dict[d] is None else torch.concat([pred_dict[d], output_similarity.cpu()])

                    y = args.temperature * output_similarity
                    ys.append(y)
                    ts.append(text_feature)
                if attn is not None:
                    ts = torch.stack(ts, 0)     # NUM, C, d
                    weights = attn(image_features, ts).mean(0)  # NUM
                    B = image_features.shape[0]

                    all_weight = all_weight + weights if all_weight is not None else weights

                    ys = torch.stack(ys, 1)     # B, NUM, C
                    y = torch.einsum('BNC,BN -> BC', ys, weights.repeat(B,1))
                    acc, = accuracy(y, target, topk=(1,))
                    top1_dict['Ensemble_result'].update(acc.item(), images.size(0))
                    cnt += 1

    for k in top1_dict:
        top1_dict[k] = top1_dict[k].avg

    if attn is not None:
        args.logger.info(f'Weight: {all_weight / cnt}')

    return top1_dict


def evaluate_all(model, val_loader, text_features, test_loaders, args, logger, device, func='val', epoch=0, attn=None):
    if func == 'val':
        val_acc = validate_resample(val_loader, model, text_features, args, device, attn)
        out_str = 'VAL @Epoch {}: '.format(epoch) 
        for i in val_acc:
            out_str += '{}={:.3f}  '.format(i,val_acc[i]) 
        logger.info(out_str)
        return val_acc['Ensemble_result']
    elif func == 'test':
        for test_loader in test_loaders:
            test_acc = validate_resample(test_loader["loader"], model, text_features, args, device, attn)
            out_str = 'TEST on [{}] @Epoch {}: '.format(test_loader['name'], epoch)
            for i in test_acc:
                out_str += '{}={:.3f}  '.format(i,test_acc[i]) 
            logger.info(out_str)
    else:
        raise


def wise_ft(theta_0, theta_1, alpha):
    assert set(theta_0.keys()) == set(theta_1.keys())
    mixed = {
        key: (1 - alpha) * theta_0[key] + alpha * theta_1[key] for key in theta_0.keys()
    }
    return mixed


