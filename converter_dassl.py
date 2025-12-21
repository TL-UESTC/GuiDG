import CoOp.datasets.oxford_pets
import CoOp.datasets.oxford_flowers
import CoOp.datasets.fgvc_aircraft
import CoOp.datasets.dtd
import CoOp.datasets.eurosat
import CoOp.datasets.stanford_cars
import CoOp.datasets.food101
import CoOp.datasets.sun397
import CoOp.datasets.caltech101
import CoOp.datasets.ucf101
import CoOp.datasets.imagenet
import CoOp.datasets.imagenet_sketch
import CoOp.datasets.imagenetv2
import CoOp.datasets.imagenet_a
import CoOp.datasets.imagenet_r
import CoOp.datasets.officehome
import CoOp.datasets.domainnet
import CoOp.datasets.terra_incognita
import CoOp.datasets.pacs
import CoOp.datasets.imagenet_dg


from dassl.config import get_cfg_default
from dassl.utils.tools import read_image
from dassl.data.datasets import build_dataset

from torchvision import transforms
from torch.utils.data import Dataset

import os


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


class DasslDataset(Dataset):

    def __init__(self, dassl_dataset, split, transform=None, need_index=False):
        self.samples = []
        self.transform = transform
        self.need_index = need_index
        self._classnames = dassl_dataset.classnames
        datum_list = {
            "train": dassl_dataset.train_x,
            "val": dassl_dataset.val,
            "test": dassl_dataset.test,
        }[split]
        for data in datum_list:
            self.samples.append((data.impath, data.label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = read_image(path)
        if self.transform is not None:
            img = self.transform(img)
        if not self.need_index:
            return img, target
        else:
            return img, target, index
    
    @property
    def classnames(self):
        return self._classnames


def get_raw_dassl_dataset(dataset_name, root, n_shot, subsample="all"):
    cfg = get_cfg_default()
    cfg.DATASET.NAME = dataset_name
    cfg.DATASET.ROOT = root
    cfg.DATASET.NUM_SHOTS = n_shot
    cfg.DATASET.SUBSAMPLE_CLASSES = subsample
    cfg.SEED = 0
    dassl_dataset = build_dataset(cfg)
    return dassl_dataset


def get_fewshot_dassl_dataset(args, domain):
    cfg = get_cfg_default()
    domain_names = {
        'OfficeHome': ["Art", "Clipart", "Product", "Real_World"],
        'DomainNet': ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"],
        'PACS': ["art_painting", "cartoon", "photo", "sketch"],
        'TerraIncognita': ["location_100", "location_38", "location_43", "location_46"],
        'VLCS': ["Caltech101", "LabelMe", "SUN09", "VOC2007"],
        'ImageNet': ['a', 'i/train', 'i/val', 'r', 's', 'v2'],
    }
    cfg.DATASET.NAME = args.data + '_fewshot'
    cfg.DATASET.NUM_SHOTS = args.n_shot
    cfg.DATASET.SUBSAMPLE_CLASSES = args.task
    cfg.DATASET.SOURCE_DOMAINS = domain_names[args.data][domain]
    cfg.DATASET.TARGET_DOMAINS = domain_names[args.data][args.targets[0]]
    cfg.DATASET.ROOT = os.path.join(args.root, args.data, cfg.DATASET.SOURCE_DOMAINS)
    cfg.SEED = 0      
    dassl_dataset = build_dataset(cfg)
    return dassl_dataset


def get_dassl_datasets(dataset_name, root, n_shot=0):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    raw_base_dataset = get_raw_dassl_dataset(dataset_name, root, n_shot, "base")
    raw_open_dataset = get_raw_dassl_dataset(dataset_name, root, n_shot, "new")
    template = CUSTOM_TEMPLATES[dataset_name]
    train_dataset = DasslDataset(raw_base_dataset, "train", train_transform)
    val_dataset = DasslDataset(raw_base_dataset, "val", val_transform)
    test_dataset = DasslDataset(raw_base_dataset, "test", val_transform)
    open_dataset = DasslDataset(raw_open_dataset, "test", val_transform)
    base_class_names, open_class_names = train_dataset.classnames, open_dataset.classnames
    return train_dataset, val_dataset, test_dataset, open_dataset, base_class_names, open_class_names, template


def get_dassl_datasets_fewshot_dg(args):
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    if args.task == 'phase1':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        if args.data not in ['ImageNet']:
            raw_dataset = get_fewshot_dassl_dataset(args, args.targets[0])
        else:
            raw_dataset = get_fewshot_dassl_dataset(args, args.source)
        #template = CUSTOM_TEMPLATES[dataset_name]
        train_dataset = DasslDataset(raw_dataset, "train", train_transform)
        val_dataset = DasslDataset(raw_dataset, "val", val_transform)
        test_dataset = DasslDataset(raw_dataset, "test", val_transform)
        base_class_names = train_dataset.classnames
    elif args.task == 'phase2':
        train_transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), 
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
        train_dataset = []
        for i in args.all_domains:
            if i not in args.targets:
                dassl = get_fewshot_dassl_dataset(args, i)
                dataset = DasslDataset(dassl, "val", train_transform)       # use 'val' split to train phase 2
                train_dataset.append(dataset)
        test_dataset = [DasslDataset(get_fewshot_dassl_dataset(args, args.targets[0]), "test", val_transform)]    # use 'test' split for testing
        base_class_names = test_dataset[0].classnames
        val_dataset = None

    return train_dataset, val_dataset, test_dataset, base_class_names


def _convert_image_to_rgb(image):
    return image.convert("RGB")