import os
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from .oxford_pets import OxfordPets
from .imagenet import ImageNet
from dassl.utils import mkdir_if_missing, listdir_nohidden
import pickle
import random
import shutil
TO_BE_IGNORED = ['README.txt']



@DATASET_REGISTRY.register()
class ImageNet_fewshot(DatasetBase):
 
    domain_idx = {'a':0, 'i/train':1, 'i/val':1, 'r':2, 's':3, 'v2':4}

    def __init__(self, cfg):
        root = os.path.join('splits', cfg.DATASET.NAME.split('_')[0])
        self.dataset_dir = root
        self.split_path = os.path.join(self.dataset_dir, f"{cfg.DATASET.SOURCE_DOMAINS.replace('/','_')}_split_target_{cfg.DATASET.TARGET_DOMAINS.replace('/','_')}.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        self.image_dir = cfg.DATASET.ROOT
        if not os.path.exists(self.image_dir):
            self.split_imagenet(cfg, 4)
        self.domain = cfg.DATASET.SOURCE_DOMAINS
        if self.domain_idx[cfg.DATASET.SOURCE_DOMAINS] == 1:
            par = os.path.dirname(os.path.dirname(self.image_dir))
        else:
            par = os.path.dirname(self.image_dir)
        classname_file = os.path.join(par, cfg.DATASET.TARGET_DOMAINS, 'classnames.txt')
        mkdir_if_missing(self.split_fewshot_dir)
        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, '')
        elif cfg.DATASET.SUBSAMPLE_CLASSES == 'domain_shift_split':
            raise RuntimeError('Split not exist')
        else:
            classnames = ImageNet.read_classnames(classname_file)
            self.source_data = self.read_data(classnames)
            train, val = OxfordPets.split_trainval(self.source_data, p_val=0.5)
            test = self.source_data
            OxfordPets.save_split(train, val, test, self.split_path, self.dataset_dir)       

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            tmp = cfg.DATASET.SOURCE_DOMAINS.replace('/','_')
            preprocessed = os.path.join(self.split_fewshot_dir, f"{tmp}_shot_{num_shots}-seed_{seed}_target_{cfg.DATASET.TARGET_DOMAINS.replace('/','_')}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=num_shots)
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        super().__init__(train_x=train, val=val, test=test)

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        folders = [f for f in folders if '.' not in f]
        items, cnt = [], 0

        for label, folder in enumerate(folders):
            if folder not in classnames:
                continue
            classname = classnames[folder] 
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(impath=impath, label=cnt, classname=classname)
                items.append(item)
            cnt += 1

        return items

    def split_imagenet(self, cfg, n=4):
        root_pth = os.path.dirname(cfg.DATASET.ROOT)        #   ..../ImageNet
        source_pth = os.path.join(root_pth, 'i/train')
        cls_lis = os.listdir(source_pth)
        for cls_name in cls_lis:
            all_pth = os.path.join(source_pth, cls_name)      # ..../ImageNet/i/train/nxxxx
            if os.path.isdir(all_pth):
                piclis = os.listdir(all_pth)
                random.shuffle(piclis)
                n_per_domain = int(len(piclis) / n)
                start, end = 0, n_per_domain
                for i in range(len(n)):
                    os.makedirs(os.path.join(root_pth, str(i), cls_name), exist_ok=True)
                    print('Copying from {} to {}'.format(all_pth, os.path.join(root_pth, str(i), cls_name)))
                    for j in range(start, end):
                        picname = piclis[j]
                        src = os.path.join(all_pth, picname)
                        tar = os.path.join(root_pth, str(i), cls_name, picname)
                        shutil.copy(src, tar)
                    start = end
                    end = min(start+n_per_domain, len(piclis))
