import os
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from .oxford_pets import OxfordPets
from dassl.utils import mkdir_if_missing
import pickle



@DATASET_REGISTRY.register()
class OfficeHome_fewshot(DatasetBase):

    dataset_dir = "office_home"
    domain_idx = {'Art':0, 'Clipart':1, 'Product':2, 'Real_World':3}

    def __init__(self, cfg):
        root = os.path.join('splits', cfg.DATASET.NAME.split('_')[0])
        self.dataset_dir = root
        self.dir = os.path.join(self.dataset_dir, cfg.DATASET.SOURCE_DOMAINS+".txt")
        self.split_path = os.path.join(self.dataset_dir, f"{cfg.DATASET.SOURCE_DOMAINS}_split_xy.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)
        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, '')
        elif cfg.DATASET.SUBSAMPLE_CLASSES == 'domain_shift_split':
            raise RuntimeError('Split not exist')
        else:
            self.src_txt = open(self.dir).readlines()
            self.source_domain = self.domain_idx[cfg.DATASET.SOURCE_DOMAINS]
            self.source_data = self.collect(self.src_txt, self.source_domain)
            train, else_ = OxfordPets.split_trainval(self.source_data, p_val=0.6)     # 40% train, 60% for val + test
            val, test = OxfordPets.split_trainval(else_, p_val=0.333)     # 60% -> 40% for val, 20% for test
            OxfordPets.save_split(train, val, test, self.split_path, self.dataset_dir)       

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"{cfg.DATASET.SOURCE_DOMAINS}_shot_{num_shots}-seed_{seed}.pkl")
            
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
        # train: pre50%-16shot; val: post50%-16shot; test: post50%-All
        super().__init__(train_x=train, val=val, test=test)

    def collect(self, image_list, domain_label):
        out = []
        for val in image_list:
            pth, label = val.split()[0], int(val.split()[1])
            cls_name = pth.split('/')[-2]
            out.append(Datum(pth, label, domain_label, cls_name))
        return out




