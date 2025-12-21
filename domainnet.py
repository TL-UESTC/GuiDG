import os
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from .oxford_pets import OxfordPets
from dassl.utils import mkdir_if_missing
import pickle


@DATASET_REGISTRY.register()
class DomainNet_fewshot(DatasetBase):


    dataset_dir = "domainnet"
    domain_idx = {'clipart':0, 'infograph':1, 'painting':2, 'real':3, 'quickdraw':4, 'sketch': 5}

    def __init__(self, cfg):
        root = os.path.join('splits', cfg.DATASET.NAME.split('_')[0])
        self.dataset_dir = root
        self.dir = os.path.join(self.dataset_dir, cfg.DATASET.SOURCE_DOMAINS+".txt")
        if not os.path.exists(self.dir):
            self.generate_image_list(cfg.DATASET.ROOT, self.dir)
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
            else_, test = OxfordPets.split_trainval(self.source_data, p_val=0.2)     # 80% phase1+2, 20% test
            train, val = OxfordPets.split_trainval(else_, p_val=0.5)     # 50-50 split for phase1+2
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

    @staticmethod
    def generate_image_list(root_dir, output_file):
        categories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        categories.sort()
        category_to_index = {category: index for index, category in enumerate(categories)}
        
        with open(output_file, 'w+') as f:
            for category in categories:
                category_path = os.path.join(root_dir, category)
                
                for filename in os.listdir(category_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                        image_path = os.path.join(category, filename)
                        category_index = category_to_index[category]
                        all_path = os.path.join(root_dir, image_path)
                        f.write(f"{all_path} {category_index}\n")
        
        print(f"Generated {output_file} with {len(categories)} classes.")

