# Pytorch implementation for GuiDG 
PyTorch implementation of [Generalizing Vision-Language Models with Dedicated Prompt Guidance (AAAI'26)](https://arxiv.org/abs/2512.02421).
The following guidance runs GuiDG on ImageNet-DG. The code supports other datasets reported in paper with similar usage.

## Environment
- Python==3.12, Pytorch==2.4.1
- Clone [CoOp](https://github.com/KaiyangZhou/CoOp) and prepare environments as instructed (including Dassl, CLIP, etc.).
- Clone [DomainBed](https://github.com/facebookresearch/DomainBed) and prepare environments as instructed.
- **Important**: Comment Line 5 in `Dassl.pytorch-master/dassl/data/datasets/__init__.py` (otherwise there might not be outputs in the logs): 
```
# from .dg import *
```
- **Important**: Add the following code between Line 223-224 of `CLIP/clip/models.py` (check [CLIPood](https://github.com/thuml/CLIPood)):
```
x = x.type(self.conv1.weight.dtype)
```
- Move `imagenet_dg.py`, `officehome.py`, `terra_incognita.py`, `pacs.py`, `domainnet.py` to `CoOp/dataset/`.

##  Data
- Download ImageNet-A, ImageNet-R, ImageNet-V2, ImageNet-Sketch, ImageNet as instructed in [CoOp](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).
- Soft link the downloaded datasets to `DomainBed/domainbed/data/ImageNet/` to obtain the directory structure as follows (a for ImageNet-A, i for ImageNet, r for ImageNet-R, s for ImageNet-Sketch, v2 for ImageNet-V2):
```
ImageNet/
|-- a/
    |-- n01498041/
    |-- ...... (200 folders)
    |-- classnames.txt
|-- i/
    |-- train/
        |-- n01440764/
        |-- ...... (1000 folders)
        |-- classnames.txt
    |-- val/
        |-- n01440764/
        |-- ...... (1000 folders)
        |-- classnames.txt
|-- r/
    |-- n01443537/
    ...... (200 folders)
    |-- classnames.txt
|-- s/
    |-- n01440764/
    |-- ...... (1000 folders)
    |-- classnames.txt
|-- v2/
    |-- n01440764/
    |-- ...... (1000 folders)
    |-- classnames.txt
|-- classnames.txt
```


- Train your own domain experts first (requires >= 24G GPU): 
```
bash scripts/ImageNetDG_Step1.sh
```
- Then fine-tune CLIP with dedicated prompt guidance: 
```
bash scripts/ImageNetDG_Step2.sh
```
- Check `log/` for outputs.
