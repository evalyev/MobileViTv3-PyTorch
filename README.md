# MobileViTv3-PyTorch : 
This repository helps get the MobileViTv3 model into PyTorch. It uses the [CVNets](https://arxiv.org/pdf/2206.02002.pdf) library and MobileViTv3 repository ([code](https://github.com/micronDLA/MobileViTv3)). <br>
<b>MobileViTv3: Mobile-Friendly Vision Transformer with Simple and Effective Fusion of Local, Global and Input Features [[arXiv](https://arxiv.org/abs/2209.15159)]</b>

## Installation:
I recommend to use Python 3.8+ and [PyTorch](https://pytorch.org) (version >= v1.8.0).

```bash
# Clone the repo
git clone https://github.com/evalyev/MobileViTv3-PyTorch.git
cd MobileViTv3-PyTorch

# install requirements
pip install -r requirements.txt
```


### MobileViTv3\-S,XS,XXS - easy of use!
Download the trained MobileViTv3 models from [here](https://github.com/micronDLA/MobileViTv3/releases/tag/v1.0.0) and save model as pt with bash code.
```bash
# Save model as pt
cd MobileViTv3-v1
python save_model.py --common.config-file ../models/MobileViTv3-v1/results_classification/mobilevitv3_S_e300_7930/config.yaml (config path)
```
Get model with pretrained weights in PyTorch.
```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = '../models/MobileViTv3-v1/results_classification/mobilevitv3_S_e300_7930/checkpoint_ema_best.pt'
model = torch.load(''model_structure.pt'')
model_weights = torch.load(model_path, map_location=device)
model.load_state_dict(model_weights)

output = model(image)
```


### MobileViTv3\-1.0,0.75,0.5
To be supplemented


## Trained models:

Download the trained MobileViTv3 models from [here](https://github.com/micronDLA/MobileViTv3/releases/tag/v1.0.0).
`checkpoint_ema_best.pt` files inside the model folder is used to generated the accuracy of models.
Low-latency models are build by reducing the number of MobileViTv3-blocks in 'layer4' from 4 to 2.
Please refer to the paper for more details.
Note that for the segmentation and detection, only the backbone architecture parameters are listed.

## Classification 

### ImageNet-1K:
| Model name | Accuracy (%) | Parameters (Million) | FLOPs (Million) | Foldername  |
| :---: | :---: | :---: | :---: | :---: |
| MobileViTv3\-S | 79.3 | 5.8 | 1841 | mobilevitv3\_S\_e300\_7930 |
| MobileViTv3\-XS | 76.7 | 2.5 | 927 | mobilevitv3\_XS\_e300\_7671 |
| MobileViTv3\-XXS | 70.98 | 1.2 | 289 | mobilevitv3\_XXS\_e300\_7098 |
| MobileViTv3\-1.0 | 78.64 | 5.1 | 1876 | mobilevitv3\_1\_0\_0 |
| MobileViTv3\-0.75 | 76.55 | 3.0 | 1064 | mobilevitv3\_0\_7\_5 |
| MobileViTv3\-0.5 | 72.33 | 1.4 | 481 | mobilevitv3\_0\_5\_0 |


### ImageNet-1K using low-latency models:
| Model name | Accuracy (%) | Parameters (Million) | FLOPs (Million) | Foldername  |
| :---: | :---: | :---: | :---: | :---: |
| MobileViTv3\-S-L2 | 79.06 | 5.2 | 1651 | mobilevitv3\_S\_L2\_e300\_7906 |
| MobileViTv3\-XS-L2 | 76.10 | 2.3 | 853 | mobilevitv3\_XS\_L2\_e300\_7610 |
| MobileViTv3\-XXS-L2 | 70.23 | 1.1 | 256 | mobilevitv3\_XXS\_L2\_e300\_7023 |

## Segmentation

### PASCAL VOC 2012:
| Model name | mIoU | Parameters (Million) | Foldername  |
| :---: | :---: | :---: | :---: |
| MobileViTv3\-S | 79.59 | 7.2 | mobilevitv3\_S\_voc\_e50\_7959 |
| MobileViTv3\-XS | 78.77 | 3.3 | mobilevitv3\_XS\_voc\_e50\_7877 |
| MobileViTv3\-XXS | 74.04 | 2.0 | mobilevitv3\_XXS\_voc\_e50\_7404 |
| MobileViTv3\-1.0 | 80.04 | 13.6 | mobilevitv3\_voc\_1\_0\_0 |
| MobileViTv3\-0.5 | 76.48 | 6.3 | mobilevitv3\_voc\_0\_5\_0 |

### ADE20K:
| Model name | mIoU | Parameters (Million) | Foldername  |
| :---: | :---: | :---: | :---: |
| MobileViTv3\-1.0 | 39.13 | 13.6 | mobilevitv3\_ade20k\_1\_0\_0 |
| MobileViTv3\-0.75 | 36.43 | 9.7 | mobilevitv3\_ade20k\_0\_7\_5  |
| MobileViTv3\-0.5 | 33.57 | 6.4 | mobilevitv3\_ade20k\_0\_5\_0 |

## Detection MS-COCO:
| Model name | mAP | Parameters (Million) | Foldername  |
| :---: | :---: | :---: | :---: |
| MobileViTv3\-S | 27.3 | 5.5 | mobilevitv3\_S\_coco\_e200\_2730 |
| MobileViTv3\-XS | 25.6 | 2.7 | mobilevitv3\_XS\_coco\_e200\_2560 |
| MobileViTv3\-XXS | 19.3 | 1.5 | mobilevitv3\_XXS\_coco\_e200\_1930 |
| MobileViTv3\-1.0 | 27.0 | 5.8 | mobilevitv3\_coco\_1\_0\_0 |
| MobileViTv3\-0.75 | 25.0 | 3.7 | mobilevitv3\_coco\_0\_7\_5 |
| MobileViTv3\-0.5 | 21.8 | 2.0 | mobilevitv3\_coco\_0\_5\_0 |


## Citation

If you find this repository useful, please consider giving a star :star: and citation :mega::
```
@inproceedings{wadekar2022mobilevitv3,
  title = {MobileViTv3: Mobile-Friendly Vision Transformer with Simple and Effective Fusion of Local, Global and Input Features},
  author = {Wadekar, Shakti N. and Chaurasia, Abhishek},
  doi = {10.48550/ARXIV.2209.15159},
  year = {2022}
}
```

