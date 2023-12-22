# Re-Write CMAL-Net for 43999 Project

**Source**

[CMAL-Net](https://github.com/Dichao-Liu/CMAL) | [TResNet](https://github.com/Alibaba-MIIL/TResNet)



## About

This is some my-self rewrite the code of CMAL for Project Sustainability Energy Fine-Grained Image Classification. I just do some rewriting (to more understand the code) and change a litte bit deprecated code and also make a litte bit modular. 

The focus rewrite is the Model where using Pre-trained of TResNet-L.

## Environment Requirements

This repository actualy running on Kaggle / Colab Notebook. To Ensure that your environment meets the following specifications:

- Python: 3.10.12
- PyTorch: 2.0.0
- Torchvision: 0.15.1
- Ubuntu: 22.04.3 LTS
- CUDA: 11.4 (or newer)

## Dependencies
To ensure completeness, refer to the original source [here](https://github.com/Dichao-Liu/CMAL?tab=readme-ov-file#dependencies)

For Dependecies is 2:

**1. Inplace-ABN**

For the Inplace-ABN dependency, follow the installation instructions outlined [here](https://github.com/Alibaba-MIIL/TResNet/blob/master/INPLACE_ABN_TIPS.md).

Install `Inplace-ABN` using `pip`:
```bash
pip install inplace-abn
```
**Note:** Ensure that CUDA is installed to enable successful installation.

## Dataset
Download your dataset and organize the structure to follow like this:
```bash
dataset/
├── train/
│   ├── class_1/
│   │   ├── 01.jpg
│   │   ├── 02.jpg
│   │   └── ...
│   ├── class_2/
│   │   ├── 01.jpg
│   │   ├── 02.jpg
│   │   └── ...
│   └── ...
└── test/
    ├── class_1/
    │   ├── 01.jpg
    │   ├── 02.jpg
    │   └── ...
    ├── class_2/
    │   ├── 01.jpg
    │   ├── 02.jpg
    │   └── ...
    └── ...
```
