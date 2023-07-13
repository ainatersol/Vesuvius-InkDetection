
# Solution for Kaggle Vesuvius Ink Detection Challenge

## Winning Solution

Our team used a two-stage model for this challenge.

1. The first stage involved training 3D models such as 3D CNNs, 3D UNETs, and UNETR. These models outputted a 3D volume with multiple channels, which were then flattened along the depth axis.
2. The second stage involved feeding this flattened output to a robust 2D segmentation model (SegFormer with different backbones) that was invariant to depth.

We utilized data augmentations and followed a standard approach using the AdamW optimizer, dice+bce loss, and hyperparameters. Our final solution is an ensemble of 9 different models.

Key success factors included the ensemble of multiple model variants, the selection of a depth-invariant solution, crop size selection, augmentations to ensure rotation and flip invariance, and the addition of post-processing techniques. We used standard tools such as PyTorch for implementation and trained our models in parallel on a setup with 3 A6000 GPUs, with each model taking roughly 10 hours to train.

## Requirements and Weights

The requirements.txt file should list all Python libraries that your notebooks depend on, and they will be installed using:

`pip install -r requirements.txt`

## Notebooks

We have two notebooks: `training` and `training-multiclass`, for the regular and multiclass solutions respectively. 

## External Folder

The `external` folder contains helper functions which need to be modified to run either multiclass or non-multiclass models. By default, they are set to run multiclass models. In particular, the model architecture needs to be configured to have either 1 or 3 outputs, and the data loading are the most critical parts to modify.

The files in the `external` folder are:

- `dataloading.py`
- `metrics.py`
- `models.py`
- `postprocessing.py`
- `training_procedures.py`
- `utils.py`

## External Repos/Command Needed

This code relies on several contributions by other users:

#3D Unet Code 

We were finding difficulties pip installing the unet3d implementation. Hence, we have 'brutally scraped' the bits we needed. You will find them under externals/unet3d/

The original repo can be cloned as follows:
!cd /root && git clone git@github.com:wolny/pytorch-3dunet.git && cd pytorch-3dunet && pip install -e .

(If you want to use the official repo; remember to add the repo to the python path to be able to use the content in the unet3d folder)

import sys
sys.path.append('/path/to/your/directory')

You'll also need `timm-pytorch-image-models`, and `efficientnet-pytorch`. You may need a custom einops. On the actual notebook we submitted, we put the models on the github links above into the kaggle datasets below:
https://www.kaggle.com/datasets/ryches/einops
https://www.kaggle.com/datasets/ryches/unet3d

Here are the links to the other repos. 
https://timm.fast.ai/
https://github.com/lukemelas/EfficientNet-PyTorch

you may also need the following commands if you encounter errors while importing open-cv

`
!sudo apt-get update
!sudo apt-get install -y libgl1-mesa-glx
`

@article {10.7554/eLife.57613,
article_type = {journal},
title = {Accurate and versatile 3D segmentation of plant tissues at cellular resolution},
author = {Wolny, Adrian and Cerrone, Lorenzo and Vijayan, Athul and Tofanelli, Rachele and Barro, Amaya Vilches and Louveaux, Marion and Wenzl, Christian and Strauss, Sören and Wilson-Sánchez, David and Lymbouridou, Rena and Steigleder, Susanne S and Pape, Constantin and Bailoni, Alberto and Duran-Nebreda, Salva and Bassel, George W and Lohmann, Jan U and Tsiantis, Miltos and Hamprecht, Fred A and Schneitz, Kay and Maizel, Alexis and Kreshuk, Anna},
editor = {Hardtke, Christian S and Bergmann, Dominique C and Bergmann, Dominique C and Graeff, Moritz},
volume = 9,
year = 2020,
month = {jul},
pub_date = {2020-07-29},
pages = {e57613},
citation = {eLife 2020;9:e57613},
doi = {10.7554/eLife.57613},
url = {https://doi.org/10.7554/eLife.57613},
keywords = {instance segmentation, cell segmentation, deep learning, image analysis},
journal = {eLife},
issn = {2050-084X},
publisher = {eLife Sciences Publications, Ltd},
}


# Citation

If you use this software or the associated data in your research, please cite us using the following format:

@software{vesuvius_2023,
author = {Ryan Chesler, Ted Kyi, Alexander Loftus, Aina Tersol},
title = {Solution for Kaggle Vesuvius Ink Detection Challenge},
year = {2023},
url = {https://github.com/ainatersol/Vesuvius-InkDetection},
version = 0,
}