
# Solution for Kaggle Vesuvius Ink Detection Challenge

## Winning Solution

Our team used a two-stage model for this challenge.

1. The first stage involved training 3D models such as 3D CNNs, 3D UNETs, and UNETR. These models outputted a 3D volume with multiple channels, which were then flattened along the depth axis.
2. The second stage involved feeding this flattened output to a robust 2D segmentation model (SegFormer with different backbones) that was invariant to depth.

We utilized data augmentations and followed a standard approach using the AdamW optimizer, dice+bce loss, and hyperparameters. Our final solution is an ensemble of 9 different models.

Key success factors included the ensemble of multiple model variants, the selection of a depth-invariant solution, crop size selection, augmentations to ensure rotation and flip invariance, and the addition of post-processing techniques. We used standard tools such as PyTorch for implementation and trained our models in parallel on a setup with 3 A6000 GPUs, with each model taking roughly 10 hours to train.

## Requirements and Weights

The requirements for running our solution are included as part of the multiclass file. The weights are not included in this repository but can be requested from the team.

## Notebooks

We have two notebooks: `basev6` and `basev6-multiclass`, for the regular and multiclass solutions respectively. 

## External Folder

The `external` folder contains helper functions which need to be modified to run either multiclass or non-multiclass models. By default, they are set to run multiclass models. In particular, the model architecture needs to be configured to have either 1 or 3 outputs, and the data loading are the most critical parts to modify.

The files in the `external` folder are:

- `dataloading.py`
- `metrics.py`
- `models.py`
- `postprocessing.py`
- `training_procedures.py`
- `utils.py`

## External Repos Needed

#3D Unet Code
!cd /root && git clone git@github.com:wolny/pytorch-3dunet.git && cd pytorch-3dunet && pip install -e .
This command is included as part of the multiclass jupyter notebook.


# Citation

If you use this software or the associated data in your research, please cite us using the following format:

@software{vesuvius_2023,
author = {Ryan Chesler, Ted Kyi, Alexander Loftus, Aina Tersol},
title = {Solution for Kaggle Vesuvius Ink Detection Challenge},
year = {2023},
url = {TBD},
version = 0,
}