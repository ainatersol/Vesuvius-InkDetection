
# Winning Solution for Kaggle Vesuvius Ink Detection Challenge

Our team used a two-stage model for this challenge.

1. The first stage involved training 3D models such as 3D CNNs, 3D UNETs, and UNETR. These models outputted a 3D volume with multiple channels, which were then flattened along the depth axis.
2. The second stage involved feeding this flattened output to a robust 2D segmentation model (SegFormer with different backbones) that was invariant to depth.

We utilized data augmentations and followed a standard approach using the AdamW optimizer, dice+bce loss, and hyperparameters. Our final solution is an ensemble of 9 different models.

Key success factors included the ensemble of multiple model variants, the selection of a depth-invariant solution, crop size selection, augmentations to ensure rotation and flip invariance, and the addition of post-processing techniques. We used standard tools such as PyTorch for implementation and trained our models in parallel on a setup with 3 A6000 GPUs, with each model taking roughly 10 hours to train.

## Quickstart

1. Install the requirements file: `pip install -r requirements.txt`
2. Run training using the `training` notebook OR download pretrained weights from https://www.kaggle.com/datasets/ryches/unet3d
3. Run inference using the `inference` notebook. Note that, by default, the inference notebook will run with the pretrained weights from the winning ensemble. If needed, replace the inference weights with the ones obtained in step2. 

Our notebooks are designed to run without any additional setup. Just click `restart kernel and run all` and you will be good to go. The default version of the `training` notebook will train the 3DCNN-Segformer 2 stage model and output a `.pth` weights file.

Disclaimer: Our winning solution is an ensemble of 9 different models, without the pretrained weights you will have to train them. The pretrained weights can be found here: https://www.kaggle.com/datasets/ryches/unet3d


## 1. Hardware requirements

We trained on 3 A6000 GPUs, if multiple GPUs are available the code will be run on all of them. We recommend using a similar system. We used vast.ai to rent the compute as well as personal ressources. 

We also did several runs on smaller systems, mostly RTX4090. If using a smaller such system, we recommend reducing the backbone to b1 instead of b3/b5 to ensure the model fits into memory.

## 2. Setup steps

Detail setup description

1. Requirements and weights
2. Notebooks
3. External repos and helper functions

### 2.1. Requirements and weights

The requirements.txt file should list all Python libraries that your notebooks depend on, and they will be installed using:

```
pip install -r requirements.txt
```

Pretrained weights used for training are included as part of the requirements.

### 2.2. Notebooks

We have two training notebooks: `training` and `training-multiclass`, for the regular and multiclass solutions respectively. 
We have one inference notebook.

### 2.3. External repos and helper functions

The `external` folder contains helper functions defining the different model architectures `model.py`, dataloading, postprocessing utils... 

Kindly note that, this implementation supports multiple sota architectures (3DCNN, 3DUNETR..). This should allow uses to easily switch between different verisons of the code (including all of the architectures from the 9-model ensemble that won the competition). The different models are available undel `model.py` and to load them just replace the name in the training notebook such as:

```
model = UNETR_Segformer(CFG) 
```

#### External Repos/Command Needed

This code relies on several contributions by other users, most of them are pip installable and have been included as part of the requirements. However, we were finding difficulties pip installing the unet3d implementation. Hence, we have 'brutally scraped' the bits we needed. You will find them under pytorch3dunet/unet3d/

The original repo can be cloned as follows:
```
cd /root && git clone git@github.com:wolny/pytorch-3dunet.git && cd pytorch-3dunet && pip install -e .
```
(If you want to use the official repo; remember to add the repo to the python path to be able to use the content in the unet3d folder)

```
import sys
sys.path.append('/path/to/your/directory')
```

You should have all of the external information used within the kaggle datasets below.
https://www.kaggle.com/datasets/ryches/einops
https://www.kaggle.com/datasets/ryches/unet3d

Here are the links to other repos used as part of this work.
https://timm.fast.ai/
https://github.com/lukemelas/EfficientNet-PyTorch

you may also need the following commands if you encounter errors while importing open-cv
```
!sudo apt-get update
!sudo apt-get install -y libgl1-mesa-glx
```

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

## 3. Data preparation

The weights for our model are to heavy to load into github but they can be found in:

https://www.kaggle.com/datasets/ryches/unet3d

The weights included in the winning ensemble:

```
"weight_path": "/kaggle/input/3d-unet/3d_unet_segformer_1024_3d_unet_segformer_final_all_train.pth"
"weight_path": "/kaggle/input/3d-unet/3d_unet_segformer_512_3d_unet_segformer_final.pth"
"weight_path": "/kaggle/input/3d-unet/3dunet_segformer_1024_swa_slow_all_train_3dunet_segformer_final.pth"
"weight_path": "/kaggle/input/3d-unet/3dcnn_segformer_all_train_swa_3dcnn_segformer_final_swa.pth"
"weight_path": "/kaggle/input/3d-unet/b5_long_train_all_frags_3dcnn_segformer_b5_final_swa.pth"
"weight_path": "/kaggle/input/3d-unet/b3_more_fmaps_all_train_3dcnn_segformerb364_final_swa.pth"
"weight_path": "/kaggle/input/3d-unet/Jumbo_Unet_Jumbo_Unet_69_final_swa_all_train.pth"
"weight_path": "/kaggle/input/3d-unet/jumbo_unetr_unetr_888_final_swa_all_train_long.pth"
"weight_path": "/kaggle/input/3d-unet/unetr_multiclass_NOVALIDATION_512_b5_unet_final_swa_all_train.pth"
```
Make sure to download them to reproduce our solution.
```
pip install kaggle
export KAGGLE_USERNAME=<kaggle_username>
export KAGGLE_KEY=<kaggle key>
kaggle datasets download -d ryches/unet3d
```

## 4. Training

Run training using the `training` notebook. (you can skip this step if you request pretrained weights)

Our notebooks are designed to run without any additional setup. Just click `restart kernel and run all` and you will be good to go. The default version of the `training` notebook will train the 3DCNN-Segformer 2 stage model and output a `.pth` weights file.

You can modify the model trained by simply switching the model name:

```
model = UNETR_Segformer(CFG) 
```

The models included in our winning ensemble:
- CNN3D_Segformer
- cnn3d_more_filters
- unet3d_segformer
- unet3d_segformer_jumbo
- UNETR_Segformer
- UNETR_SegformerMC
- UNETR_MulticlassSegformer*

* this model should be trained using the multiclass training notebook


## 5. Inference

Run inference using the `inference` notebook. Note that, by default, the inference notebook will run with the pretrained weights from the winning ensemble. If needed, replace the inference weights with the ones obtained during training.

Our notebooks are designed to run without any additional setup. Just click `restart kernel and run all` and you will be good to go. 

Disclaimer: Our winning solution is an ensemble of 9 different models, without the pretrained weights you will have to train them. The pretrained weights can be found here: https://www.kaggle.com/datasets/ryches/unet3d


# Citation

If you use this software or the associated data in your research, please cite us using the following format:

@software{vesuvius_2023,
author = {Ryan Chesler, Ted Kyi, Alexander Loftus, Aina Tersol},
title = {Solution for Kaggle Vesuvius Ink Detection Challenge},
year = {2023},
url = {https://github.com/ainatersol/Vesuvius-InkDetection},
version = 0,
}

The code in this repository is MIT lisence, but the pre-training model and libraries used is under the license of the major source. For example, segformer is licensed under nvidia.
