from pytorch3dunet.unet3d.model import get_model
from transformers import SegformerForSemanticSegmentation, SegformerModel
from torch import nn
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F 

class CNN3D_MulticlassSegformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 3D Convolution Layers
        self.conv3d_1 = nn.Conv3d(1, 4, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(4, 8, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))

        # Segformer for Binary Classification
        self.xy_encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
                                            "nvidia/mit-b1",
                                            num_labels=3,  #Change
                                            ignore_mismatched_sizes=True,
                                            num_channels=48  # Note: Channels set to 33 or 35 one hot(32 + 1 for the mask)
                                )
        
        # Upscale Layers
        self.upscaler1 = nn.ConvTranspose2d(3, 3, kernel_size=(4, 4), stride=2, padding=1) # Note: Channels set to 3
        self.upscaler2 = nn.ConvTranspose2d(3, 3, kernel_size=(4, 4), stride=2, padding=1) # Note: Channels set to 3

    def forward(self, image):
        # print(image.shape)
        output = self.conv3d_1(image)
        output = self.conv3d_2(output)
        output = self.conv3d_3(output)
        output = self.conv3d_4(output).max(axis=2)[0]

        # Segmentation Output
        segmentation_output = self.xy_encoder_2d(output).logits
        segmentation_output = self.upscaler1(segmentation_output)
        segmentation_output = self.upscaler2(segmentation_output)
        
        return segmentation_output
    

class CNN3D_Segformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.conv3d_1 = nn.Conv3d(1, 4, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(4, 8, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))



        self.xy_encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
                                            "nvidia/mit-b3",
                                            num_labels=1,
                                            ignore_mismatched_sizes=True,
                                            num_channels=32
                                )
        self.upscaler1 = nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride = 2, padding=1)
        self.upscaler2 = nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride = 2, padding=1)
    def forward(self, image):
        output = self.conv3d_1(image)
        output = self.conv3d_2(output)
        output = self.conv3d_3(output)
        output = self.conv3d_4(output).max(axis = 2)[0]
        output = self.xy_encoder_2d(output).logits
        output = self.upscaler1(output)
        output = self.upscaler2(output)
        return output

class CNN3D_SegformerBIG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.conv3d_1 = nn.Conv3d(1, 4, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(4, 8, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))



        self.xy_encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
                                            "nvidia/mit-b5",
                                            num_labels=1,
                                            ignore_mismatched_sizes=True,
                                            num_channels=32,
                                            # attention_probs_dropout_prob = 0.3,
                                            classifier_dropout_prob = 0.3,
                                            drop_path_rate = 0.3,
                                            hidden_dropout_prob = 0.3,
                                            
                                            
                                )
        self.upscaler1 = nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride = 2, padding=1)
        self.upscaler2 = nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride = 2, padding=1)
    def forward(self, image):
        output = self.conv3d_1(image)
        output = self.conv3d_2(output)
        output = self.conv3d_3(output)
        output = self.conv3d_4(output).max(axis = 2)[0]
        output = self.xy_encoder_2d(output).logits
        output = self.upscaler1(output)
        output = self.upscaler2(output)
        return output
    
class CNN3D_SegformerB4(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.conv3d_1 = nn.Conv3d(1, 4, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(4, 8, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))



        self.xy_encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
                                            "nvidia/mit-b4",
                                            num_labels=1,
                                            ignore_mismatched_sizes=True,
                                            num_channels=32,
                                            
                                            
                                )
        self.upscaler1 = nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride = 2, padding=1)
        self.upscaler2 = nn.ConvTranspose2d(1, 1, kernel_size=(4, 4), stride = 2, padding=1)
    def forward(self, image):
        output = self.conv3d_1(image)
        output = self.conv3d_2(output)
        output = self.conv3d_3(output)
        output = self.conv3d_4(output).max(axis = 2)[0]
        output = self.xy_encoder_2d(output).logits
        output = self.upscaler1(output)
        output = self.upscaler2(output)
        return output
    

class CNN3D_Unet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.conv3d_1 = nn.Conv3d(1, 4, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(4, 8, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))

        self.xy_encoder_2d = smp.Unet(
                encoder_name="tu-efficientnetv2_s",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=32,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=3,                      # model output channels (number of classes in your dataset)
            )
        
    def forward(self, image):
        output = self.conv3d_1(image)
        output = self.conv3d_2(output)
        output = self.conv3d_3(output)
        output = self.conv3d_4(output).max(axis = 2)[0]
        output = self.xy_encoder_2d(output)
        return output
    
class CNN3D_MANet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.conv3d_1 = nn.Conv3d(1, 4, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(4, 8, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))

        self.xy_encoder_2d = smp.MAnet(
                encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=32,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,                      # model output channels (number of classes in your dataset)
            )
        
    def forward(self, image):
        output = self.conv3d_1(image)
        output = self.conv3d_2(output)
        output = self.conv3d_3(output)
        output = self.conv3d_4(output).max(axis = 2)[0]
        output = self.xy_encoder_2d(output)
        return output

class CNN3D_EfficientUnetplusplusb5(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.conv3d_1 = nn.Conv3d(1, 4, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_2 = nn.Conv3d(4, 8, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_3 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.conv3d_4 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))

        self.xy_encoder_2d = smp.EfficientUnetPlusPlus(
                encoder_name="timm-efficientnet-b5",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=32,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,                      # model output channels (number of classes in your dataset)
            )
        
    def forward(self, image):
        output = self.conv3d_1(image)
        output = self.conv3d_2(output)
        output = self.conv3d_3(output)
        output = self.conv3d_4(output).max(axis = 2)[0]
        output = self.xy_encoder_2d(output)
        return output
    
    
class Unet3D_Segformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.model = get_model({"name":"UNet3D", "in_channels":1, "out_channels":16, "f_maps":8, "num_groups":4, "is_segmentation":False})
        self.encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
                                            "nvidia/mit-b5",
                                            num_labels=3,
                                            id2label={2:"ink", 0:"b", 1:"papyrus"},
                                            label2id={"ink":2, "papyrus":1, "b":0},
                                            ignore_mismatched_sizes=True,
                                            num_channels=16
                                )
        self.upscaler1 = nn.ConvTranspose2d(3, 3, kernel_size=(4, 4), stride = 2, padding=1)
        self.upscaler2 = nn.ConvTranspose2d(3, 3, kernel_size=(4, 4), stride = 2, padding=1)
    def forward(self, image):
        output = self.model(image).max(axis = 2)[0]
        output = self.encoder_2d(output).logits
        output = self.upscaler1(output)
        output = self.upscaler2(output)
        return output

from unetr import UNETR
class UNETR_Segformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        dropout = .1
        self.dropout = nn.Dropout2d(dropout)
        self.encoder = UNETR(
            in_channels=1,
            out_channels=32,
            img_size=(16, 512, 512),
        )
        self.encoder_2d = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b5",
            num_labels=3,   #switch to 1 for single class
            ignore_mismatched_sizes=True,
            num_channels=32
        )
        self.upscaler1 = nn.ConvTranspose2d(
            3, 3, kernel_size=(4, 4), stride=2, padding=1)
        self.upscaler2 = nn.ConvTranspose2d(
            3, 3, kernel_size=(4, 4), stride=2, padding=1)

    def forward(self, image):
        output = self.encoder(image).max(axis=2)[0]
        output = self.dropout(output)
        output = self.encoder_2d(output).logits
        output = self.upscaler1(output)
        output = self.upscaler2(output)
        return output
    
