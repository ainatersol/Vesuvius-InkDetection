from warmup_scheduler import GradualWarmupScheduler
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from externals.metrics import AverageMeter
import torch.nn.functional as F

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.epochs, eta_min=1e-7)
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    return scheduler

def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)

class GeneralizedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(GeneralizedFocalLoss, self).__init__()
        self.alpha = alpha  # Weighting factor for class imbalance
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction  # Reduction method for the loss

    def forward(self, y_pred, y_true):
        ce_loss = F.cross_entropy(y_pred, y_true, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            # Create a mask for intensity values 
            mask_intensity = torch.zeros(y_pred.size()).to(y_true.device)
            for idx in range(y_pred.size(1)):
                mask_intensity += torch.where(y_true == idx, self.alpha[idx], 0).float().to(y_true.device)
  
            weight_tensor = mask_intensity.sum(axis=1)
            focal_loss = weight_tensor * focal_loss
            
            del mask_intensity, weight_tensor
        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()

        return focal_loss

DiceLoss = smp.losses.DiceLoss(mode='binary')
MulticlassDiceLoss = smp.losses.DiceLoss(mode='multilabel')
BCELoss = smp.losses.SoftBCEWithLogitsLoss()
alpha = torch.tensor([0, 1, 2])
# FocalLoss = GeneralizedFocalLoss(alpha=torch.tensor([1, 1, 2]), gamma=2)
# CrossEntropyLoss = nn.CrossEntropyLoss()

def criterion(y_pred, y_true):
    return 0.5 * BCELoss(y_pred, y_true) + 0.5 * DiceLoss(y_pred, y_true)

def multiclass_criterion(y_pred, y_true):
    # alpha = torch.tensor([1, 1, 2]).to(y_true.device)
    # FocalLoss = GeneralizedFocalLoss( alpha=alpha, gamma=2).to(y_true.device)
    return 0.5 * BCELoss(y_pred, y_true) + 0.5 * MulticlassDiceLoss(y_pred, y_true)