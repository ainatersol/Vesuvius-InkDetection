import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score, roc_auc_score

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def fbeta_numpy(targets, preds, beta=0.5, smooth=1e-5):
    """
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """
    y_true_count = targets.sum()
    ctp = preds[targets==1].sum()
    cfp = preds[targets==0].sum()
    ctn = ((preds == 0)[targets == 0]).sum()
    cfn = ((preds == 0)[targets == 1]).sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice, c_precision, c_recall, ctp, cfp, ctn, cfn

def calc_fbeta(mask, mask_pred):
    mask = mask.astype(int).flatten()
    # mask = np.where(mask == 2, 1, 0)
    mask_pred = mask_pred.flatten()

    best_th = 0
    best_dice = 0
    best_metrics = []
    for th in np.array(range(10, 100+1, 5)) / 100:
        
        # dice = fbeta_score(mask, (mask_pred >= th).astype(int), beta=0.5)
        dice, precision, recall, ctp, cfp, ctn, cfn = fbeta_numpy(mask, (mask_pred >= th).astype(int), beta=0.5)
        print(f'th: {th}, fbeta: {dice}, precision: {precision}, recall: {recall}')
        if dice > best_dice:
            best_dice = dice
            best_th = th
            best_precision = precision
            best_recall = recall
            best_metrics = [ctp, cfp, ctn, cfn]
        
        roc_auc = roc_auc_score(mask, np.where(mask > best_th, 1, 0))
                
    print(f'best_th: {best_th}, fbeta: {best_dice}, precision: {best_precision}, recall: {best_recall}, auc: {roc_auc}')
    return best_dice, best_th, best_metrics