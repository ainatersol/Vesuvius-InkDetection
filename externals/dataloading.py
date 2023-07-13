from tqdm.auto import tqdm
import numpy as np
import cv2
import albumentations as A
from torch.utils.data import Dataset

def read_image_mask(CFG, fragment_id, is_multiclass=False):

    images = []

    mid = 65 // 2
    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2
    idxs = range(start, end)
    
    for i in tqdm(idxs):
        
        image = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/surface_volume/{i:02}.tif", 0)

        pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
        pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)
        
    images = np.stack(images, axis=2)
    
    if is_multiclass: # multiclass code
        
        lbl = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0)
        mask_background = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/mask.png", 0)

        lbl /= 255.0
        mask_background /= 255.0

        mask = mask_background + lbl
        mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)
    
    else: # single-class code
        mask = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0)
        mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)
        mask /= 255.0
        
    mask = mask.astype('float32')
    
    return images, mask

def read_image_mask_downsampling(CFG, fragment_id, t='mean'):
    images = []

    mid = 65 // 2
    start = mid - CFG.in_chans 
    end = mid + CFG.in_chans
    idxs = range(start, end, 2)  # take each pair of slices

    for i in tqdm(idxs):
        
        image1 = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/surface_volume/{i:02}.tif", 0)
        
        if t == 'mean':
             # load two slices
            image2 = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/surface_volume/{(i+1):02}.tif", 0)

            # make sure the two images have the same size
            assert image1.shape == image2.shape
            
            # downsample
            image = np.mean([image1, image2], axis=0).astype(np.uint8)
            del image1, image2
            
        if t == 'remove':
            image = image1
            del image1    
            
        pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
        pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)
        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)

    images = np.stack(images, axis=2)
    
    # single-class code
    # mask = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0)
    # mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)
    # mask /= 255.0
    
    # multiclass code
    lbl = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0)
    mask_background = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/mask.png", 0)
    lbl = lbl / 255.0
    mask_background = mask_background / 255.0
    mask = mask_background + lbl
    
    mask = mask.astype('float32')
    
        
    return images, mask

def get_train_valid_dataset(CFG):
    train_images = []
    train_masks = []

    valid_images = []
    valid_masks = []
    valid_xyxys = []

    for fragment_id in range(1, 4):

        image, mask = read_image_mask(CFG, fragment_id)

        x1_list = list(range(0, image.shape[1]-CFG.tile_size+1, CFG.stride))
        y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, CFG.stride))

        for y1 in y1_list:
            for index, x1 in enumerate(x1_list):
                y2 = y1 + CFG.tile_size
                x2 = x1 + CFG.tile_size
                # xyxys.append((x1, y1, x2, y2))
                if fragment_id == CFG.valid_id:
                    if image[y1:y2, x1:x2, None].max() != 0:
                        valid_images.append(image[y1:y2, x1:x2])
                        valid_masks.append(mask[y1:y2, x1:x2, None])
                        valid_xyxys.append([x1, y1, x2, y2])
                else:
                    # offset = np.random.randint(0, 0)
                    offset = 0
                    if image[y1+offset:y2+offset, x1+offset:x2+offset, None].max() != 0:
                        train_images.append(image[y1+offset:y2+offset, x1+offset:x2+offset])
                        train_masks.append(mask[y1+offset:y2+offset, x1+offset:x2+offset, None])
    if CFG.valid_id == None:
        return train_images, train_masks
    return train_images, train_masks, valid_images, valid_masks, valid_xyxys

def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)
    # print(aug)
    return aug

class CustomDataset(Dataset):
    def __init__(self, images, cfg, labels=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # return len(self.df)
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            data = self.transform(image=image, mask=label)
            image = data['image']
            label = data['mask']

        return image[None, :, :, :], label