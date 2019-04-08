import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import utils
from PIL import Image, ImageFile
from torchvision.transforms import transforms
import numpy as np
import random

#from utils import n_class

new_h = 576
new_w = 800
size = 224
image_transform = transforms.Compose([
     transforms.Resize(size),
     transforms.CenterCrop(size),
     transforms.ToTensor(),
     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
 ])

# label_transformation = transforms.Compose([
#     transforms.ToTensor()
# ])

class SegDataset(Dataset):
    def __init__(self, dir_path, crop=True):
        self.image_dir = os.path.join(dir_path, 'images')
        self.label_dir = os.path.join(dir_path, 'labels')
        self.all_names = [ f for f in os.listdir(self.image_dir)]
        self.crop = crop

    def __len__(self):
        return len(self.all_names)

    def __getitem__(self, idx):
        name = self.all_names[idx]
        image = Image.open(os.path.join(self.image_dir, name))
        label = Image.open(os.path.join(self.label_dir, name))

        img_tensor, lbl_tensor = self.transform(np.array(image, dtype=np.uint8), np.array(label, dtype=np.int32))
        return img_tensor, lbl_tensor

        # create one-hot encoding
        # h, w = lbl_tensor.size()
        # target = torch.zeros(n_class, h, w)
        # for c in range(n_class):
        #     target[c][lbl_tensor == c] = 1

        # return img_tensor, target.long()

    def transform(self, img, lbl):
        new_img = image_transform(img)
        new_lbl = image_transform(lbl)
        return new_img, new_lbl
    '''
    def transform(self, img, lbl):
        if self.crop:
            h, w, _ = img.shape
            top = random.randint(0, h - new_h)
            left = random.randint(0, w - new_w)
            img = img[top:top + new_h, left:left + new_w]
            lbl = lbl[top:top + new_h, left:left + new_w]

        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        # img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl
    '''