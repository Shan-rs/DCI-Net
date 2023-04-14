import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as F

def random_aug_transform():
    flip_h = T.RandomHorizontalFlip(p=1)
    flip_v = T.RandomVerticalFlip(p=1)
    angles = [0, 90, 180, 270]
    rot_angle = angles[np.random.choice(4)]
    rotate = T.RandomRotation((rot_angle, rot_angle))
    r = np.random.random()
    if r <= 0.25:
        flip_rot = T.Compose([flip_h, flip_v, rotate])
    elif r <= 0.5:
        flip_rot = T.Compose([flip_h, rotate])
    elif r <= 0.75:
        flip_rot = T.Compose([flip_v, flip_h, rotate])  
    else:
        flip_rot = T.Compose([flip_v, rotate])
    return flip_rot

class DehazingSet(data.Dataset):
    def __init__(self, root, aug, is_train):  

        self.aug = aug
        self.is_train = is_train
        self.gt_imgs_path = root + '/clear/'
        hazy_imgs = os.listdir(root + '/hazy/')
        self.hazy_imgs = [root + '/hazy/' + img for img in hazy_imgs]
        self.hazy_imgs.sort()
        self.transform = T.Compose([T.ToTensor()]) 

    def __getitem__(self, index):
            
        hazy_path = self.hazy_imgs[index]
#        gt_path = self.gt_imgs_path + hazy_path.split('_')[0].split('/')[-1] + '.jpg'
        gt_path = self.gt_imgs_path + hazy_path.split('/')[-1]
#        gt_path = self.gt_imgs_path + hazy_path.split('/')[-1][:-6] + '.jpg'
        hazy_im = Image.open(hazy_path)
        gt_img = Image.open(gt_path)
        if self.is_train:
            i,j,h,w=T.RandomCrop.get_params(hazy_im, output_size=(224, 224))
            hazy_im=F.crop(hazy_im,i,j,h,w)
            gt_img=F.crop(gt_img,i,j,h,w)
        if self.aug:
            flip_rot = random_aug_transform()
            hazy_img = self.transform(flip_rot(hazy_im))
            label = self.transform(flip_rot(gt_img))
        else:
            hazy_img = self.transform(hazy_im)
            label = self.transform(gt_img)
        
        return hazy_img, label
    
    def __len__(self):
        return len(self.hazy_imgs)

