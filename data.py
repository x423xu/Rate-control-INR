from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from glob import glob
import numpy as np
import torch
import pytorch_lightning as pl
from torchvision.transforms.functional import center_crop, resize
from torch.nn.functional import interpolate
from torchvision.io import read_image

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

class UVGDataset(Dataset):
    def __init__(self, args, preload = True):
        self.args = args
        self.preload = preload
        self.crop_list, self.resize_list = args.crop_list, args.resize_list
        self.data = self._read_data(args.data_path)
        
    def get_mgrid(self, sidelen, dim=2, max=1.0):
        '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
        sidelen: int
        dim: int'''
        assert len(sidelen) == dim, 'len(sidelen) should match dim'
        if dim==2:
            coord = torch.stack(
                torch.meshgrid(
                    [
                        torch.linspace(-max, max, sidelen[0]),
                        torch.linspace(-max, max, sidelen[1]),
                    ]
                ),
                dim=-1,
            ).view(-1,2)
        elif dim==3:
            coord = torch.stack(
                torch.meshgrid(
                    [
                        torch.linspace(-max, max, sidelen[0]),
                        torch.linspace(-max, max, sidelen[1]),
                        torch.linspace(-max, max, sidelen[2]),
                    ]
                ),
                dim=-1,
            ).view(-1,3)
        return coord
    
    def _read_data(self, data_path):
        data = []
        vpath = os.listdir(data_path)
        for vp in vpath:
            vpath = os.path.join(data_path, vp)
            imgs_path = glob(os.path.join(vpath, '*.png'))
            crop_h, crop_w = [int(x) for x in self.crop_list.split('_')[:2]]
            coord = self.get_mgrid((crop_h, crop_w), dim=self.args.num_dim, max=1.0)
            imgs = [[img, self.img_transform(read_image(img))/255., coord] for img in imgs_path]
            data.append(imgs)
        return data

    def img_transform(self, img):
        if self.crop_list != '-1': 
            crop_h, crop_w = [int(x) for x in self.crop_list.split('_')[:2]]
            if 'last' not in self.crop_list:
                img = center_crop(img, (crop_h, crop_w))
        if self.resize_list != '-1':
            if '_' in self.resize_list:
                resize_h, resize_w = [int(x) for x in self.resize_list.split('_')]
                img = interpolate(img, (resize_h, resize_w), 'bicubic')
            else:
                resize_hw = int(self.resize_list)
                img = resize(img, resize_hw,  'bicubic')
        if 'last' in self.crop_list:
            img = center_crop(img, (crop_h, crop_w))
        return img

    def __len__(self):
        # return len(self.data)
        return 1

    def __getitem__(self, idx):
        if not self.preload:
            self.data = self._read_data(self.args.data_path)
        video = self.data[idx]
        frame_path = [v[0] for v in video]
        frames = [v[1] for v in video]
        coord = [v[2] for v in video]
        if not self.preload:
            imgs = []
            for frame in frames:
                frame = self.img_transform(frame)
                imgs.append(frame)
        else:
            imgs = frames
        return {
            'path': frame_path,
            'imgs': torch.stack(imgs, dim=0),
            'coord': coord
        }
    
class UVGDataModule(pl.LightningDataModule):
    def __init__(self, args, batch_size = 1):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
    
    def setup(self, stage):
        self.dataset = UVGDataset(self.args, preload=True)
        return
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, pin_memory=True, num_workers=2, shuffle=False)
    
    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, pin_memory=True, num_workers=2, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, pin_memory=True, num_workers=2, shuffle=False)
    
