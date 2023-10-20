from torch.utils.data import Dataset, DataLoader
import os
import decord
import re
from PIL import Image
from torchvision.transforms.functional import center_crop, resize
# from torch.nn.functional import interpolate
from torchvision.transforms.functional import to_tensor, normalize
from torchvision.transforms import InterpolationMode

import pytorch_lightning as pl

def get_number_suffix(filename):
    match = re.search(r'_(\d+)\.png$', filename)
    return int(match.group(1)) if match else 0


'''
dataset for bunny dataset,
input: args, preload
    args: 
        data_path: directory contains bunny frmames
        crop_list: crop size for each frame, e.g. 100_100 crop to (h,w) = (100, 100), "-1" means no crop
        resize_list: resize size for each frame, e.g. 100_100 resize to (h,w) = (100, 100); "-1" means no resize
'''
class VideoDataSet(Dataset):
    def __init__(self, args, preload=False):
        self.args = args
        if os.path.isfile(args.data_path):
            self.video = decord.VideoReader(args.data_path)
        else:
            self.video = [os.path.join(args.data_path, x) for x in sorted(os.listdir(args.data_path))]
        self.video = sorted(self.video, key=get_number_suffix)
        # Resize the input video and center crop
        self.crop_list, self.resize_list = args.crop_list, args.resize_list  
        # import pdb; pdb.set_trace; from IPython import embed; embed()     
        first_frame = self.img_transform(self.img_load(0))
        self.final_size = first_frame.size(-2) * first_frame.size(-1)
        if preload:
            self.video = [self.img_transform(self.img_load(idx)) for idx in range(len(self.video))] 
        self.preload = preload

    def img_load(self, idx):
        if isinstance(self.video[idx], str):
            img = Image.open(self.video[idx])
        else:
            img = self.video[idx]
        return img


    def img_transform(self, img):
        # to tensor, normalize
        if isinstance(img, Image.Image):
            img = to_tensor(img)
            img = normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            raise NotImplementedError
        if self.crop_list != '-1': 
            crop_h, crop_w = [int(x) for x in self.crop_list.split('_')[:2]]
            if 'last' not in self.crop_list:
                img = center_crop(img, (crop_h, crop_w))
        if self.resize_list != '-1':
            if '_' in self.resize_list:
                resize_h, resize_w = [int(x) for x in self.resize_list.split('_')]
                # img = interpolate(img, size=(resize_h, resize_w), mode='bicubic')
                # interpolate the img to (resize_h, resize_w)
                img = resize(img, (resize_h, resize_w), interpolation=InterpolationMode.BICUBIC)
            else:
                resize_hw = int(self.resize_list)
                img = resize(img, resize_hw,  'bicubic')
        if 'last' in self.crop_list:
            img = center_crop(img, (crop_h, crop_w))
        return img

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        if self.preload:
            tensor_image = self.video[idx]
        else:
            tensor_image = self.img_transform(self.img_load(idx))
        norm_idx = float(idx) / len(self.video)
        sample = {'img': tensor_image, 'idx': idx, 'norm_idx': norm_idx}
        
        return sample

'''
data module for bunny dataset
batch_size
num_workers
'''
class UVGDataModule(pl.LightningDataModule):
    def __init__(self, args, preload= False):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.preload = preload
    
    def setup(self, stage):
        self.dataset = VideoDataSet(self.args, preload=self.preload)
        return
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, shuffle=False, prefetch_factor=6)
    
    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, shuffle=False, prefetch_factor=6)
    
    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, pin_memory=True, num_workers=self.num_workers, shuffle=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/xxy/Documents/data/UVG/ReadySetGo', help='data path for vid')
    parser.add_argument('--crop_list', type=str, default='960_1920', help='video crop size',)
    parser.add_argument('--resize_list', type=str, default='256_256', help='video resize size',)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers')
    args = parser.parse_args()
    bunny_module = UVGDataModule(args, preload=True)
    bunny_module.setup('train')
    for batch in bunny_module.train_dataloader():
        print(batch['img'].shape)