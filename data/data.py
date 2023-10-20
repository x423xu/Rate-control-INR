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
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, CenterCrop

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

class UVGDataset(Dataset):
    def __init__(self, args, preload = True):
        self.args = args
        self.preload = preload
        if self.args.resize_list == "-1":
            self.crop_h, self.crop_w = [int(x) for x in args.crop_list.split('_')[:2]]
        else:
            self.crop_h, self.crop_w = [int(x) for x in args.resize_list.split('_')]
        self.crop_list, self.resize_list = args.crop_list, args.resize_list

        data, coord = self._read_data(args.data_path, vid_ids = [2], frames = args.num_frame)
        self.data = data
        self.coord = coord
        
        
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
            ).view(1, -1,2)
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
            ).view(sidelen[0],-1,3)
        return coord
    
    def _read_data(self, data_path, vid_ids = [0], frames = 1):
        def get_number(file_path):
            return int(file_path.split('_')[-1].rstrip('.png'))
        data = []
        vpath = os.listdir(data_path)
        vpath = [vpath[v] for v in vid_ids]
        
        if self.args.num_dim == 2:
                side_len = (self.crop_h, self.crop_w)
        elif self.args.num_dim == 3:
            side_len = (self.args.num_frame, self.crop_h, self.crop_w)
        coord = self.get_mgrid(side_len, dim=self.args.num_dim, max=1.0)
        for vp in vpath:
            vpath = os.path.join(data_path, vp)
            imgs_path = glob(os.path.join(vpath, '*.png'))   
            imgs_path = sorted(imgs_path, key=get_number)
            imgs_path = imgs_path[:frames]
            imgs = []
            for img in imgs_path:
                # rgb = read_image(img)
                rgb = Image.open(img)
                if self.args.out_channels == 1:
                    rgb = rgb.convert('L')
                rgb = self.img_transform(rgb)
                imgs.append([img, rgb])
            data.append(imgs)
        if len(data) == 1:
            data = data[0]
        return data, coord

    def img_transform(self, img):
        if self.crop_list != '-1': 
            crop_h, crop_w = [int(x) for x in self.crop_list.split('_')[:2]]
            transform = Compose([
                CenterCrop((crop_h, crop_w)),
                ])
        if self.resize_list != '-1':
            if '_' in self.resize_list:
                resize_h, resize_w = [int(x) for x in self.resize_list.split('_')]
                transform = Compose([
                    *transform.transforms,
                    Resize((resize_h, resize_w), interpolation=Image.BICUBIC),
                    ])
        transform = Compose([
            *transform.transforms,
            ToTensor(),
            Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
            ])
        img = transform(img)
        return img

    def __len__(self):
        return len(self.data)
        # return self.args.num_frame

    def get_transformed_coords(self, coords, i, j, side_len=256, add_ij=True):
        m = coords.shape[1]
        n = coords.shape[2]
        coords[...,0] = (coords[...,0] + 1) * (side_len[0] -1)/(m-1) - i * 2 * m/(m-1) - 1
        coords[...,1] = (coords[...,1] + 1) * (side_len[1] -1)/(n-1) - j * 2 * n/(n-1) - 1
        if add_ij:
            i_grid = torch.ones_like(coords[...,0]) * i
            j_grid = torch.ones_like(coords[...,0]) * j
            coords = torch.cat([coords, i_grid.unsqueeze(-1), j_grid.unsqueeze(-1)], dim=-1)
        return coords  

    def split_chunks(self, data, coords_transform=False,  add_ij=False):
        d = data.shape[-1]
        data = data.view(self.crop_h,self.crop_w, d)
        chunk_size_h = self.crop_h // self.args.num_grid  # 64
        chunk_size_w = self.crop_w // self.args.num_grid  # 64
        grids_row = data.split(chunk_size_h, dim=0) # [4, 64, 256, 2]
        grids = [r.split(chunk_size_w, dim=1) for r in grids_row] # [4, 4, 64, 64, 2]
        flat_grids = []
        for i in range(self.args.num_grid):
            for j in range(self.args.num_grid):
                flat_grids.append(grids[i][j])
        if coords_transform:
            transformed_chunks = [None] * self.args.num_grid**2
            for i in range(self.args.num_grid):
                for j in range(self.args.num_grid):
                    chunk = grids[i][j] # [64, 64, 2]
                    transformed_chunks[i*self.args.num_grid+j]=self.get_transformed_coords(chunk.clone().detach(), i, j, [self.crop_h,self.crop_w], add_ij)
            return transformed_chunks
        return flat_grids

    def __getitem__(self, idx):
        if not self.preload:
            data, coord_ = self._read_data(self.args.data_path)
            self.data = data[0]
            self.coord = coord_
        video = self.data[idx]
        frame_path = video[0]
        frame = video[1]
        coord = self.coord[idx]
        if not self.preload:
            imgs = self.img_transform(frame)/255.
        else:
            imgs = frame
        # transform coord to girdXgrid
        if self.args.parallel:
            coord = self.split_chunks(coord, coords_transform=self.args.coords_transform, add_ij=self.args.add_ij)
            coord = torch.stack(coord, dim=0)
            imgs = self.split_chunks(imgs.permute(1,2,0))
            imgs = torch.stack(imgs, dim=0)
        else:
            coord = self.split_chunks(coord, coords_transform=self.args.coords_transform, add_ij=self.args.add_ij)
            coord = torch.stack(coord, dim=0)
            imgs = imgs.permute(1,2,0)
        return {
            'path': frame_path,
            'imgs': imgs,
            'coord': coord
        }
    
class UVGDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.batch_size = args.batch_size
    
    def setup(self, stage):
        self.dataset = UVGDataset(self.args, preload=True)
        return
    
    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, pin_memory=True, num_workers=8, shuffle=False, prefetch_factor=6)
    
    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, pin_memory=True, num_workers=8, shuffle=False, prefetch_factor=6)
    
    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, pin_memory=True, num_workers=8, shuffle=False)
    
