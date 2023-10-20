from typing import List, Optional, Union
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
import pytorch_lightning as pl
from pytorch_msssim import ms_ssim, ssim
from pytorch_lightning.utilities.model_summary import ModelSummary
from concurrent.futures import ThreadPoolExecutor
from queue import Empty 


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
        # relu activation
        # return F.relu(self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))
        
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))
            # if i<hidden_layers-1:
            #     self.net.append(nn.Dropout(0.25))
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords      
    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()
        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1
        return activations
    
class GridSiren(nn.Module):
    def __init__(self, args, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30., grid_size=4, num_net=10, coords_transform=True,
                 use_mask=False, add_ij=True):
        super().__init__()
        self.args = args
        if self.args.resize_list == "-1":
            self.crop_h, self.crop_w = [int(x) for x in args.crop_list.split('_')[:2]]
        else:
            self.crop_h, self.crop_w = [int(x) for x in args.resize_list.split('_')]
        self.grid_size = grid_size
        self.num_net = grid_size ** 2
        self.coords_transform = coords_transform
        self.use_mask = use_mask
        self.add_ij = add_ij
        # create a mask of size grid_size x grid_size x num_net
        if self.use_mask:
            self.mask = nn.Parameter(5e-3 * torch.randn(grid_size, grid_size, self.num_net))  # [4, 4, 4]
            self.num_net = num_net
        # transform coordinates in each grid to between -1 and 1
        # if add_ij, add i and j to the coordinates
        if coords_transform and add_ij:
            in_features = in_features + 2
        # construct a base Siren from which to copy weights
        # self.base_net = Siren(in_features, hidden_features, hidden_layers, out_features, outermost_linear, 
        #                     first_omega_0, hidden_omega_0)
        
        # construct #num_net Siren
        self.net = []
        for i in range(self.num_net):
            siren_net = Siren(in_features, hidden_features, hidden_layers, out_features, outermost_linear,
                            first_omega_0, hidden_omega_0)
            # siren_net.load_state_dict(self.base_net.state_dict())
            if i > 0:
                siren_net.load_state_dict(self.net[0].state_dict())
            self.net.append(siren_net)
        self.net = nn.ModuleList(self.net)
        
    def mask_parameters(self):
        return [self.mask]
    

    def forward_single_chunk(self,j, input_queue, output_queue, start_event, end_event, stop_event, load_event):
        while not stop_event.is_set():
            print('waiting for start-{}'.format(j))
            start_event.wait()
            print('start processing-{}'.format(j))  
            try:  
                module_input_chunk = input_queue.get_nowait()
                i, module, input_chunk = module_input_chunk
                out = module(input_chunk)[0]
                print('putting output {}'.format(j))
                start_event.clear()
                print('waiting for end-{}'.format(j))
                while not load_event.is_set():
                    end_event.wait()
                    end_event.clear()

                print('end-{}'.format(j))
                    
            except Empty:
                print('{} input queue is empty'.format(j)) 


    
    def sequential_forward(self, coords):

        outputs = [None] * self.grid_size ** 2
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                network_idx = i * self.grid_size + j
                chunk = coords[:,network_idx]
                outputs[network_idx], _ = self.net[network_idx](chunk)
        tensor_grid = [[outputs[i * self.grid_size + j] for j in range(self.grid_size)] for i in range(self.grid_size)]
        vertical_stacks = [torch.cat(row, dim=2) for row in tensor_grid]
        output = torch.cat(vertical_stacks, dim=1)

        return output, coords

class GridSiren2(nn.Module):
    def __init__(self, args, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30., grid_size=4, num_net=1, coords_transform=False,
                 use_mask=False, add_ij=False):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_net = grid_size ** 2
        self.coords_transform = coords_transform
        self.use_mask = use_mask
        self.add_ij = add_ij

        # create a mask of size grid_size x grid_size x num_net
        if self.use_mask:
            self.num_net = num_net
            self.mask = nn.Parameter(5e-3 * torch.randn(grid_size, grid_size, self.num_net))  # [4, 4, 4]

        # transform coordinates in each grid to between -1 and 1
        # if add_ij, add i and j to the coordinates
        if coords_transform and add_ij:
            in_features = in_features + 2

        
        # construct #num_net Siren
        self.net = []
        # self.net.append(self.base_net)
        for i in range(self.num_net):
            siren_net = Siren(in_features, hidden_features, hidden_layers, out_features, outermost_linear,
                            first_omega_0, hidden_omega_0)
            if i > 0:
                siren_net.load_state_dict(self.net[0].state_dict())
            self.net.append(siren_net)
        self.net = nn.ModuleList(self.net)


    def get_weight_residule(self):
        net0 = self.net[0]
        net1 = self.net[1]

        # gather all the parameters from net0 and net1
        # and calculate the difference
        params0 = list(net0.parameters())
        params1 = list(net1.parameters())
        diffs = []
        for i in range(len(params0)):
            diff = params0[i] - params1[i]
            diffs.append(diff)

        return diffs, params0

        
    def mask_parameters(self):
        return [self.mask]
    
    def get_transformed_coords(self, coords, i, j, side_len=256, add_ij=True):
        n = coords.shape[0]
        coords[...,0] = (coords[...,0] + 1) * (side_len -1)/(n-1) - i * 2 * n/(n-1) - 1
        coords[...,1] = (coords[...,1] + 1) * (side_len -1)/(n-1) - j * 2 * n/(n-1) - 1

        if add_ij:
            i_grid = torch.ones_like(coords[...,0]) * i
            j_grid = torch.ones_like(coords[...,0]) * j
            # concat i_grid and j_grid to coords
            coords = torch.stack([coords[...,0], coords[...,1], i_grid, j_grid], dim=-1) # [64, 64, 4]

        return coords
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input  

        # partition the coords into #num_net grids
        # and feed each grid to a separate Siren
        # coords: [1, 65536, 2]
        coords = coords.view(256,256,2)
        chunk_size = 256 // self.grid_size  # 64
        grids_row = coords.split(chunk_size, dim=0) # [4, 64, 256, 2]
        grids = [r.split(chunk_size, dim=1) for r in grids_row] # [4, 4, 64, 64, 2]

        # apply softmax to the mask along the last dimension
        mask = mask_hard = torch.ones(self.grid_size, self.grid_size, self.num_net).cuda()
        if self.use_mask:
            mask = F.softmax(self.mask / 0.1, dim=-1) # [4, 4, 4]
            mask_hard = mask
            # apply gumbel softmax to the mask along the last dimension
            # mask = F.gumbel_softmax(self.mask, tau=0.1, hard=True, dim=-1) # [4, 4, 4]
            # get one-hot mask from the mask along the last dimension
            # mask_hard = F.one_hot(torch.argmax(mask, dim=-1), num_classes=self.num_net).float() # [4, 4, 4]
            # mask_hard + mask - mask.detach() # [4, 4, 4]
            # mask_hard = mask_hard + mask - mask.detach() # [4, 4, 4]

        # Process each chunk with the corresponding neural network
        outputs = []
        for i in range(self.grid_size):
            row_outputs = []
            for j in range(self.grid_size):
                chunk = grids[i][j] # [64, 64, 2]
                # transform chunk to between -1 and 1
                if self.coords_transform:
                    chunk = self.get_transformed_coords(chunk.clone().detach(), i, j, 256, self.add_ij)
                if self.use_mask:
                    # parallely feed the chunk to all networks and average the outputs with mask
                    mask_ij = mask_hard[i,j,:] # [4]
                    network_outputs = []
                    for k in range(self.num_net):
                        network_output, _ = self.net[k](chunk)
                        network_outputs.append(network_output * mask_ij[k])
                    network_output = torch.sum(torch.stack(network_outputs, dim=-1), dim=-1) # [64, 64, 1]
                else:
                    # one network for each chunk
                    network_idx = i * self.grid_size + j
                    network_output, _ = self.net[network_idx](chunk)   # [64, 64, 1]
                row_outputs.append(network_output)  
            # stack the row outputs together
            row_outputs = torch.cat(row_outputs, dim=1) # [64, 256, 1]
            outputs.append(row_outputs)
        outputs = torch.cat(outputs, dim=0) # [256, 256, 1]

        outputs = outputs.view(1,-1,1) # [1, 65536, 1]

        return outputs, coords, mask_hard, mask 

class PLSiren(pl.LightningModule):
    def __init__(self, args):
        super(PLSiren, self).__init__()
        # self.automatic_optimization = False
        self.args = args
        self.save_hyperparameters()
        self.model = GridSiren(args,in_features=self.args.num_dim, hidden_features=self.args.hidden_width, hidden_layers=self.args.num_hidden, out_features=3, grid_size=args.num_grid, coords_transform=False, use_mask=False, add_ij=False)
        # self.model = Siren(in_features=self.args.num_dim, hidden_features=self.args.hidden_width, hidden_layers=self.args.num_hidden, out_features=3)
        if self.args.resize_list == "-1":
            self.crop_h, self.crop_w = [int(x) for x in args.crop_list.split('_')[:2]]
        else:
            self.crop_h, self.crop_w = [int(x) for x in args.resize_list.split('_')]

    def training_step(self, batch, batch_idx):
        target = batch['imgs']
        coord = batch['coord']
        pred = self.model(coord)
        pred = pred[0] if len(pred) > 1 else pred
        pred = pred.reshape(target.shape[0], self.crop_h, self.crop_w, 3)
        pred = pred.permute(0,3,1,2)
    
        mse_loss = F.mse_loss(pred, target, reduction='none').mean()
        ssim_loss = ssim(pred, target, data_range=1, size_average=False).mean()
        loss = 0.7 * mse_loss + 0.3 * (1-ssim_loss)
        self.log('train_total_loss', loss, prog_bar=True)
        self.log('train_mse_loss', mse_loss, prog_bar=True)
        self.log('train_ssim_loss', ssim_loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        summary = ModelSummary(self)
        parameters_size = summary.total_parameters
        img = batch['imgs']
        img_size = np.prod(img.shape[-2:])*self.args.num_frame
        coord = batch['coord']
        pred = self.model(coord)
        pred = pred[0] if len(pred) > 1 else pred
        pred = pred.reshape(img.shape[0], self.crop_h, self.crop_w, 3)
        pred = pred.permute(0,3,1,2)
        mse = F.mse_loss(img, pred)
        psnr = -10 * torch.log10(mse)
        ssim_score = ssim(pred, img, data_range=1, size_average=False).mean()
        ppp = parameters_size / img_size
        self.log('val_psnr', psnr)
        self.log('val_ssim', ssim_score)
        self.log('val_ppp', ppp)
        return {'psnr':psnr,
                'ssim':ssim_score,
                'ppp':ppp}

    def validation_epoch_end(self, outputs):
        outputs = self.all_gather(outputs)

        avg_psnr = torch.stack([x['psnr'] for x in outputs]).mean()
        avg_ssim = torch.stack([x['ssim'] for x in outputs]).mean()
        avg_ppp = torch.stack([x['ppp'] for x in outputs]).mean()

        self.log('avg_val_psnr', avg_psnr, prog_bar=True, logger=True, sync_dist=True)
        self.log('avg_val_ssim', avg_ssim, prog_bar=True, logger=True, sync_dist=True)
        self.log('avg_val_ppp', avg_ppp, prog_bar=True, logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        # target = batch['imgs']
        coord = batch['coord']
        pred,_ = self.model(coord)
        pred = pred.reshape(self.crop_h, self.crop_w, 3)
        pred = pred.reshape(self.args.batch_size, self.crop_h, self.crop_w, 3)
        pred = pred[0]
        
        pred = (pred-pred.min())/(pred.max()-pred.min())*255
        pred = pred.detach().cpu().numpy()
        pred = pred.astype(np.uint8)
        import cv2
        cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        cv2.imwrite('results/pred.png', pred)
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=0)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 600])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
        return [optimizer], [scheduler]
        # return optimizer