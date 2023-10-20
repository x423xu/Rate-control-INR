import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
from copy import deepcopy
from dahuffman import HuffmanCodec

import time

seed = 43
torch.manual_seed(seed)
np.random.seed(seed)
# Set a random seed
# torch.manual_seed(2023)

def get_mgrid(sidelen, dim=2, max=1.0):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-max, max, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

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


class PosProject(nn.Module):
    def __init__(self, in_features, step_size, N=2) -> None:
        super().__init__()
        self.in_features = in_features
        self.step_size = step_size
        self.N = N

    def forward(self, x):
        # x: [1, 65536, 2]
        x_project = []
        # construct range between -N -N+1 to N-1, N

        for i in range(-self.N, self.N+1):
            for j in range(-self.N, self.N+1):
                offset = torch.Tensor([i*self.step_size, j*self.step_size]).cuda() # [2]
                x_project.append(x + offset)
        
        x_project = torch.cat(x_project, dim=-1) # [1, 65536, 18]

        return x_project

    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30., pos_project=False, side_len=256, N=2):
        super().__init__()

        self.pos_project = pos_project
        if pos_project:
            self.pos_project_net = PosProject(in_features=2, step_size=2.0/side_len, N=N)
            in_features = in_features * ((2*N+1)**2)

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))
        
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

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
        if self.pos_project:
            coords_project = self.pos_project_net(coords)
            output = self.net(coords_project)
        else:
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


class MultiSiren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30., num_net=10):
        super().__init__()
        
        self.num_net = num_net
        
        # construct mask_net as a classification MLP
        self.mask_net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            # nn.Linear(hidden_features, hidden_features),
            # nn.ReLU(),
            nn.Linear(hidden_features, num_net),
            # nn.Softmax(dim=-1)
        )

        # construct #num_net Siren
        self.net = []
        for i in range(num_net):
            siren_net = Siren(in_features, hidden_features, hidden_layers, out_features, outermost_linear, 
                            first_omega_0, hidden_omega_0)
            self.net.append(siren_net)
        self.net = nn.ModuleList(self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input  

        mask = self.mask_net(coords) # [1, 65536, 10]
        # implement softmax with temperature scaling
        mask = mask / 0.1
        mask = F.softmax(mask, dim=-1)

        # gumbel softmax
        # mask = F.gumbel_softmax(mask, tau=0.1, hard=True, dim=-1)


        output = []
        for i in range(self.num_net):
            tmp, _ = self.net[i](coords)
            tmp = tmp * mask[:,:,i:i+1]
            output.append(tmp)
            # output += tmp * mask[:,:,i]
        output = torch.stack(output, dim=-1) # [1, 65536, 10]
        # output = torch.sum(output * mask, dim=-1) # [1, 65536]
        output = torch.sum(output, dim=-1) # [1, 65536]

        return output, coords, mask      

class GridSiren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30., grid_size=4, num_net=10, coords_transform=False,
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

        # construct a base Siren from which to copy weights
        # self.base_net = Siren(in_features, hidden_features, hidden_layers, out_features, outermost_linear, 
        #                     first_omega_0, hidden_omega_0)
        
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

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def get_cameraman_tensor(sidelength):
    # img = Image.fromarray(skimage.data.camera())  
    img = Image.open('../../data/UVG/ShakeNDry/frame_0.png')
    img = img.convert('L')   
    transform = Compose([
        Resize([sidelength, sidelength]),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img


def evaluate(model, input, gt, model_bit=8, huffman_coding=True, quant_res=False, quant_together=False):
    # quantize the model to model_bit
    model_list, quant_ckt = quant_model(model, model_bit, quant_res, quant_together)

    # evaluate the model
    for i, model in enumerate(model_list):
        model.eval()
        model.cuda()
        with torch.no_grad():
            model_output, _, _, _ = model(input)
            loss = ((model_output - gt)**2).mean()
            psnr = -10*torch.log10(loss)
            print("Model %d, Total loss %0.6f, PSNR %0.6f" % (i, loss, psnr))

    # get num of parameters
    num_param = (sum([p.data.nelement() for p in model_list[0].parameters()]) / 1e6) 
    print("Number of parameters %0.6f M" % num_param)

    # huffman coding
    quant_v_list = []
    tmin_scale_len = 0
    if huffman_coding:
        for k, layer_wt in quant_ckt.items():
            quant_v_list.extend(layer_wt['quant'].flatten().tolist())
            if tmin_scale_len == 0 or not quant_together:
                # if quant_together, only count once tmin and scale
                tmin_scale_len += layer_wt['min'].nelement() + layer_wt['scale'].nelement()

        # get the element name and its frequency
        unique, counts = np.unique(quant_v_list, return_counts=True)
        num_freq = dict(zip(unique, counts))

        # generating HuffmanCoding table
        codec = HuffmanCodec.from_data(quant_v_list)
        sym_bit_dict = {}
        for k, v in codec.get_code_table().items():
            sym_bit_dict[k] = v[0]

        # total bits for quantized embed + model weights
        total_bits = 0
        for num, freq in num_freq.items():
            total_bits += freq * sym_bit_dict[num]
        bits_per_param = total_bits / len(quant_v_list) # bits per parameter
        
        # including the overhead for min and scale storage, 
        total_bits += tmin_scale_len * 16               #(16bits for float16)
        full_bits_per_param = total_bits / len(quant_v_list) # bits per parameter including the overhead for min and scale storage

        # bits per pixel
        pixels_per_frame = 256 * 256
        num_frame = 1
        total_bpp = total_bits / pixels_per_frame / num_frame # bits per pixel

        print("Bits per param %0.6f, full bits per param %0.6f, bpp %0.6f" % (bits_per_param, full_bits_per_param, total_bpp))




def quant_model(model, quant_model_bit=8, quant_res=True, quant_together=False):
    model_list = [deepcopy(model)]
    if quant_model_bit == -1:
        return model_list, None
    else:
        cur_model = deepcopy(model)
        quant_ckt, cur_ckt = [cur_model.state_dict() for _ in range(2)]

        min_v_all = min([v.min() for v in cur_ckt.values()]) if quant_together else None
        max_v_all = max([v.max() for v in cur_ckt.values()]) if quant_together else None

        if quant_res:
            # modify cur_ckt to store residual except for the first network
            for k,v in cur_ckt.items():
                if k.startswith('net.0'):
                    # keep the first network
                    continue
                # store remaining networks as residual
                # substitute the second element in k with 0 (splitted by '.')
                base_k = k.split('.')
                base_k[1] = '0'
                base_k = '.'.join(base_k)
                residual = v - cur_ckt[base_k]
                cur_ckt[k] = residual

        for k,v in cur_ckt.items():
            quant_v, new_v = quant_tensor(v, quant_model_bit, min_v_all, max_v_all)
            quant_ckt[k] = quant_v
            cur_ckt[k] = new_v
            if not k.startswith('net.0') and quant_res:
                # add residual back to the quantized weight
                base_k = k.split('.')
                base_k[1] = '0'
                base_k = '.'.join(base_k)
                cur_ckt[k] += cur_ckt[base_k]

        cur_model.load_state_dict(cur_ckt)
        model_list.append(cur_model)
        
        return model_list, quant_ckt
    

def quant_tensor(t, bits=8, t_min=None, t_max=None):
    
    if t_min is not None and t_max is not None:
        # use given t_min and t_max for quantization
        scale = (t_max - t_min) / (2**bits-1)

        t_min, scale = t_min.expand_as(t), scale.expand_as(t)
        quant_t = ((t - t_min) / (scale)).round().clamp(0, 2**bits-1)
        quant_t = quant_t.to(torch.uint8)
        new_t = t_min + scale * quant_t

        quant_t = {'quant': quant_t, 'min': t_min, 'scale': scale}

        return quant_t, new_t
    
    tmin_scale_list = []
    # quantize over the whole tensor, or along each dimenstion
    if t.dim() == 1:
        best_quant_t = torch.Tensor([0.])
        best_tmin = t
        best_scale = torch.Tensor([0.])

        best_quant_t = best_quant_t.to(torch.uint8)
        best_tmin = best_tmin.to(torch.float16) 
        best_scale = best_scale.to(torch.float16)

        best_new_t = t
    else:
        t_min, t_max = t.min(), t.max()
        scale = (t_max - t_min) / (2**bits-1)
        tmin_scale_list.append([t_min, scale])
        for axis in range(t.dim()):
            t_min, t_max = t.min(axis, keepdim=True)[0], t.max(axis, keepdim=True)[0]
            if t_min.nelement() / t.nelement() < 0.02:
                scale = (t_max - t_min) / (2**bits-1)
                # tmin_scale_list.append([t_min, scale]) 
                tmin_scale_list.append([t_min.to(torch.float16), scale.to(torch.float16)]) 
        # import pdb; pdb.set_trace; from IPython import embed; embed() 
        
        quant_t_list, new_t_list, err_t_list = [], [], []
        for t_min, scale in tmin_scale_list:
            t_min, scale = t_min.expand_as(t), scale.expand_as(t)
            quant_t = ((t - t_min) / (scale)).round().clamp(0, 2**bits-1)
            new_t = t_min + scale * quant_t
            err_t = (t - new_t).abs().mean()
            quant_t_list.append(quant_t)
            new_t_list.append(new_t)
            err_t_list.append(err_t)   

        # choose the best quantization 
        best_err_t = min(err_t_list)
        best_quant_idx = err_t_list.index(best_err_t)
        best_new_t = new_t_list[best_quant_idx]
        best_quant_t = quant_t_list[best_quant_idx].to(torch.uint8)
        best_tmin = tmin_scale_list[best_quant_idx][0]
        best_scale = tmin_scale_list[best_quant_idx][1]
    quant_t = {'quant': best_quant_t, 'min': best_tmin, 'scale': best_scale}

    return quant_t, best_new_t             


class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img = get_cameraman_tensor(sidelength)
        self.pixels = img.permute(1, 2, 0).view(-1, 1) # [65536, 1]
        self.coords = get_mgrid(sidelength, 2) 

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels
    
cameraman = ImageFitting(256)
dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)

# img_siren = Siren(in_features=2, out_features=1, hidden_features=256, 
#                   hidden_layers=3, outermost_linear=True, pos_project=False)

img_siren = GridSiren(in_features=2, out_features=1, hidden_features=64, 
                  hidden_layers=3, outermost_linear=True, grid_size=1, num_net=1, 
                  coords_transform=True, use_mask=False, add_ij=True)

img_siren.cuda()

total_steps = 3001 # Since the whole image is our dataset, this just means 500 gradient descent steps.
steps_til_summary = 50
steps_til_plot = 3000

optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

model_input, ground_truth = next(iter(dataloader))  # [1, 65536, 2], [1, 65536, 1]
model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

mask_train = torch.ones_like(ground_truth) # [1, 65536, 1]
mask_train[:,::5] = 0
mask_valid = 1 - mask_train
masks = []

for step in range(total_steps):
    # model_output, coords, mask = img_siren(model_input)    
    model_output, coords, mask_hard, mask = img_siren(model_input)

    loss = ((model_output - ground_truth)**2).mean()
    psnr = -10*torch.log10(loss)

    masks.append(mask.detach().cpu().numpy())
    # masks_hard.append(mask_hard[0,0].detach().cpu().numpy())

    
    if not step % steps_til_summary:
        # weighted average loss1 and loss2
        # loss = loss1 * 0.8 + loss2 * 0.2
        print("Step %d, Total loss %0.6f, PSNR %0.6f, wr_avg %0.6f, wo_avg %0.6f" % (step, loss, psnr, 0, 0))
        
    if not step % steps_til_plot:
        mask_max, mask_arg_max = torch.max(mask, dim=-1)   # [1, 65536]
        # normalize the mask to be 0-1
        # mask_arg_max = mask_arg_max.float() / 9.
        # mask_arg_max = mask_arg_max.view(256,256).detach().cpu().numpy()

        fig, axes = plt.subplots(2,3, figsize=(18,12))
        axes[0,0].imshow(model_output.cpu().view(256,256).detach().numpy())
        axes[0,1].imshow(mask_arg_max.detach().cpu().numpy())
        axes[0,2].imshow(mask_max.detach().cpu().numpy())
        # plot the softmax distribution of the first grid in the mask along the last dimension
        masks_array = np.array(masks)
        # masks_hard_array = np.array(masks_hard)
        num_net = masks_array.shape[-1]
        for i in range(num_net):
            axes[1,0].plot(masks_array[:,0,0,i], label=str(i))
            # axes[1,1].plot(masks_array[:,0,1,i], label=str(i))
            # axes[1,2].plot(masks_array[:,0,2,i], label=str(i))
        axes[1,0].legend()
        # axes[1,1].legend()
        # axes[1,2].legend()

        # get argmax of masks_hard 
        # masks_hard_argmax = np.argmax(masks_hard_array, axis=-1)
        # axes[1,1].plot(masks_hard_argmax)
        plt.show()

    optim.zero_grad()
    loss.backward()
    optim.step()

# evaluate the model and get the bits per pixel
evaluate(img_siren, model_input, ground_truth, model_bit=8, huffman_coding=True, quant_res=False, quant_together=False)
