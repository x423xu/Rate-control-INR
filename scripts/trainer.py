from config import args
from models.model import Siren, GridSiren,GridSiren2
from data.data import UVGDataModule

import torch
from pytorch_msssim import ssim
import time
from tqdm import tqdm
from glob import glob
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from dahuffman import HuffmanCodec
from copy import deepcopy
from PIL import Image
import datetime
from torch.multiprocessing import Process, Queue, Event, Manager, set_start_method
from queue import Empty 
set_start_method('spawn', force=True)

timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
seed = 43
torch.manual_seed(seed)
np.random.seed(seed)

WANDB = False

if WANDB:
    import wandb
    name = timestr+'hw{}:nh{}:nd{}:no{}'.format(args.hidden_width, args.num_hidden, args.num_dim, args.out_channels)
    if args.resize_list != '-1':
        name += ':s{}'.format(args.resize_list)
    else:
        name += ':s{}'.format(args.crop_list)
    name += ':g{}'.format(args.num_grid)
    wandb.init(project='Rate-control-INR', entity='xxy', name = name)

def train():
    epochs = 3000

    # model = Siren(in_features=args.num_dim, hidden_features=args.hidden_width, hidden_layers=args.num_hidden, out_features=args.out_channels)
    model = GridSiren(args,in_features=args.num_dim, hidden_features=args.hidden_width, 
                       hidden_layers=args.num_hidden, out_features=args.out_channels, 
                       grid_size=args.num_grid, coords_transform=args.coords_transform, use_mask=False, 
                       add_ij=args.add_ij, outermost_linear=True,num_net=1)
    total_param_size = sum(p.numel() for p in model.parameters())
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3000, 6000], gamma=0.1)
    # args.batch_size = 1
    loader = UVGDataModule(args)
    loader.setup(stage='fit')
    train_loader = loader.train_dataloader()
    imgs = []
    coords = []
    for batch in train_loader:
        imgs.append(batch['imgs'])
        coords.append(batch['coord'])
    imgs = torch.stack(imgs).cuda()
    coords = torch.stack(coords).cuda()


    # nn = 2
    # imgs = imgs[:nn]
    # coords = coords[:nn]
    #load model
    # best_model_path = glob('model_*.pth')
    # if best_model_path:
    #     model.load_state_dict(torch.load(best_model_path[0]))
    #     os.system('rm model_*.pth')

    
    if args.resize_list == "-1":
        crop_h, crop_w = [int(x) for x in args.crop_list.split('_')[:2]]
    else:
        crop_h, crop_w = [int(x) for x in args.resize_list.split('_')]
    
    

    # parallel training
    if args.parallel:
        processes = []
        input_queue = Queue()
        output_queue = Queue()
        start_event = []
        end_event = []
        stop_event = Event()
        load_event = Event()
        for i in range(args.num_grid**2):
            end_event.append(Event())
            start_event.append(Event())
            p = Process(target=model.forward_single_chunk, args=(i, input_queue, output_queue, start_event[i], end_event[i], stop_event, load_event))
            processes.append(p)
        try:
            for p in processes:
                p.start()
            for coord, target in zip(coords, imgs):
                # while not all(([not ee.is_set() for ee in end_event])):
                #     pass
                pass

               
                
                
                
            stop_event.set()
        except (KeyboardInterrupt, Exception):
            print('exception')
            for p in processes:
                if p.is_alive():
                    p.terminate()
    else:
        start_time = time.time()
        max_psnr = 0
        val_every = 100
        progress_bar = tqdm(total = epochs*len(imgs))
        model.train()
        for epoch in range(epochs):
            losses = []
            psnrs = []
            ssims = []    
            for coord, target in zip(coords, imgs):
                progress_bar.set_description('Epoch %i' % epoch)
                progress_bar.update()
                optimizer.zero_grad()
                
                pred = model.sequential_forward(coord)
                pred = pred[0] if len(pred) > 1 else pred
                pred = pred.reshape(target.shape[0], crop_h, crop_w, args.out_channels)
                # pred = pred.permute(0,3,1,2)

                mse_loss = (pred-target).pow(2).mean()
                ssim_loss = ssim(pred, target, data_range=1, size_average=False).mean()
                # loss = 0.7 * mse_loss + 0.3 * (1-ssim_loss)
                loss = mse_loss
                psnr = 10 * torch.log10(1 / mse_loss)
                loss.backward()
                optimizer.step()
                # scheduler.step()
                dt = time.time() - start_time
                losses.append(loss.item())
                psnrs.append(psnr.item())
                ssims.append(ssim_loss.item())
            if np.mean(psnrs) > max_psnr:
                max_psnr = np.mean(psnrs)
                best_model_path = glob('model_*.pth')
                if best_model_path:
                    ckpt_psnr = float(best_model_path[0].split('_')[-1].split('.')[0])
                    if max_psnr>ckpt_psnr:
                        os.system('rm model_*.pth')
                        torch.save(model.state_dict(), 'model_{:.2f}.pth'.format(max_psnr))
                else:   
                    torch.save(model.state_dict(), 'model_{:.2f}.pth'.format(max_psnr))
            if epoch % val_every == 0:
                model.eval()
                if WANDB:
                    wandb.log({'loss': np.mean(losses), 'psnr': np.mean(psnrs), 'ssim': np.mean(ssims), 'max_psnr': max_psnr, 'model_size': total_param_size*1e-6})
                print('time {:.4f}s--epoch: {}, loss: {:.4f}, psnr: {:.4f}, ssim: {:.4f}, max_psnr: {:.4f}, model_size: {:.4f}M'.format(dt, epoch, np.mean(losses), np.mean(psnrs), np.mean(ssims), max_psnr, total_param_size*1e-6))
                evaluate(model, coords, imgs, model_bit=8, huffman_coding=True, quant_res=False, quant_together=False)
                log_image(model, coords, imgs)
                model.train()
def log_image(model, coords, imgs):
    if WANDB:
        with torch.no_grad():
            model_output = model.sequential_forward(coords[0])
            model_output = model_output[0] if len(model_output) > 1 else model_output
            if len(model_output.shape) == 3:
                model_output = model_output.reshape(imgs[0].shape[0], imgs[0].shape[-2], imgs[0].shape[-1], imgs[0].shape[1])
            model_output = model_output.permute(0,3,1,2)
            model_output = model_output.cpu().numpy()
            model_output = model_output[0]
            model_output = model_output.transpose(1,2,0)
            if model_output.shape[-1] == 1:
                model_output = model_output.squeeze(-1)
            model_output = (model_output-model_output.min())/(model_output.max()-model_output.min())
            model_output = (model_output * 255).astype(np.uint8)
            model_output = Image.fromarray(model_output)
            model_output = model_output.resize((256,256))
            wandb.log({'model_output': wandb.Image(model_output)})

            # target = imgs[0,0].cpu().numpy()
            # target = Image.fromarray((target * 255).astype(np.uint8))   
            # wandb.log({'target': wandb.Image(target)})

def evaluate(model, input, gt, model_bit=8, huffman_coding=True, quant_res=True, quant_together=False):
    # quantize the model to model_bit
    model_list, quant_ckt = quant_model(model, model_bit, quant_res, quant_together)

    # evaluate the model
    for i, model in enumerate(model_list):
        # model.eval()
        model.cuda()
        with torch.no_grad():
            avg_loss = 0
            for coord, target in zip(input, gt):
                model_output = model.sequential_forward(coord)
                model_output = model_output[0] if len(model_output) > 1 else model_output
                if len(model_output.shape) == 3:
                    model_output = model_output.reshape(target.shape[0], target.shape[2], target.shape[3], target.shape[1])
                # model_output = model_output.permute(0,3,1,2)
                loss = ((model_output - target)**2).mean()
                avg_loss += loss
            loss = avg_loss / input.shape[0]
            psnr = -10*torch.log10(loss)
            if WANDB:
                wandb.log({'val_psnr_model_{}'.format(i): psnr.item()})
                wandb.log({'val_loss_model_{}'.format(i): loss.item()})
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
        pixels_per_frame = gt.shape[-2]*gt.shape[-1]
        num_frame = gt.shape[0]
        total_bpp = total_bits / pixels_per_frame / num_frame # bits per pixel
        if WANDB:
            wandb.log({'bpp': total_bpp})
            wandb.log({'bpp_per_param': bits_per_param})
            wandb.log({'bpp_per_param_with_overhead': full_bits_per_param})
        print("Bits per param %0.6f, full bits per param %0.6f, bpp %0.6f\n" % (bits_per_param, full_bits_per_param, total_bpp))
    # model.train()




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

if __name__ == '__main__':
    train()