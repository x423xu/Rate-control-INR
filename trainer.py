from config import args
from model import Siren
from data import UVGDataModule
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
import time

def train():
    model = Siren(in_features=args.num_dim, hidden_features=256, hidden_layers=3, out_features=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
    loader = UVGDataModule(args)
    loader.setup(stage='fit')
    train_loader = loader.train_dataloader()
    data_iter = iter(train_loader)
    batch = next(data_iter)
    target = batch['imgs'][0,0]
    target = target.unsqueeze(0)
    coord = batch['coord'][0]

    epochs = 6000
    crop_h, crop_w = [int(x) for x in args.crop_list.split('_')[:2]]
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred,_ = model(coord)
        pred = pred.reshape(crop_h, crop_w, 3)
        pred = pred.permute(2,0,1).unsqueeze(0)

        mse_loss = F.mse_loss(pred, target, reduction='none').flatten(1).mean(1) 
        ssim_loss = ssim(pred, target, data_range=1, size_average=False)
        loss = 0.7 * mse_loss + 0.3 * (1-ssim_loss)
        psnr = 10 * torch.log10(1 / mse_loss)
        loss.backward()
        optimizer.step()
        scheduler.step()
        dt = time.time() - start_time
        # if epoch % 100 == 0:
        print('time {:.4f}s--epoch: {}, loss: {}, psnr: {}, ssim: {}'.format(dt, epoch, loss.item(), psnr.item(), ssim_loss.item()))


if __name__ == '__main__':
    train()