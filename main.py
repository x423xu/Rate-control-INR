'''
Main file for the project Rate-control-INR
'''

import pytorch_lightning as pl
from config import args
from model import PLSiren
from data import UVGDataModule
from datetime import datetime

# Get the current time
current_time = datetime.now()
# Format the time as a string
time_str = current_time.strftime("%Y%m%d_%H:%M:%S")

def main():
    '''
    train
    '''
    logger = pl.loggers.CSVLogger(save_dir = '.', name = 'log', version = time_str)
    model = PLSiren(args)
    loader = UVGDataModule(args)
    kwargs = {
        'accelerator': 'gpu',
        'num_sanity_val_steps': 2,
        'max_epochs': 30,
        'check_val_every_n_epoch': 50,
        'devices': 2,
        'enable_progress_bar': False,
        'logger': logger,
        'profiler': pl.profilers.AdvancedProfiler(dirpath="log/"+time_str, filename = 'perf_log'),
    }
    trainer = pl.Trainer(**kwargs)
    trainer.fit(model, loader)

    '''
    test
    '''
    # model = PLSiren(args)
    # loader = UVGDataModule(args)
    # ckpt_path = '/home/xxy/Documents/code/Rate-control-INR/log/20230910_21:34:31/checkpoints/epoch=2999-step=3000.ckpt'
    # model = model.load_from_checkpoint(ckpt_path)
    # trainer = pl.Trainer(accelerator='gpu', devices = 1)
    # trainer.test(model, loader)
if __name__ == '__main__':
    main()
    