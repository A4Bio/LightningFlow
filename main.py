import datetime
import os
import sys
os.environ["WANDB_API_KEY"] = "97202a52488fcf2762c99ff8c68367f9bc5d4033"
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import math
import argparse
import shutil
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer import Trainer
import pytorch_lightning.callbacks as plc
import pytorch_lightning.loggers as plog
from model import MInterface
from data import DInterface
# from utils.utils import load_model_path_by_args # 返回最优chckpoint的路径
from utils.logger import SetupCallback, BackupCodeCallback

def create_parser():
    parser = argparse.ArgumentParser()
    # Set-up parameters
    parser.add_argument('--res_dir', default='./results', type=str)
    parser.add_argument('--ex_name', default='debug1', type=str)
    parser.add_argument('--dataset', default='PDB', type=str)
    parser.add_argument('--model_name', default='SEDD', type=str)
    
    # dataset parameters
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=1, type=int)
    
    # model parameters
    parser.add_argument('--graph_type', default='absorb', type=str) # absorb uniform
    parser.add_argument('--noise', default='loglinear', type=str) # loglinear geometric
    parser.add_argument('--tokens', default=258, type=int)
    parser.add_argument('--block_size', default=512, type=int)

    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--cond_dim', default=128, type=int)
    parser.add_argument('--n_heads', default=12, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--n_blocks', default=12, type=int)
    parser.add_argument('--scale_by_sigma', default=True, type=bool)

    # Training parameters
    parser.add_argument('--epoch', default=100, type=int, help='end epoch')
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--lr_scheduler', default='cosine')
    parser.add_argument('--lr_decay_steps', default=4000, type=int)
    parser.add_argument('--lr_decay_min_lr', default=0.8, type=float)
    # parser.add_argument('--warmup_steps', default=2500, type=int)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate')
    parser.add_argument('--offline', default=0, type=int) # 如果offline=1,不会上传到wandb; 否则结果会同步到wandb
    parser.add_argument('--seed', default=111, type=int)
    
    args = parser.parse_args()
    return args



def load_callbacks(args):
    callbacks = []
    
    logdir = str(os.path.join(args.res_dir, args.ex_name))
    
    ckptdir = os.path.join(logdir, "checkpoints")
    
    callbacks.append(BackupCodeCallback('./',logdir))
    

    metric = "val_loss"
    sv_filename = 'best-{epoch:02d}-{val_loss:.3f}'
    callbacks.append(plc.ModelCheckpoint(
        monitor=metric,
        filename=sv_filename,
        save_top_k=15,
        mode='min',
        save_last=True,
        dirpath = ckptdir,
        verbose = True,
        every_n_epochs = args.check_val_every_n_epoch,
    ))

    
    now = datetime.datetime.now().strftime("%m-%dT%H-%M-%S")
    cfgdir = os.path.join(logdir, "configs")
    callbacks.append(
        SetupCallback(
                now = now,
                logdir = logdir,
                ckptdir = ckptdir,
                cfgdir = cfgdir,
                config = args.__dict__,
                argv_content = sys.argv + ["gpus: {}".format(torch.cuda.device_count())],)
    )
    
    
    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval=None))
    return callbacks



if __name__ == "__main__":
    args = create_parser()
    pl.seed_everything(args.seed)
    
    data_module = DInterface(**vars(args))
    data_module.setup()
    
    gpu_count = torch.cuda.device_count()
    args.steps_per_epoch = math.ceil(len(data_module.trainset)/args.batch_size/gpu_count)
    print(f"steps_per_epoch {args.steps_per_epoch},  gpu_count {gpu_count}, batch_size {args.batch_size}")
    
    model = MInterface(**vars(args))
    
    trainer_config = {
        'devices': -1,  # Use all available GPUs
        # 'precision': 'bf16',  # Use 32-bit floating point precision
        'precision': '32',
        'max_epochs': args.epoch,  # Maximum number of epochs to train for
        'num_nodes': 1,  # Number of nodes to use for distributed training
        "strategy": 'ddp',
        "accumulate_grad_batches": 1,
        'accelerator': 'cuda',  
        'callbacks': load_callbacks(args),
        'logger': [
                    plog.WandbLogger(
                    project = 'TokenDiff',
                    name=args.ex_name,
                    save_dir=str(os.path.join(args.res_dir, args.ex_name)),
                    offline = args.offline,
                    id = args.ex_name.replace('/', '-',5),
                    entity = "gmondy",
                    ),
                   plog.CSVLogger(args.res_dir, name=args.ex_name)],
         'gradient_clip_val': 0.5
    }

    trainer = Trainer(**trainer_config)
    
    trainer.fit(model, data_module)
    
    print(trainer_config)
