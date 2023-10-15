import datetime
import os
import sys
os.environ["WANDB_API_KEY"] = "ddb1831ecbd2bf95c3323502ae17df6e1df44ec0"
import warnings

warnings.filterwarnings("ignore")
import math
import argparse
import shutil
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
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
    parser.add_argument('--ex_name', default='debug', type=str)
    parser.add_argument('--dataset', default='CLS') 
    parser.add_argument('--model_name', default='CLS')
    
    # dataset parameters
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    
    # Training parameters
    parser.add_argument('--epoch', default=10, type=int, help='end epoch')
    parser.add_argument('--check_val_every_n_epoch', default=1, type=int)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--lr_scheduler', default='onecycle')
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--offline', default=1, type=int) # 如果offline=1,不会上传到wandb; 否则结果会同步到wandb
    parser.add_argument('--seed', default=111, type=int)
    
    
    args = parser.parse_args()
    return args




def load_callbacks(args):
    callbacks = []
    
    logdir = str(os.path.join(args.res_dir, args.ex_name))
    
    ckptdir = os.path.join(logdir, "checkpoints")
    
    callbacks.append(BackupCodeCallback('./',logdir))
    

    metric = "val_acc"
    sv_filename = 'best-{epoch:02d}-{val_seq_loss:.3f}'
    callbacks.append(plc.ModelCheckpoint(
        monitor=metric,
        filename=sv_filename,
        save_top_k=15,
        mode='max',
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
    print(f"steps_per_epoch {args.steps_per_epoch},  gpu_count {gpu_count}, batch_size{args.batch_size}")
    
    model = MInterface(**vars(args))

    
    trainer_config = {
        'gpus': -1,  # Use the all GPUs
        'max_epochs': args.epoch,  # Maximum number of epochs to train for
        'num_nodes': 1,  # Number of nodes to use for distributed training
        "strategy": 'ddp', # 'ddp', 'deepspeed_stage_2
        "precision": 'bf16', # "bf16", 16
        # 'auto_scale_batch_size': 'binsearch',
        'accelerator': 'gpu',  # Use distributed data parallel
        'callbacks': load_callbacks(args),
        'logger': plog.WandbLogger(
                    project = 'VQProtein',
                    name=args.ex_name,
                    save_dir=str(os.path.join(args.res_dir, args.ex_name)),
                    offline = args.offline,
                    id = args.ex_name,
                    entity = "gaozhangyang"),
        'gradient_clip_val':1.0
    }

    trainer_opt = argparse.Namespace(**trainer_config)
    trainer = Trainer.from_argparse_args(trainer_opt)
    
    trainer.fit(model, data_module)
    
    print(trainer_config)
