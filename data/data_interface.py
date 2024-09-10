import inspect
import importlib
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from utils.utils import cuda

class MyDataLoader(DataLoader):
    '''
    在这里你可以实现: 
        1. 对数据进行在线预处理,比如使用预训练模型提取特征, 然后将预训练特征作为data的一部分存在batch里面。通过设置self.memory存储预训练特征, 从第二个epoch开始直接从memory取特征,而无需再次运行预训练模型。
        2. 将cpu的data预先放进gpu, 进一步提升训练速度
    '''
    def __init__(self, batch_size=64, num_workers=8, device=0, *args, **kwargs):
        super().__init__(batch_size=batch_size, num_workers=num_workers, *args, **kwargs)
        self.device = device
        self.stream = torch.cuda.Stream(
            self.device
        )  # create a new cuda stream in each process
        
        self.memory = {}
    
    def __iter__(self):
        for batch in super().__iter__():
            # 在这里对batch进行处理
            # 提前将变量塞进GPU
            with torch.cuda.stream(self.stream):
                batch = cuda(batch, device=self.device, non_blocking=True)
                
            yield batch

def collate_fn(examples):
    # print(examples)
    return {
        'seq': torch.stack([torch.tensor(example['seq']) for example in examples]),
        'vqid': torch.stack([torch.tensor(example['vqid']) for example in examples]),
    }

class DInterface(pl.LightningDataModule):
    def __init__(self, num_workers=8,
                 dataset='PDB',
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.num_workers = num_workers
        self.dataset = dataset
        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']
        self.data_module = self.init_data_module(dataset)
        self.device = 'cuda:0'

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.trainset = self.instancialize_module(module = self.data_module, split = 'train')
            self.valset = self.instancialize_module(module = self.data_module, split='valid')

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.testset = self.instancialize_module(module = self.data_module, split='test')

    def train_dataloader(self):
        train_loader = DataLoader(
            self.trainset,
            batch_size= self.hparams.batch_size,
            pin_memory=True,
            shuffle=True,
            collate_fn=collate_fn)
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valset,
            batch_size= self.hparams.batch_size,
            pin_memory=True,
            shuffle=False,
            collate_fn=collate_fn)
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.testset,
            batch_size= self.hparams.batch_size,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn)
        return test_loader
    
    def instancialize_module(self, module, **other_args):
        class_args =  list(inspect.signature(module.__init__).parameters)[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return module(**args1)

    def init_data_module(self, name, **other_args):
        self.data_module = getattr(importlib.import_module(f'.{name}_dataset', package='data'), f'{name}_Dataset')
        return self.data_module
    
    