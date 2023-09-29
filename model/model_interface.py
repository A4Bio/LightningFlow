import inspect
import torch
import importlib
import torch.optim.lr_scheduler as lrs
import pytorch_lightning as pl
import os
import torch.nn as nn
from utils.utils import get_text_logger


class MInterface(pl.LightningModule):
    def __init__(self, model_name=None, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.model_module = self.init_model_module(model_name)
        self.model = self.instancialize_module(self.model_module)
        self.configure_loss()
        os.makedirs(os.path.join(self.hparams.res_dir, self.hparams.ex_name), exist_ok=True)
        self.text_logger = get_text_logger(self.hparams.res_dir, self.hparams.ex_name) # 你能使用text_logger在本地的.log文件中记录任何信息
        

    def forward(self, batch):
        results = self.model(batch['data'])
        return results
    

    def training_step(self, batch, batch_idx, **kwargs):
        results = self(batch)
        output = results['output']
        loss = self.loss_function(output, batch['label'])
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        results = self(batch)
        output = results['output']
        
        acc = (output.argmax(dim=-1) == batch['label']).float().mean()

        self.log_dict({"val_acc": acc})
        return self.log_dict

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        # Make the Progress Bar leave there
        self.print('')
    
    def get_schedular(self, optimizer, lr_scheduler='onecycle'):
        if lr_scheduler == 'step':
            scheduler = lrs.StepLR(optimizer,
                                    step_size=self.hparams.lr_decay_steps,
                                    gamma=self.hparams.lr_decay_rate)
        elif lr_scheduler == 'cosine':
            scheduler = lrs.CosineAnnealingLR(optimizer,
                                                T_max=self.hparams.lr_decay_steps,
                                                eta_min=self.hparams.lr_decay_min_lr)
        elif lr_scheduler == 'onecycle':
            scheduler = lrs.OneCycleLR(optimizer, max_lr=self.hparams.lr, steps_per_epoch=self.hparams.steps_per_epoch, epochs=self.hparams.epoch, three_phase=False)
        else:
            raise ValueError('Invalid lr_scheduler type!')

        return scheduler

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=weight_decay)

        schecular = self.get_schedular(optimizer, self.hparams.lr_scheduler)

        return [optimizer], [{"scheduler": schecular, "interval": "step"}]
        
    def lr_scheduler_step(self, *args, **kwargs):
        scheduler = self.lr_schedulers()
        scheduler.step()

    def configure_loss(self):
        self.loss_function = nn.NLLLoss()

    def init_model_module(self, name):
        Model = getattr(importlib.import_module(f'.{name}_model', package=__package__), f"{name}_Model")
        return Model
    
    def instancialize_module(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams, arg)
        args1.update(other_args)
        return Model(**args1)