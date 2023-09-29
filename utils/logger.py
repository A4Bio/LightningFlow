import os
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback
import shutil

class SetupCallback(Callback):
    def __init__(self,  now, logdir, ckptdir, cfgdir, config, argv_content=None):
        super().__init__()
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
    
        self.argv_content = argv_content

    # 在pretrain例程开始时调用。
    def on_fit_start(self, trainer, pl_module):
        # Create logdirs and save configs
        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.ckptdir, exist_ok=True)
        os.makedirs(self.cfgdir, exist_ok=True)

        print("Project config")
        print(OmegaConf.to_yaml(self.config))
        OmegaConf.save(self.config,
                        os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))
        
        with open(os.path.join(self.logdir, "argv_content.txt"), "w") as f:
            f.write(str(self.argv_content))


class BackupCodeCallback(Callback):
    def __init__(self, source_dir, backup_dir):
        super().__init__()
        self.source_dir = source_dir
        self.backup_dir = backup_dir

    def on_train_start(self, trainer, pl_module):
        os.makedirs(self.backup_dir+'/model/', exist_ok=True)
        os.makedirs(self.backup_dir+'/modules/', exist_ok=True)
        # 复制代码文件到备份目录
        for root, dirs, files in os.walk(f"{self.source_dir}/model/"):
            for file in files:
                source_file = os.path.join(root, file)
                destination_file = os.path.join(self.backup_dir+'/model/', file)
                
                # 如果目标文件已存在，则跳过拷贝操作
                if os.path.exists(destination_file):
                    continue
                
                # 复制文件到目标文件夹
                shutil.copy2(source_file, destination_file)
        
        
        for root, dirs, files in os.walk(f"{self.source_dir}/modules/"):
            for file in files:
                source_file = os.path.join(root, file)
                destination_file = os.path.join(self.backup_dir+'/modules/', file)
                
                # 如果目标文件已存在，则跳过拷贝操作
                if os.path.exists(destination_file):
                    continue
                
                # 复制文件到目标文件夹
                shutil.copy2(source_file, destination_file)
        
        destination_file = os.path.join(self.backup_dir+'/main.py')
        shutil.copy2(f'{self.source_dir}/main.py', destination_file)

        print(f"Code file backed up to {self.backup_dir}")