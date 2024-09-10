import argparse
import torch
from omegaconf import OmegaConf
from model import MInterface
import os
import tqdm
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_out', default='/huyuqi/lmd/TokenDiff/results/TokenDiff_baseline/pred_token', type=str)
    parser.add_argument('--config', default='cFoldGen/results/FoldFlow_cath_baseline/configs/06-29T22-49-10-project.yaml', type=str)
    parser.add_argument('--checkpoint', default='cFoldGen/results/FoldFlow_cath_baseline/checkpoints/best-epoch=767-val_loss=0.0622.pth', type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--sample", type=str, default='euler')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = create_parser()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = OmegaConf.load(args.config)
    config = OmegaConf.to_container(config, resolve=True)
    model = MInterface(**config)
    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cuda'))
    for key in list(checkpoint.keys()):
        if '_forward_module.' in key:
            checkpoint[key.replace('_forward_module.', '')] = checkpoint[key]
            del checkpoint[key]
    model.load_state_dict(checkpoint, strict=False)
    model = model.to('cuda')

    os.makedirs(args.path_out, exist_ok=True)

    for file_name in tqdm(range(10)):
        protein_token = model.sample(batch_size = args.batch_size, steps = args.steps, sample = args.sample, device = device)
        print(protein_token)