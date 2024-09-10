import json
from torch.utils.data import Dataset



class PDB_Dataset(Dataset):
    def __init__(self, split='train', block_size=512):
        super().__init__()
        with open('/huyuqi/xmyu/FoldToken2/foldtoken2_data/pdb_vqids_ft4/pdb_256.jsonl', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if split=='train':
            lines = lines[:-100]
        else:
            lines = lines[-100:]

        self.ESO_TOKEN = 256
        self.PAD_TOKEN = 257
        self.lines = lines
        self.block_size = block_size

    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        line = self.lines[idx]
        entry = json.loads(line)
        chain_zero_indices = [i for i, chain in enumerate(list(entry.values())[0]['chain']) if chain == 0]
        seq_chain_zero = [list(entry.values())[0]['seq'][index] for index in chain_zero_indices]
        vqid_chain_zero= [list(entry.values())[0]['vqid'][index] for index in chain_zero_indices]
        
        padded_seq = [self.ESO_TOKEN] + seq_chain_zero + [self.PAD_TOKEN] * (self.block_size - len(seq_chain_zero) - 1)
        padded_vqid = [self.ESO_TOKEN] + vqid_chain_zero + [self.PAD_TOKEN] * (self.block_size - len(vqid_chain_zero) - 1)
        each_data = {'seq': padded_seq[:self.block_size], 'vqid': padded_vqid[:self.block_size]}
        return each_data