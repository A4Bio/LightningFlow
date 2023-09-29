import torch
from torch.utils.data import Dataset



class CLS_Dataset(Dataset):
    def __init__(
        self,
        split='train'
    ) -> None:
        super().__init__()
        self.data = torch.rand(1000,784)
        self.label = torch.randint(0,10,(1000,))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(
        self, index
    ):  
        retval = {
            "data": self.data[index],
            "label": self.label[index]
        }
        return retval
    
    
if __name__ == '__main__':
    dataset = CLSDataset()
    print(dataset[0])