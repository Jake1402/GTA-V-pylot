
'''
https://discuss.pytorch.org/t/splitting-up-sequential-batches-into-randomly-shuffled-train-test-subsets/106466/2
Thank you ptrblck for the example code.
'''

import torch
from torch.utils.data import Dataset

class CustDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        
    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]
    
    def __len__(self):
        return len(self.data) - self.window_size + 1