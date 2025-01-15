import numpy as np
import torch
from torch import nn

'''

Credits for this code go to statquest. specifically
from their transformer from scratch video. The code
modified for my use case is the forwad3D function
we allows to positionally encode an RGB image.

Our use case of this positional encoding allows the model
to have some understanding of sequential data. By stack 
3 frames that are fetched every 5 frames we can generate
a rough understanding of the world and how time passes.

'''

class PositionEncoding(nn.Module):
    def __init__(self, d_model = 2, max_len=6):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()

        div_term = 1/torch.tensor(10000.0)**(embedding_index/d_model)

        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
         
        self.register_buffer("pe", pe)

    def forward(self, pos_embedding):
        return pos_embedding + self.pe[:pos_embedding.size(0), :]
    
    def forward3D(self, td_embedding):
        pe_cat = torch.concatenate((
            self.pe[:td_embedding.size(1), :].unsqueeze(dim=2),
            self.pe[:td_embedding.size(1), :].unsqueeze(dim=2),
            self.pe[:td_embedding.size(1), :].unsqueeze(dim=2)
        ), dim=2)
        return td_embedding + pe_cat
        #return td_embedding + 0.125*torch.add(pe_cat, 1)