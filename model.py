import torch
import torch.nn as nn
import torchvision
import numpy as np

class CNN_Car(nn.Module):
    def __init__(self, device="cuda"):
        super(CNN_Car, self).__init__()
        self.use_device = device
        
        '''
        Our current Gen 1 model is a linear model that has no "memory" a future version may
        consist of some form of RNN network or could feature stacked image with positional 
        encoding so the model can learn the sequence of events. I think my Gen 2 model 
        however will consist of residual connections.
        '''
        self.CNN_Model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0), #chagne 128
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Flatten(),
            nn.Linear(in_features=35328, out_features=8192),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(in_features=8192, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(in_features=4096, out_features=6),
        )

    def forward(self, x):
        self.model_logits = self.CNN_Model(x)
        return nn.functional.log_softmax(self.model_logits, dim=1)
