import torch
import torch.nn as nn
import torchvision
import numpy as np

class GRU_Car(nn.Module):
    def __init__(self, device="cuda"):
        super(GRU_Car, self).__init__()

        self.use_device = device

        self.ConvLayers = nn.Sequential(
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
            #nn.Softmax()
        )

    def forward(self, x):
        self.conLayers = self.ConvLayers(x)
        #self.gru_output, self.h0 = self.gru_unit(self.conLayers, self.h0)
        #self.output = self.LinearSoftmax(self.gru_output)
        return nn.functional.log_softmax(self.conLayers, dim=1)

    def resetParams(self):
        self.h0 = torch.zeros(5, 9).to(self.use_device)