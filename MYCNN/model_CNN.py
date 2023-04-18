import cv2
import os
import numpy as np
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
import torch.nn.functional as F


class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=32*15*21*0+10560, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = nn.ReLU()(x)
        x = self.pool1(x)
        # print(x.shape)
        x = self.dropout1(x)

        x = self.conv2(x)
        # print(x.shape)

        x = nn.ReLU()(x)
        x = self.pool2(x)
        # print(x.shape)
        x = torch.flatten(x, 1)

        # x = x.view(-1, 32*15*21*0+337920)
        # print(x.shape)

        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        return x

