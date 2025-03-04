import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional block: 3 input channels (RGB), 32 output channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        # Second convolutional block: 32 input channels, 64 output channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        # Third convolutional block: 64 input channels, 128 output channels
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(128)
        # Pooling layer to reduce spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers after convolutions.
        # CIFAR-10 images are 32x32. After three rounds of 2x2 pooling, the spatial dimensions are 4x4.
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)  # 10 output classes for CIFAR-10

    def forward(self, x):
        # Convolutional Block 1
        x = self.pool(F.relu(self.batchnorm1(self.conv1(x))))
        # Convolutional Block 2
        x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))
        # Convolutional Block 3
        x = self.pool(F.relu(self.batchnorm3(self.conv3(x))))
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
