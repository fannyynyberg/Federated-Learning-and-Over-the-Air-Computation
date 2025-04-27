import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Första convolutionslagret: från 3 kanaler (RGB) till 32 feature maps
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Andra convolutionslagret: från 32 feature maps till 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Fullt kopplat lager
        self.fc1 = nn.Linear(64 * 8 * 8, 10)  # efter två poolingar är bilderna 8x8 stora

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv -> BN -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv -> BN -> ReLU -> Pool
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)  # Fully connected
        return x
