"""
Federated Learning and Over-the-Air Computation
------------------------------------------------
This script is part of a bachelors thesis project evaluating the performance of Orthogonal FL 
and Over-the-Air FL under different system and data conditions.

Author: Filip Svebeck and Fanny Nyberg
Date: Spring 2025
Institution: KTH Royal Institute of Technology
Course: EF112X

For more information, see the corresponding thesis report:
"Federated Learning and Over-the-Air Computation: A Comparative Study"
TRITA-EECS-EX-2025:140
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.fc1 = nn.Linear(64 * 8 * 8, 10)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
