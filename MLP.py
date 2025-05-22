"""
Federated Learning and Over-the-Air Computation
------------------------------------------------
This script is part of a bachelor’s thesis project evaluating the performance of Orthogonal FL 
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

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
