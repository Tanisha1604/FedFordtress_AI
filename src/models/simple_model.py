"""
Simple CNN Model for CIFAR-10
A lightweight convolutional neural network for image classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for CIFAR-10 dataset.
    Architecture: Conv -> Conv -> MaxPool -> Conv -> FC -> FC
    """
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After 3 pooling layers: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # First conv block: 3x32x32 -> 32x32x32 -> 32x16x16
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Second conv block: 32x16x16 -> 64x16x16 -> 64x8x8
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Third conv block: 64x8x8 -> 128x8x8 -> 128x4x4
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten: 128x4x4 -> 2048
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class SimpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron for simpler experiments.
    """
    
    def __init__(self, input_size=3072, num_classes=10):
        super(SimpleMLP, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Flatten input
        x = x.view(-1, 3072)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

