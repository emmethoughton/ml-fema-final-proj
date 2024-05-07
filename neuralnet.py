import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class TensorDataset(Dataset):
    def __init__(self, features, labels):
        '''
        features: Numpy array of shape (n_samples, n_features)
        labels: Numpy array of shape (n_samples, 1)
        '''

        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class NeuralNetRegressor(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, 16) 
        self.fc2 = nn.Linear(16, 16)   
        self.fc3 = nn.Linear(16, 16)         
        self.fc4 = nn.Linear(16, 16)         
        self.fc5 = nn.Linear(16, 8)               
        self.fc6 = nn.Linear(8, 4)         
        self.fc7 = nn.Linear(4, 1)         

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        x = torch.sigmoid(self.fc7(x))
        return x
