import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

'''
Object for tensorizing a numpy array for training of PyTorch neural network
'''
class TensorDataset(Dataset):
    '''
    Initialize the dataset
        features = Numpy array of shape (n_samples, n_features)
        labels = Numpy array of shape (n_samples, 1)
    '''
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    '''
    Obtain the number of features
    '''
    def __len__(self):
        return len(self.features)

    '''
    Get the sample at index idx
        idx = the element index of interest
    '''
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

'''
PyTorch neural network for linear regression on labels in (0, 1]
'''
class NeuralNetRegressor(nn.Module):
    '''
    Initialize the neural network
        input_size = the number of input features
    '''
    def __init__(self, input_size):
        super(NeuralNetRegressor, self).__init__()
        self.fc1 = nn.Linear(input_size, 16) 
        self.fc2 = nn.Linear(16, 16)   
        self.fc3 = nn.Linear(16, 16)         
        self.fc4 = nn.Linear(16, 16)         
        self.fc5 = nn.Linear(16, 8)               
        self.fc6 = nn.Linear(8, 4)         
        self.fc7 = nn.Linear(4, 1)         

    '''
    Forward propagation of input vector x through neural network
        x = the input features
    '''
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = torch.sigmoid(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        x = torch.sigmoid(self.fc7(x))
        return x
