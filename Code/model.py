import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    def __init__(self, g_output_dim):
        super(Generator, self).__init__()       
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    # forward method
    def forward(self, x): 
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))

        
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(d_input_dim, 1024)
        self.ln1 = nn.LayerNorm(1024)
        
        self.fc2 = nn.Linear(1024, 512)
        self.ln2 = nn.LayerNorm(512)
        
        self.fc3 = nn.Linear(512, 256)
        self.ln3 = nn.LayerNorm(256)
        
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.leaky_relu(self.ln1(self.fc1(x)), 0.2)
        x = F.leaky_relu(self.ln2(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.ln3(self.fc3(x)), 0.2)
        return self.fc4(x)
"""
class Discriminator(nn.Module):
    def __init__(self, d_input_dim):
        super(Discriminator, self).__init__()
        #self.fc1 = nn.Linear(d_input_dim, 1024))
        #self.fc2 = spectral_norm(nn.Linear(self.fc1.out_features, self.fc1.out_features//2))
        #self.fc3 = spectral_norm(nn.Linear(self.fc2.out_features, self.fc2.out_features//2))
        #self.fc4 = spectral_norm(nn.Linear(self.fc3.out_features, 1))
        self.fc1 = nn.LayerNorm(nn.Linear(d_input_dim, 1024))
        self.fc2 = nn.LayerNorm(nn.Linear(self.fc1.out_features, self.fc1.out_features//2))
        self.fc3 = nn.LayerNorm(nn.Linear(self.fc2.out_features, self.fc2.out_features//2))
        self.fc4 = nn.LayerNorm(nn.Linear(self.fc3.out_features, 1))
    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        #return torch.sigmoid(self.fc4(x))
        return self.fc4(x)
"""
