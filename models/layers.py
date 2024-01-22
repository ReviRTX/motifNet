
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.conv = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.conv.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.conv.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.conv(input))
    

class LinearSine(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))




class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, bias=True, residual=True):
        super().__init__()
        self.residual = residual

        self.conv = nn.Sequential(
                nn.Conv1d(in_features, out_features, kernel_size=kernel_size, stride=1, 
                    padding=(kernel_size-1)//2, bias=bias),
                nn.BatchNorm1d(out_features), 
                #nn.ReLU(),
                nn.LeakyReLU(0.2, inplace=True),
            )
    
    def forward(self, x):
        if self.residual:
            return x + self.conv(x) 
        else:
            return self.conv(x) 
