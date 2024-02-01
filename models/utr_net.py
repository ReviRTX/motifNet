import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import SineLayer


class motifNet(nn.Module):
    def __init__(self, args):
        super(motifNet, self).__init__()
        self.args = args

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mgrid = torch.linspace(-1, 1, steps=args.seq_len).to(self.device)
        self.coords = self.mgrid.view(1, 1, args.seq_len)

        filters = 32
        self.layer1 = nn.Sequential(
                nn.Conv1d(4, filters, kernel_size=11, stride=1, padding=5, bias=True),
                nn.ReLU(),
                #nn.Conv1d(filters, filters, kernel_size=1, stride=1, padding=0, bias=True),
                #nn.Sigmoid(),
                #nn.ReLU(),
                )

        self.layer_a = nn.Sequential(
                nn.Conv1d(4, filters, kernel_size=11, padding=5),
                #nn.MaxPool1d(5, stride=2, padding=2),
                nn.Conv1d(filters, 128, kernel_size=11, padding=5), 
                #nn.MaxPool1d(5, stride=2, padding=2),
                #nn.BatchNorm1d(32), 
                nn.ReLU(),

                nn.Conv1d(128, 128, kernel_size=11, padding=5), 
                #nn.MaxPool1d(5, stride=2, padding=2),
                #nn.Conv1d(64, 64, kernel_size=11, padding=5), 
                #nn.MaxPool1d(5, stride=2, padding=2),
                #nn.BatchNorm1d(64), 
                nn.ReLU(),
                
                #nn.Conv1d(64, 64, kernel_size=11, padding=5), 
                #nn.MaxPool1d(5, stride=2, padding=2),
                nn.Conv1d(128, 32, kernel_size=11, padding=5), 
                #nn.MaxPool1d(5, stride=2, padding=2),
                #nn.BatchNorm1d(128), 
                nn.Tanh(),
                nn.ReLU(),
                #nn.Sigmoid(),
                )

        if self.args.task == 'regression': 
            self.layer_b = nn.Linear(32, 1)
        elif self.args.task == 'binary_classification':
            self.layer_b = nn.Sequential(
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
        elif self.args.task == 'multi_class':
            self.layer_b = nn.Sequential(
                nn.Linear(32, self.num_classes),
                )
        else:
            raise ValueError("Invalid task option. Please choose from 'regression', 'binary_classification', or 'multi_class'.")
        
        self.conv_sin = nn.Sequential(
                SineLayer(1, 128, is_first=True, omega_0=100.),
                nn.Conv1d(128, filters, kernel_size=5, stride=1, padding=2, bias=True),
                )

    def forward_motif_map(self, input_x):
        seq_x = self.layer1(input_x) 
        pos_feature = self.conv_sin(self.coords)
        seq_x = seq_x * pos_feature
        seq_x = nn.ReLU()(seq_x)
        
        heatmap = self.layer_a(input_x)
        heatmap = F.interpolate(heatmap, scale_factor=8)
        heatmap = heatmap[:,:, 3:-3]

        return seq_x, heatmap

    def forward(self, input_x):
        seq_x = self.layer1(input_x) 
        pos_feature = self.conv_sin(self.coords)
        seq_x = seq_x * pos_feature
        seq_x = nn.ReLU()(seq_x)

        #heatmap = self.layer_a(input_x)
        #heatmap = self.layer_a(seq_x)

        #heatmap = F.interpolate(heatmap, scale_factor=4)
        #heatmap = heatmap[:,:, 3:-3]

        #seq_x = heatmap * seq_x

        heatmap = seq_x.mean(dim=2)
        x = self.layer_b(heatmap).squeeze(1)

        return x


    def forward_with_feature(self, input_x):
        seq_x = self.layer1(input_x) 
        pos_feature = self.conv_sin(self.coords)
        seq_x = seq_x * pos_feature
        seq_x = nn.ReLU()(seq_x)

        #heatmap = self.layer_a(input_x)
        #seq_x = heatmap * seq_x
        #x = self.layer_b(heatmap).squeeze(1)

        return seq_x




