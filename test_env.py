import pandas as pd
import numpy as np

from models.utr_net import motifNet
from siren_trainer import baseTrainer
from tqdm import tqdm
import random

import torch
from torch import nn, optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from datasets.utr_dataset import BaseDataset

import argparse
import utils
import time
import os

model = motifNet()

seq = np.random.random((2, 4, 256))
seq = torch.from_numpy(np.float32(seq))
res = model(seq)
print(res.shape)

print('\n', 'environment has been installed sucessfully!')
