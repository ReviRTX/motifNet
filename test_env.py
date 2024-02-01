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
from datasets.custom_dataset import BaseDataset

import argparse
import utils
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument("command", metavar="<command>", help="train or infer")
parser.add_argument("--seq_len", type=int, required=True, help="the path of model weight file")
parser.add_argument("--task", type=str, required=True, choices=["binary_classification", 
                                                            "multi_class",
                                                            "regression",
                                                        ], 
                                                            help="task type")

args = parser.parse_args()
model = motifNet(args)
seq = np.random.random((2, 4, args.seq_len))
seq = torch.from_numpy(np.float32(seq))

if torch.cuda.is_available():
    model = model.cuda()
    seq = seq.cuda()

res = model(seq)
print('output shape: ', res.shape)

print('\n', 'environment has been installed sucessfully!')
