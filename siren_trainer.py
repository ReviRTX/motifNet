
import pandas as pd
import numpy as np
import torch

from config import config
import time, os
from tqdm import tqdm

class baseTrainer():
    def __init__(self):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _init_model(self):
        raise NotImplementedError

    def _init_dataloader(self):
        raise NotImplementedError

    def log(self, info):
        print(info)

    def save_ckpt(self, state, model_save_dir, epoch):
        os.makedirs(model_save_dir, exist_ok=True)
        model_path = os.path.join(model_save_dir, str(epoch)+'.pth')
        torch.save(state, model_path)

    def load_ckpt(self, ckpt_path):
        state = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(state)
        self.log("=> loaded checkpoint from %s" % ckpt_path) 

   
    def optimize_parameters(self, inputs, target):
        self.optimizer.zero_grad()
        output = self.model(inputs)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        loss = loss.item()
        
        return loss
    
    
