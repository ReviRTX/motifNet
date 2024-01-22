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

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.enabled = False


class sinTrainer(baseTrainer):
    def __init__(self, args):
        super(sinTrainer, self).__init__()
        self.args = args
        self._init_model()

    def _init_model(self):
        self.model = motifNet(self.args.seq_len)
        self.model = self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

        self.criterion = nn.MSELoss()
        self.criterion_motif = nn.MSELoss()

    def _init_dataloader(self, input_path, seq_len, train=True):
        df = pd.read_csv(input_path) 

        if train: 
            train_dataset = BaseDataset(df, seq_len, train=True)
            self.train_dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=8)
            #self.coords = train_dataset.coords.to(self.device)

        val_df = df.iloc[:1000]
        val_dataset = BaseDataset(val_df, seq_len, train=False)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.config.batch_size, num_workers=4)

        if train:
            print("train_datasize", len(train_dataset), "val_datasize", len(val_dataset))
        else:
            print("val_datasize", len(val_dataset))
    
    def val_to_device(self, val_list):
        new_vals = []
        for val in val_list:
            new_vals.append(val.to(self.device))
        return new_vals

    def train(self):
        self._init_dataloader(self.args.i, self.args.seq_len)
        model_save_dir = os.path.join(self.config.ckpt, self.__class__.__name__)

        for epoch in range(self.config.max_epoch):
            since = time.time()
            
            self.model.train()
            
            for n, val_list in enumerate(tqdm(self.train_dataloader)):
                inputs, target = self.val_to_device(val_list)

                self.optimizer.zero_grad()

                output = self.model(inputs)

                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                loss = loss.item()
                
                if n % 1000 == 0:
                    print('e%02d step-%d loss:%.4f ' % (epoch, n, loss))
        
            train_loss = loss

            self.model.eval()
            loss_meter, it_count = 0, 0 
            with torch.no_grad():
                for val_list in self.val_dataloader:
                    inputs, target = self.val_to_device(val_list)

                    output = self.model(inputs)

                    loss = self.criterion(output, target)
                    loss_meter += loss.item()
                    it_count += 1
            val_loss =  loss_meter / it_count
    

            self.log('#epoch:%02d loss:%.4f | %.4f time:%s'
                  % (epoch, train_loss, val_loss, utils.print_time_cost(since)))
            
        self.save_ckpt(self.model.state_dict(), model_save_dir, '%s_%d' % (args.prefix, epoch+1))


    def onehot(self, seq):
        nuc_d = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1], 'n':[0,0,0,0]}
        pad_len = 100-len(seq)
        seq = seq + 'N'* pad_len

        seq = seq.lower()
        enc = np.array([nuc_d[x] for x in seq], dtype=np.float32)
        return enc



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("command", metavar="<command>", help="train or infer")
    parser.add_argument("-i", type=str, required=True, default=None, help="the path of model weight file")
    parser.add_argument("--seq_len", type=int, required=True, default=None, help="the path of model weight file")
    parser.add_argument("--prefix", type=str, required=True, default=None, help="output path prefix")
    parser.add_argument("--ckpt", type=str, default=None, help="the path of model weight file")
    parser.add_argument("--seed", type=int, default=None, help="the path of model weight file")
    args = parser.parse_args()


    if args.seed is not None:
        seed_torch(args.seed)

    trainer = sinTrainer(args)
    if (args.command == "train"):
        if args.ckpt is not None:
            trainer.load_ckpt(args.ckpt)
        trainer.train()

    if (args.command == "inference"):
        assert args.ckpt is not None
        trainer.load_ckpt(args.ckpt)
        #trainer.transfer_ckpt(args.ckpt)
        
        trainer.inference()

