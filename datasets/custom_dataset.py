import os, copy
import scipy.io as sio
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from sklearn import preprocessing
import pandas as pd

class BaseDataset(Dataset):
    def __init__(self, args, train=True):
        super(BaseDataset, self).__init__()
        self.args = args

        if not train and args.v is not None:
            df = pd.read_csv(args.v)
        else:
            df = pd.read_csv(args.i)

        df.reset_index(drop=True, inplace=True)

        self.seq_e = self.one_hot_encode(df, seq_len=args.seq_len)
        
        if args.task == 'regression':
            self.scaler = preprocessing.StandardScaler().fit(df.loc[:, args.label_colname].values.reshape(-1,1))
            self.original_label = df.loc[:, args.label_colname].values 
            self.label = self.scaler.transform(df.loc[:, args.label_colname].values.reshape(-1,1))[:, 0]
        else:
            self.label = df.loc[:, args.label_colname].values 

    def one_hot_encode(self, df, seq_len):
        # Dictionary returning one-hot encoding of nucleotides. 
        nuc_d = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1], 'n':[0,0,0,0]}
        
        # Creat empty matrix.
        vectors=np.empty([len(df),seq_len,4])
        
        for i,seq in enumerate(df.seq.values):
            seq = seq.lower()
            a = np.array([nuc_d[x] for x in seq])
            vectors[i] = a
        
        return vectors

    def __getitem__(self, i):
        seq = self.seq_e[i].transpose()
        label = self.label[i]
        
        seq = np.float32(seq)
        
        if self.args.task in ["regression", "binary_classification"]:
            label = np.float32(label)

        return seq, label

    def __len__(self):
        return len(self.seq_e)


if __name__ == '__main__':
    d = BaseDataset()

    for n in range(20):
        x,y = d[n]
        print(x.shape, x, y)

