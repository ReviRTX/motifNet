import os, copy
import scipy.io as sio
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from sklearn import preprocessing
import pandas as pd

class BaseDataset(Dataset):
    def __init__(self, df, seq_len, train=True):
        super(BaseDataset, self).__init__()

        df.reset_index(drop=True, inplace=True)

        #self.scaler = preprocessing.StandardScaler().fit(df.loc[:,'delta_psi'].values.reshape(-1,1))

        self.seq_e = self.one_hot_encode(df, seq_len=seq_len)

        self.scaler = preprocessing.StandardScaler().fit(df.loc[:,'delta_psi'].values.reshape(-1,1))
        self.original_rl = df.loc[:,'delta_psi'].values 
        #self.rl = self.original_rl = (df.loc[:,'delta_psi'].values * 10) ** 2
        self.rl = self.scaler.transform(df.loc[:,'delta_psi'].values.reshape(-1,1))[:, 0]
            
        # Dictionary returning one-hot encoding of nucleotides. 
        self.nuc_d = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1], 'n':[0,0,0,0]}


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
        utr_seq = self.seq_e[i].transpose()
        rl = self.rl[i]
        
        #x = torch.from_numpy(utr_seq)
        #x = x.float()
        #y = torch.tensor(rl, dtype=torch.float32)
        
        x = np.float32(utr_seq)
        y = np.float32(rl)

        return x, y

    def __len__(self):
        return len(self.seq_e)


if __name__ == '__main__':
    d = BaseDataset()

    for n in range(20):
        x,y = d[n]
        print(x.shape, x, y)

