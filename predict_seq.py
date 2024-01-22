import torch
from torch import nn
import torch.nn.functional as F
import os
import numpy as np

from models.utr_net import motifNet

import os
import pandas as pd


nuc_d = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1], 'n':[0,0,0,0]}
def one_hot_encode(seq):    
    seq = seq.lower()
    a = np.array([nuc_d[x] for x in seq])
    return a

def predict(input_file, model, prefix):
    model.eval()

    df = pd.read_csv(input_file)
    
    feature_list = []
    pred_delta_psi = []
    for i, line in df.iterrows():
        x = one_hot_encode(line.seq)
        x = torch.from_numpy(np.float32(x.transpose())).unsqueeze(0)

        feature = model.forward_with_feature(x)
        feature_list.append(feature.detach().numpy())

        pred_delta_psi.append(model(x).detach().numpy()[0])

    feature_arr = np.concatenate(feature_list, axis=0)
    print('motif fingerprint shape:', feature_arr.shape)


    df.loc[:, "predict_delta_psi"] = pred_delta_psi

    np.save("%s_fingerprint" % prefix, feature_arr)
    df.to_csv("%s.csv" % prefix, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, required=True, default=None, help="input data")
    parser.add_argument("--prefix", type=str, required=True, default=None, help="output prefix")
    parser.add_argument("--ckpt", type=str, required=True, default=None, help="the path of model weight file")
    parser.add_argument("--seq_len", type=int, required=True, default=None, help="sequence length")
    args = parser.parse_args()

    model = motifNet(args.seq_len)
    state = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(state)
    
    predict(args.i, model, args.prefix)




