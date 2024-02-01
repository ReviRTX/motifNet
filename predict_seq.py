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

    df = pd.read_csv(input_file)
    
    feature_list = []
    pred_label = []
    for i, line in df.iterrows():
        x = one_hot_encode(line.seq)
        x = torch.from_numpy(np.float32(x.transpose())).unsqueeze(0)

        if torch.cuda.is_available():
            x = x.cuda()

        feature = model.forward_with_feature(x)
        if torch.cuda.is_available():
            feature_list.append(feature.detach().cpu().numpy())
            pred_label.append(model(x).detach().cpu().numpy()[0])
        else:
            feature_list.append(feature.detach().numpy())
            pred_label.append(model(x).detach().numpy()[0])


    feature_arr = np.concatenate(feature_list, axis=0)
    print('motif fingerprint shape:', feature_arr.shape)


    df.loc[:, "predict_label"] = pred_label

    np.save("%s_fingerprint" % prefix, feature_arr)
    df.to_csv("%s.csv" % prefix, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, required=True, default=None, help="input data")
    parser.add_argument("--prefix", type=str, required=True, default=None, help="output prefix")
    parser.add_argument("--ckpt", type=str, required=True, default=None, help="the path of model weight file")
    parser.add_argument("--seq_len", type=int, required=True, default=None, help="sequence length")
    parser.add_argument("--task", type=str, required=True, choices=["binary_classification", 
                                                                "multi_class",
                                                                "regression",
                                                            ], 
                                                            help="task type")
    args = parser.parse_args()

    model = motifNet(args)
    model.eval()
    
    state = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(state)
    
    if torch.cuda.is_available():
        model = model.cuda()
    predict(args.i, model, args.prefix)




