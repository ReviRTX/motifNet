import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np

from sklearn.decomposition import PCA
from models.utr_net import motifNet

import os
import seaborn as sns
import pandas as pd

from draw_utils import draw_motif_logo, draw_PCA

# Dictionary returning one-hot encoding of nucleotides.
nuc_d = {'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1], 'n':[0,0,0,0]}
def one_hot_encode(seq):    
    seq = seq.lower()
    a = np.array([nuc_d[x] for x in seq])
    return a

def draw_graph(input_file, model, prefix, seq_len, threshold):
    model.eval()
    coords = torch.linspace(-1, 1, steps=seq_len).view(1, 1, seq_len)
    pos_feature = model.conv_sin(coords)[0].detach().numpy()
    kernel = model.layer1[0].weight.detach().numpy()

    #rank kernel by positional activated signal
    max_val = np.max(pos_feature, axis=1)
    sort_idx = np.argsort(max_val)[::-1]
    np.save("%s_sorted_kernel_index.npy" % prefix, sort_idx)

    kernel = kernel[sort_idx]
    pos_feature = pos_feature[sort_idx]

    output_filepath = "%s_ranked_motif.pdf" % prefix
    draw_motif_logo(kernel, output_filepath, pos_feature, seq_len)


    df = pd.read_csv(input_file)
    neg_df = df[df.delta_psi==0].copy()
    df = df[np.abs(df.delta_psi) > threshold]

    neg_seq = np.random.choice(neg_df.seq.values, 1000, replace=False)

    feature_list = []
    for i, line in df.iterrows():
        x = one_hot_encode(line.seq)
        x = torch.from_numpy(np.float32(x.transpose())).unsqueeze(0)

        feature = model.forward_with_feature(x)
    #     feature = torch.cat([feature[:,:, :128-10], feature[:,:, 128+10:]], dim=2)
        feature_list.append(feature.detach().numpy().reshape(1, -1))

    feature_arr = np.concatenate(feature_list, axis=0)

    feature_list = []
    for seq in neg_seq:
        x = one_hot_encode(seq)
        x = torch.from_numpy(np.float32(x.transpose())).unsqueeze(0)

        feature = model.forward_with_feature(x)
    #     feature = torch.cat([feature[:,:, :128-10], feature[:,:, 128+10:]], dim=2)
        feature_list.append(feature.detach().numpy().reshape(1, -1))

    neg_feature_arr = np.concatenate(feature_list, axis=0)

    pca = PCA(n_components=2, whiten=True)
    pca.fit(np.concatenate([feature_arr, neg_feature_arr], axis=0))

    comp = pca.transform(feature_arr)

    df.loc[:,'label'] = np.where(df.delta_psi>threshold, 1, 0).copy()
    df.loc[:,'size'] = df.delta_psi.abs() 
    df.loc[:, ["PC1", "PC2"]] = comp

    neg_comp = pca.transform(neg_feature_arr)
    neg_df = pd.DataFrame({
        'PC1': neg_comp[:, 0],
        'PC2': neg_comp[:, 1],
    })

    pos_df = df.loc[:, ['PC1', 'PC2', 'size']].copy()
    pos_df.loc[:, 'glabel'] = df.label.apply(lambda x:'up_regulated' if x==1 else 'down_regulated')
    pos_df.head()

    neg_df.loc[:, 'size'] = 0.2
    neg_df.loc[:, 'glabel'] = 'not_regulated'
    plot_df = pd.concat([neg_df,pos_df], axis=0)
    plot_df.head()

    output_filepath = "%s_PCA.pdf" % prefix
    draw_PCA(output_filepath, pca, plot_df)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, required=True, default=None, help="input data")
    parser.add_argument("--prefix", type=str, required=True, default=None, help="output prefix")
    parser.add_argument("--ckpt", type=str, required=True, default=None, help="the path of model weight file")
    parser.add_argument("--seq_len", type=int, required=True, default=None, help="sequence length")
    parser.add_argument("--psi_threshold", type=float, required=True, default=None, help="threshold for defining group by delta psi")
    args = parser.parse_args()

    model = motifNet(args.seq_len)
    state = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(state)
    
    draw_graph(args.i, model, args.prefix, args.seq_len, args.psi_threshold)




