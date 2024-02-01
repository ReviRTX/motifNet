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

    if torch.cuda.is_available():
        model = model.cuda()

    df = pd.read_csv(input_file)
    feature_list = []
    for i, line in df.iterrows():
        x = one_hot_encode(line.seq)
        x = torch.from_numpy(np.float32(x.transpose())).unsqueeze(0)
        if torch.cuda.is_available():
            x = x.cuda()

        feature = model.forward_with_feature(x)
        
        if torch.cuda.is_available():
            feature_list.append(feature.detach().cpu().numpy().reshape(1, -1))
        else:
            feature_list.append(feature.detach().numpy().reshape(1, -1))

    feature_arr = np.concatenate(feature_list, axis=0)

    pca = PCA(n_components=2, whiten=True)
    pca.fit(feature_arr)

    comp = pca.transform(feature_arr)
    df.loc[:, ["PC1", "PC2"]] = comp

    output_filepath = "%s_PCA.pdf" % prefix
    draw_PCA(output_filepath, pca, df, args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, required=True, default=None, help="input data")
    parser.add_argument("--prefix", type=str, required=True, default=None, help="output prefix")
    parser.add_argument("--ckpt", type=str, required=True, default=None, help="the path of model weight file")
    parser.add_argument("--seq_len", type=int, required=True, default=None, help="sequence length")
    parser.add_argument("--color_by", type=str, required=True, default=None)
    parser.add_argument("--task", type=str, required=True, choices=["binary_classification", 
                                                                "multi_class",
                                                                "regression",
                                                            ], 
                                                            help="task type")
    parser.add_argument("--num_classes", type=int, required=False, default=3)
    args = parser.parse_args()

    model = motifNet(args)
    state = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(state)
    
    draw_graph(args.i, model, args.prefix, args.seq_len, args.color_by)




