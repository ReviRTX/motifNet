
import os
import numpy as np
import seaborn as sns
import logomaker
import pandas as pd
import matplotlib.pyplot as plt


sns.set(style="ticks", color_codes=True)
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.labelpad'] = 5
plt.rcParams['axes.linewidth']= 2
plt.rcParams['xtick.labelsize']= 18
plt.rcParams['ytick.labelsize']= 18
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['xtick.minor.size'] = 3
plt.rcParams['ytick.minor.size'] = 3
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'

def draw_motif_logo(kernel, output_filepath, pos_feature, seq_len):
    fig, axes = plt.subplots(nrows=32, ncols=2, figsize=(8,60), gridspec_kw={'width_ratios': [0.5, 1]})

    for i in range(32):
        nn_df = pd.DataFrame(kernel[i].transpose(), columns=list('ACGU'))
        ax = axes[i][0]
        ax.set_title('motif_%d' % (i+1), fontdict={"size": 16})
        nn_logo = logomaker.Logo(nn_df, figsize=(4, 1), center_values=False, ax=ax)
        nn_logo.style_spines(visible=False)

        nn_logo.ax.set_xticks([])
        nn_logo.ax.set_yticks([])

        ax = axes[i][1]
        v = pos_feature[i]
        ax.plot(np.arange(seq_len)-(seq_len//2), v)
        vmin = min(-20, min(v))
        vmax = max(20, max(v))
        
        ax.vlines(0, ymin=vmin, ymax=vmax, color='red',  linestyles='dashed')
        ax.hlines(0, xmin=-seq_len//2, xmax=seq_len//2, color='grey',  linestyles='dashed')
        ax.set_title("motif_%d period" % (i+1),  fontdict={"size": 16})
        
        for n in range(-100, seq_len//2, 50):
            if abs(n) < 20:
                continue 
            ax.vlines(x=n, ymin=vmin, ymax=vmax, colors='grey', linestyles='--')

    plt.tight_layout()

    fig.savefig(output_filepath)

def draw_PCA(output_filepath, pca, plot_df):
    fig = plt.figure(figsize=(8,6))
    ax = fig.gca()

    h_order = ['not_regulated', 'down_regulated', 'up_regulated']
    sns.scatterplot(x='PC1', y='PC2', data=plot_df, size='size', hue='glabel', alpha=0.8, hue_order=h_order, ax=ax) 

    PC1_ratio, PC2_ratio = pca.explained_variance_ratio_[:2] * 100
    ax.set_xlabel('PC1 (%.1f' % PC1_ratio + '%)', fontdict={"size": 18})
    ax.set_ylabel('PC2 (%.1f' % PC2_ratio + '%)', fontdict={"size": 18})
    ax.set_xlim(plot_df.PC1.min()-1, plot_df.PC1.max() + 1.5)
    ax.set_ylim(plot_df.PC2.min()-1, plot_df.PC2.max() + 1.5)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[1:4], labels[1:4], loc='upper right', fontsize=12)
    
    fig.savefig(output_filepath, bbox_inches='tight')
