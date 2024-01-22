import numpy as np
import pandas as pd

def data_processor(input_path, output_path, seq_len):
    df = pd.read_csv(input_path, delimiter='\t')

    df = df[df.loc[:, "avg_read_counts(treat)"] > 100]

    out_df = pd.DataFrame()
    out_df.loc[:, "delta_psi"] = df.apply(lambda x: 0 if x.p_value > 0.05 else x.delta_psi, axis=1).values
    out_df.loc[:, "seq"] = df.donor_seq.apply(lambda x: x[2500-seq_len//2:2500+seq_len//2]).values

    print(out_df.head())
    out_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, required=True, default=None, help="the path of model weight file")
    parser.add_argument("-o", type=str, required=True, default=None, help="the path of model weight file")
    parser.add_argument("--seq_len", type=int, required=True, default=None, help="the path of model weight file")
    args = parser.parse_args()

    data_processor(args.i, args.o, args.seq_len)
