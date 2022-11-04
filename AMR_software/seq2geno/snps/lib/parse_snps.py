#!/usr/bin/env python2

import pandas as pd
def snp2bin(snp_f, out):
    #m = pd.read_csv(snp_f, sep = "\t", na_values = ['NR'], header = 0)
    m = pd.read_csv(snp_f, sep = "\t", na_values = ['NR', ''], header = 0)
    name_cols = m.iloc[:, 0:6].astype('string')
    names = name_cols.apply(lambda x: "_".join(x), axis = 1)
    #names = m["gene"].map(str) + "_" + m["pos"].map(str) + m["ref aa"].map(str)
    m.index = names 
    m.drop(["gene", "pos", "ref", "alt", "ref aa", "alt aa"], axis = 1, inplace = True)
    #replace nas with 0
    #m.replace(["NR", ""], [0, 0], inplace = True)
    #m[pd.isnull] = 0
    m= m.fillna(0)
    m = m.astype('int')
    #replace snps with quality score over 1 with 1
    mask = m > 0
    m = m.where(~mask, 1)
    m.T.to_csv(out, sep = "\t", index_label= 'Isolate')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("convert snp table")
    parser.add_argument("snp_f", help='snp table')
    parser.add_argument("out", help='out table')
    a = parser.parse_args()
    snp2bin(**vars(a))
