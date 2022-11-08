#!/usr/bin/env python2
import pandas as pd
# noinspection PyCompatibility
def parse(mapping, clustered_proteins):

    m = pd.read_csv(mapping, sep = "\t", header = None, index_col = 0)
    prokka_set = set(m.index.tolist())
    with open(clustered_proteins) as cp:
        for l in cp.readlines():
            gf, gene_ids = l.strip().split(":")
            matched = False
            for p in gene_ids.strip().split("\t"):
                if p in prokka_set:
                    # noinspection PyCompatibility
                    print "%s\t%s,%s" % (gf, m.loc[p,].iloc[0], gf)
                    matched = True
                    break
            if not matched:
                # noinspection PyCompatibility
                print "%s\t,,%s" % (gf, gf)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("create mapping between Prokka and NCBI PA14 annotation")
    parser.add_argument("mapping", help='PA14 genes mapping')
    parser.add_argument("clustered_proteins", help='Roary clustered proteins')
    #parser.add_argument("out", help='out file for created mapping')
    args = parser.parse_args()
    parse(**vars(args))
