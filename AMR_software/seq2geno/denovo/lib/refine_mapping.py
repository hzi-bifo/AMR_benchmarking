#!/usr/bin/env python2
import pandas as pd


def parse(mapping):
    m_dict = {}
    with open(mapping) as m:
        for l in m.readlines():
#            print('---')
#            print(l.strip())
            if len(l.strip().split("\t")) != 3:
                continue
            pa_id, gene, p_raw = l.strip().split("\t")
            p_dict = dict([attr.split("=") for attr in p_raw.split(";")])
            name = "%s,%s"
            if not "gene" in p_dict:
                prokka_gene = ""
            else:
                prokka_gene = p_dict["gene"]
            name %= pa_id, gene
            print "%s\t%s" % (p_dict["ID"], name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("create mapping between Prokka and NCBI PA14 annotation")
    parser.add_argument("mapping", help='raw mapping created with bash')
    #parser.add_argument("out", help='out file for created mapping')
    args = parser.parse_args()
    #parse(**vars(args))
    parse(args.mapping)
     
