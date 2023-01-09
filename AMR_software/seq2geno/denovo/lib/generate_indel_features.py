#!/usr/bin/env python2
import pandas as pd
import os

def summarize(gene_list, roary_gpa, table_out, stat_out, roary_PA14_abricate_map):
    """summarize indel info"""
    indel_genes = []
    #read in roary group to PA14/abricate gene names mapping
    roary2PA14_abricate = pd.read_csv(roary_PA14_abricate_map, sep = "\t", index_col = 0, header = None).iloc[:, 0]
    with open(gene_list) as gl:
        gene_list = [roary2PA14_abricate.loc[i.strip()] for i in gl.readlines()]
    gpa = pd.read_csv(roary_gpa, sep = "\t", index_col = 0)
    isolate2freq = pd.Series(pd.np.zeros(gpa.shape[0]))
    isolate2freq.index = gpa.index
    #restrict to the genes in gene list
    gpa = gpa.loc[:, gene_list]
    for l in gene_list:

        gene_id = "%s_indels.txt" % (l.split(",")[2])
        if os.path.exists(gene_id):
            indels = pd.read_csv(gene_id, sep = "\t", index_col = 0).iloc[:, 0]

            indels.name = l 
            if (indels == 0).any():
                indel_genes.append(l)
                gpa.loc[indels.index, l] = indels 
                isolate2freq.loc[indels.loc[indels == 0].index,] += 1 
    gpa_indel = gpa.loc[:, indel_genes]
    gpa_indel.columns = ["%s_indel" % l for l in gpa_indel.columns]
    gpa_indel.to_csv(table_out, sep = "\t")
    isolate2freq.to_csv(stat_out, sep = "\t")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("process indel calling into per sample table")
    parser.add_argument("gene_list", help='list of genes to be processed')
    parser.add_argument("roary_gpa", help='roary gene / presence absence out')
    parser.add_argument("table_out", help='indel extended feature table output file name')
    parser.add_argument("stat_out", help='per isolate indel frequency')
    parser.add_argument("roary_PA14_abricate_map", help='mapping between roary groups and PA14, and abricate')
    args = parser.parse_args()
    summarize(**vars(args))

