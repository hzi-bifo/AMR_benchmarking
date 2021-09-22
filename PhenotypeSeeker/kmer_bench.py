#!/usr/bin/python
import sys
import os
sys.path.append('../')
sys.path.insert(0, os.getcwd())
import pandas as pd
import numpy as np
import ast
import errno
import itertools
from itertools import repeat
import multiprocessing as mp
import amr_utility.name_utility
import amr_utility.math_utility
# from amr_utility import name_utility
import warnings
import argparse
import os
import logging
import math


def get_feature_vector(cls):
    # Feature vector loop
    iterate_to_union = [[x] for x in list(Input.samples.values())]
    for i in range(math.log(cls.no_samples, 2).__trunc__()):
        iterate_to_union = [x[0] for x in iterate_to_union]
        iterate_to_union = [
            iterate_to_union[j: j + 4 if len(iterate_to_union) < j + 4 else j + 2] for j in
            range(0, len(iterate_to_union), 2) if j + 2 <= len(iterate_to_union)
        ]
        Input.pool.map(partial(cls.get_union, round=i), iterate_to_union)
    call(["mv %s K-mer_lists/feature_vector.list" % cls.union_output[-1]], shell=True)
    # [(lambda x: call(["rm -f {}".format(x)], shell=True))(union) for union in cls.union_output[:-1]]

def kmer():
    #1.Generating the k-mer lists for input samples
    #bash glistmaker

    #2.Generating the k-mer feature vector.


    #3.Mapping samples to the feature vector space
    # glistquery, split






def extract_info(s,k,canonical,n_jobs):
    '''extract feature w.r.t. each species and drug
    k: k mer feature
    '''

    # data = pd.read_csv('metadata/list_species_final_bq.txt', index_col=0,dtype={'genome_id': object}, sep="\t")
    data = pd.read_csv('metadata/loose_Species_antibiotic_FineQuality.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    # data=data[data['number'] != 0]#drop the species with 0 in column 'number'.# quality controled data file Species_antibiotic_FineQuality.csv

    #delete later:
    #data=data.loc[['Escherichia coli'],:]
    #data = data.loc[['Pseudomonas aeruginosa', 'Escherichia coli'], :]
    #-------------
    if s!=[]:
        data = data.loc[s, :]
    df_species = data.index.tolist()

    # antibiotics=data['modelling antibiotics'].tolist()
    print(data)
    print(k,canonical)

    vocab=amr_utility.math_utility.vocab_build(canonical,k)# all the kmer combinations in list

    pool = mp.Pool(processes=n_jobs)
    pool.starmap(kmer, zip(df_species,repeat(k),repeat(canonical),repeat(vocab)))
    #kmer('Pseudomonas aeruginosa',str(['ceftazidime', 'ciprofloxacin', 'meropenem']),k,canonical)


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k',default=13, type=int, required=True,
                        help='Kmer size')
    parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    # parser.add_argument('--a','--add', dest='odh',
    #                     help='Construct feature matrix with more samples, w.r.t. kmer features.',action='store_true',)#default:false
    parser.add_argument('-c','--canonical', dest='canonical',action='store_true',
                        help='Canonical kmer or not: True')
    parser.add_argument('-n','--non_canonical',dest='canonical',action='store_false',
                        help='Canonical kmer or not: False')

    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
                \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
                \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('-l', '--level', default='loose', type=str,
                        help='Quality control: strict or loose')
    parser.add_argument('-cutoff', '--cutoff', default=1, type=int,
                        help='specify frequency cut-off (default 1) for kmer counting. ')

    #parser.set_defaults(canonical=True)
    # parser.print_help()
    parsedArgs=parser.parse_args()
    print(parsedArgs)
    extract_info(parsedArgs.species,parsedArgs.k,parsedArgs.canonical,parsedArgs.n_jobs)


