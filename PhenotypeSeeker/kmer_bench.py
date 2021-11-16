#!/usr/bin/python
# @Author: kxh
# @Date:   Nov 8 2021
# @Last Modified by:
# @Last Modified time:
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
import amr_utility.file_utility
import amr_utility.load_data
# from amr_utility import name_utility
import warnings
import argparse
import os
import logging
import math





def get_kmer(level,log_addr,k, species,antibiotics, n_jobs):
    # 1.Generating the k-mer lists for every samples for each species
    #using scripts named kmer.sh
    # glistmaker
    # glistquery

    # 2. Mapping samples to the feature vector space
    antibiotics_selected = ast.literal_eval(antibiotics)
    print(species)
    print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)
    antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
    #
    # antibiotics, ID, Y = antibiotics[5:], ID[5:], Y[5:]
    i_anti = 0
    for anti in antibiotics:
        vocab = amr_utility.math_utility.vocab_build(False, k)  # all the kmer combinations in list
        data = np.zeros((len(vocab), 1), dtype='uint16')
        feature_matrix = pd.DataFrame(data, index=vocab, columns=['initializer'])  # delete later
        feature_matrix.index.name = 'feature'

        _, meta_temp,_=amr_utility.name_utility.Pts_GETname(level, species, anti,'')
        # l = 0
        for i in ID[i_anti]: #todo check
            print(i)
            # l += 1
            # if (l % 1000 == 0):
            #     print(l, species)
            # map feature txt from stored data to feature matrix.
            f = pd.read_csv('K-mer_lists/'+str(i)+'_mapped',
                            names=['combination', str(i)], dtype={'genome_id': object}, sep="\t")
            # print('f.shape')
            # print(f.shape)
            f = f.set_index('combination')
            feature_matrix = pd.concat([feature_matrix, f], axis=1)
        i_anti+=1
        feature_matrix = feature_matrix.drop(['initializer'], axis=1)# delete initializer column
        feature_matrix = feature_matrix.fillna(0)

        # feature_matrix = feature_matrix.T
        # print(feature_matrix)
        # feature_matrix.to_hdf(meta_temp+'_'+str(k)+'mer.h5', key='df', mode='w', complevel=9)  # overwriting.
        print(feature_matrix)
        exit()



def extract_info(s,level,k,n_jobs):
    '''extract feature w.r.t. each species and drug
    k: k mer feature
    '''

    data = pd.read_csv('metadata/' + str(level) + '_Species_antibiotic_FineQuality.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    # data = pd.read_csv('metadata/Species_antibiotic_FineQuality.csv', index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    # for training model on part of the dataset:-------------
    # data=data.loc[['Escherichia coli'],:]
    data = data.loc[s, :]
    # data.at['Mycobacterium tuberculosis', 'modelling antibiotics']=['capreomycin', 'ciprofloxacin']
    # --------------------------------------------------------
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()

    fileDir = os.path.dirname(os.path.realpath('__file__'))
    for species, antibiotics in zip(df_species, antibiotics):
        log_addr=fileDir+'/log/temp/' + str(level) + '/' + str(species.replace(" ", "_"))
        amr_utility.file_utility.make_dir(log_addr)
        get_kmer(level,log_addr,k, species,antibiotics, n_jobs)


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k',default=13, type=int, required=False,
                        help='Kmer size')
    parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    # parser.add_argument('--a','--add', dest='odh',
    #                     help='Construct feature matrix with more samples, w.r.t. kmer features.',action='store_true',)#default:false
    # parser.add_argument('-c','--canonical', dest='canonical',action='store_true',
    #                     help='Canonical kmer or not: True')
    # parser.add_argument('-n','--non_canonical',dest='canonical',action='store_false',
    #                     help='Canonical kmer or not: False')

    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
                \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
                \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('-l', '--level', default='loose', type=str,
                        help='Quality control: strict or loose')
    # parser.add_argument('-cutoff', '--cutoff', default=1, type=int,
    #                     help='specify frequency cut-off (default 1) for kmer counting. ')

    #parser.set_defaults(canonical=True)
    # parser.print_help()
    parsedArgs=parser.parse_args()
    print(parsedArgs)
    extract_info(parsedArgs.species,parsedArgs.level,parsedArgs.k,parsedArgs.n_jobs)


