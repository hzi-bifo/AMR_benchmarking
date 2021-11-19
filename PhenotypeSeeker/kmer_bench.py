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


# def prepare_meta(level,log_addr,k, species,antibiotics, n_jobs):#todo
#     antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
#     ALL = []
#     for anti in antibiotics:
#         name, path, _, _, _, _, _ = amr_utility.name_utility.s2g_GETname(level, species, anti)
#         name_list = pd.read_csv(name, index_col=0, dtype={'genome_id': object}, sep="\t")
#         if Path(fileDir).parts[1] == 'vol':
#             # path_list=np.genfromtxt(path, dtype="str")
#             name_list['path'] = '/vol/projects/BIFO/patric_genome/' + name_list['genome_id'].astype(str) + '.fna'
#         else:
#             name_list['path'] = '/net/projects/BIFO/patric_genome/' + name_list['genome_id'].astype(str) + '.fna'
#         name_list['genome_id'] = 'iso_' + name_list['genome_id'].astype(str)
#
#         name_list['path_pseudo'] = pseudo
#         ALL.append(name_list)
#         # print(name_list)
#     _, _, dna_list, assemble_list, yml_file, run_file_name, wd = amr_utility.name_utility.s2g_GETname(level, species,
#                                                                                                       '')
#
#     # combine the list for all antis
#     species_dna = ALL[0]
#     for i in ALL[1:]:
#         species_dna = pd.merge(species_dna, i, how="outer",
#                                on=["genome_id", 'path', 'path_pseudo'])  # merge antibiotics within one species
#     print(species_dna)
#     species_dna_final = species_dna.loc[:, ['genome_id', 'path']]
#     species_dna_final.to_csv(assemble_list, sep="\t", index=False, header=False)
#     species_pseudo = species_dna.loc[:, ['genome_id', 'path_pseudo']]
#     species_pseudo.to_csv(dna_list, sep="\t", index=False, header=False)



def get_kmer(level,log_addr,k, species,antibiotics, n_jobs):
    # 1.Generate the k-mer lists for every samples for each species
    #using scripts named kmer.sh
    # glistmaker


    # 2. generate unions of k mer and then Map samples to the feature vector space
    # get_mash_distances, perfrom chi-square test.
    # map.sh <species>
    # e.g. map.sh "Campylobacter_jejuni"

    # 3.
    antibiotics_selected = ast.literal_eval(antibiotics)
    print(species)


    print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)
    antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
    # _, meta_temp,_=amr_utility.name_utility.Pts_GETname(level, species, anti,'')#todo, may use this in the future.
    i_anti = 0
    for anti in antibiotics:
        id_list=ID[i_anti]
        i_anti+=1





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
        prepare_meta(level,log_addr,k, species,antibiotics, n_jobs)
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


