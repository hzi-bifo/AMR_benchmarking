#!/usr/bin/env/python
__author__ = "Erki Aun"
__version__ = "0.7.3"
# @Last Modified by:   Khu
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
from scipy import stats
import collections
import math
# Functions for filtering the k-mers based on the p-values of
    # conducted tests.

def read_pvalue(chi_result):
    # read in pvalues from chi_result
    pvalues_list=[]
    inputfile = open(chi_result)
    counter=0
    for line in inputfile:
        counter += 1
        line_to_list = line.split()
        p_val =float(line_to_list[2])
        pvalues_list.append(p_val)
    inputfile.close()
    return pvalues_list

def get_kmers_filtered(level,species,anti,cv, n_jobs):
    # Filters the k-mers by their p-value achieved in statistical
    kmer_limit=1000 #default setting in the Phenotypeseeker "0.7.3"
    pvalue_cutoff=0.05 #default setting in the Phenotypeseeker "0.7.3"
    _, meta_txt, _ = amr_utility.name_utility.Pts_GETname(level, species, anti, '')
    chi_result=meta_txt+'_temp/CV_tr'+str(cv)+'/chi-squared_test_results.txt'
    kmers_for_ML={}
    names_of_samples_list = np.genfromtxt(meta_txt + '_Train_' + str(cv) + '_id2', dtype="str")
    names_of_samples_list = names_of_samples_list.tolist()
    # read in pvalues from chi_result
    pvalues_list=read_pvalue(chi_result)
    pvalues_list.sort()
    # nr_of_kmers_tested = float(len(pvalues_list))
    # self.get_pvalue_cutoff(self.pvalues, nr_of_kmers_tested)
    pval_limit = float('{:.2e}'.format(pvalues_list[kmer_limit]))
    del pvalues_list


    inputfile = open(chi_result) #"chi-squared_test_results_"
    outputfile = open(meta_txt+'_temp/CV_tr'+str(cv)+'/k-mers_filtered_by_pvalue.txt', "w")# Truncate file to zero length or create text file for writing.
    outputfile.write(
    "K-mer\tWelch's_t-statistic\tp-value\t+_group_mean\
    \t-_group_mean\tNo._of_samples_with_k-mer\
    \tSamples_with_k-mer\n"
    )


    counter = 0
    unique_patterns = set()
    drop_collinearity = False
    for line in inputfile:
        counter += 1
        line_to_list = line.split()
        kmer, p_val = line_to_list[0], float(line_to_list[2])
        samples_with_kmer = set(line.split("|")[1].split())



        if p_val <  pvalue_cutoff:
            outputfile.write(line)
            # if p_val in pvalues_for_ML_kmers:
            if drop_collinearity:#false
                # if line.split("|")[1] not in unique_patterns:
                #     unique_patterns.add(line.split("|")[1])
                #     kmers_for_ML[kmer] = [
                #         1 if sample in samples_with_kmer else 0 for sample in names_of_samples_list
                #         ] + [p_val]
                pass
            else:

                if p_val <= pval_limit:
                    kmers_for_ML[kmer] = [
                            1 if sample in samples_with_kmer else 0 for sample in names_of_samples_list
                            ] + [p_val]


        # else:#
        #     # names_of_samples_list = np.genfromtxt(meta_txt + '_Test_' + str(cv) + '_id2', dtype="str")
        #     # names_of_samples_list = names_of_samples_list.tolist()
        #     # outputfile.write(line)
        #     # kmers_for_ML[kmer] = [1 if sample in samples_with_kmer else 0 for sample in names_of_samples_list
        #     #                           ] + [p_val]
        #     pass





    if len(kmers_for_ML) == 0:
        print("warning: No k-mers passed the filtration by p-value.")
    inputfile.close()
    outputfile.close()
    return kmers_for_ML



def get_ML_dataframe(kmers_for_ML,level,species,anti,cv, n_jobs):
    kmer_limit=1000
    _, meta_txt, _ = amr_utility.name_utility.Pts_GETname(level, species, anti, '')
    names_of_samples_list = np.genfromtxt(meta_txt + '_Train_' + str(cv) + '_id2', dtype="str")
    names_of_samples_list = names_of_samples_list.tolist()
    index = list(names_of_samples_list + ['p_val'])
    ML_df = pd.DataFrame( kmers_for_ML, index=index)
    ML_df.index.name = 'genome_id'
    ML_df.sort_values('p_val', axis=1, ascending=True, inplace=True)
    ML_df = ML_df.iloc[:, :kmer_limit]

    ML_df.drop('p_val', inplace=True)

    # preparing for pheno infor.
    dic_pheno = collections.defaultdict(list)  # Note: the order are not the same as final matrix.
    pheno_matrix = pd.read_csv(meta_txt + '_Train_' + str(cv) + '_data.pheno', index_col=0,
                               dtype={'genome_id': object}, sep="\t")
    for sample in names_of_samples_list:
        pheno = pheno_matrix[pheno_matrix['genome_id'] == str(sample)].iloc[0][anti]
        dic_pheno[sample] = pheno
    # ---------------------------------

    ML_df['phenotype'] = [dic_pheno[sample] for sample in names_of_samples_list]
    # ML_df['weight'] = [sample.weight for sample in names_of_samples_list]#todo, maybe needed in the future. but not so far.
    # ML_df = ML_df.loc[ ML_df.phenotype != 'NA']
    ML_df.phenotype =  ML_df.phenotype.apply(pd.to_numeric)
    ML_df.to_csv(meta_txt + "_" + str(cv) + '_Train_df.csv',sep="\t")
    print(ML_df)


def get_ML_dataframe_testset(level,species,anti,cv):
    #construct directly from the glistquery output.

    _, meta_temp, _ = amr_utility.name_utility.Pts_GETname(level, species, anti, '')
    # vocab = amr_utility.math_utility.vocab_build(False, 13)  # all the kmer combinations in list
    # Initalization using the training set with already selected kmers.
    vocab_df=pd.read_csv(meta_temp + "_" + str(cv) + '_Train_df.csv', dtype={'genome_id': object}, sep="\t")
    vocab_df = vocab_df.set_index('genome_id')

    vocab=vocab_df.columns.values



    data = np.zeros((len(vocab), 1), dtype='uint16')
    feature_matrix = pd.DataFrame(data, index=vocab, columns=['initializer'])  # delete later
    feature_matrix.index.name = 'feature'
    print(feature_matrix)

    names_of_samples_list = np.genfromtxt(meta_temp + '_Test_' + str(cv) + '_id2', dtype="str")
    names_of_samples_list = names_of_samples_list.tolist()

    for i in names_of_samples_list:
        # map feature txt from stored data to feature matrix.
        f = pd.read_csv(meta_temp+'_temp/CV_te'+str(cv)+'/'+str(i)+'_mapped',
                        names=['combination', str(i)], sep="\t")
        # print(f)
        # print(f.shape)
        # print(f.shape)
        f = f.set_index('combination')
        feature_matrix = pd.concat([feature_matrix, f], axis=1, join="inner")

    feature_matrix = feature_matrix.drop(['initializer'], axis=1)  # delete initializer column
    # feature_matrix = feature_matrix.fillna(0)
    # feature_matrix.loc[~feature_matrix.isnull()] = 1
    # feature_matrix.loc[feature_matrix.isnull()] = 0
    #replace all the values >1 with 1.
    feature_matrix[feature_matrix > 0] = 1
    feature_matrix = feature_matrix.T
    feature_matrix.index.name = 'genome_id'
    feature_matrix.to_csv(meta_temp + "_" + str(cv) + '_Test_df.csv')
    print(feature_matrix)

def extract_info(s,anti,level,cv,f_test,n_jobs):
    '''extract feature w.r.t. each species and drug
    k: k mer feature
    '''
    s = [str(i.replace("_", " ")) for i in s]
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
        # antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
        # i_anti = 0
        # for anti in antibiotics:
        if f_test:
            get_ML_dataframe_testset(level,species,anti,cv)

        else:
            kmers_for_ML=get_kmers_filtered(level,species,anti, cv, n_jobs)
            get_ML_dataframe(kmers_for_ML,level,species,anti,cv, n_jobs)

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--k',default=13, type=int, required=True,
    #                     help='Kmer size')
    parser.add_argument('-anti', '--anti', type=str, required=True,
                        help='antibiotics.')
    parser.add_argument('--n_jobs', default=1, type=int, required=False, help='Number of jobs to run in parallel.')
    parser.add_argument("-cv", "--cv", default=0, type=int, required=True,
                        help='CV splits you are working now')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
                \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
                \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('-l', '--level', default='loose', type=str,
                        help='Quality control: strict or loose')
    parser.add_argument('-f_test', '--f_test', dest='f_test', action='store_true',
                        help='Filter the testing set according to training set; otherwise, filter the training set '
                             'according to the Chi-squared test.')

    #parser.set_defaults(canonical=True)
    # parser.print_help()
    parsedArgs=parser.parse_args()
    print(parsedArgs)
    extract_info(parsedArgs.species,parsedArgs.anti,parsedArgs.level,parsedArgs.cv,parsedArgs.f_test,parsedArgs.n_jobs)