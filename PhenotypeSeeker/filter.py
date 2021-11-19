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
# from amr_utility import name_utility
import warnings
import argparse
import os
from scipy import stats
import logging
import math


def conduct_chi_squared_test( kmer, kmer_presence, test_results_file, samples):
    samples_w_kmer = []  #
    (
        w_pheno_w_kmer, w_pheno_wo_kmer, wo_pheno_w_kmer, wo_pheno_wo_kmer,
        no_samples_wo_kmer
    ) = self.get_samples_distribution_for_chisquared(
        kmer_presence, samples_w_kmer, samples
    )
    no_samples_w_kmer = len(samples_w_kmer)
    if self.pvalue_cutoff == 1:  # added by khu. Oct 5th. This is only for testind set in a extreme case.
        pass
    else:
        if no_samples_w_kmer < Samples.min_samples or no_samples_wo_kmer < 2 \
                or no_samples_w_kmer > Samples.max_samples:
            # print('1:',no_samples_w_kmer)#1,1099
            # print('2:',Samples.max_samples)#1097
            # print('3:',Samples.min_samples)#2
            # print('4:',no_samples_wo_kmer)#1098,0
            # print('sample number issues.from khu.')
            return None
    (w_pheno, wo_pheno, w_kmer, wo_kmer, total) = self.get_totals_in_classes(
        w_pheno_w_kmer, w_pheno_wo_kmer, wo_pheno_w_kmer, wo_pheno_wo_kmer
    )

    (
        w_pheno_w_kmer_expected, w_pheno_wo_kmer_expected,
        wo_pheno_w_kmer_expected, wo_pheno_wo_kmer_expected
    ) = self.get_expected_distribution(
        w_pheno, wo_pheno, w_kmer, wo_kmer, total)
    chisquare_results = stats.chisquare(
        [
            w_pheno_w_kmer, w_pheno_wo_kmer,
            wo_pheno_w_kmer, wo_pheno_wo_kmer
        ],
        [
            w_pheno_w_kmer_expected, w_pheno_wo_kmer_expected,
            wo_pheno_w_kmer_expected, wo_pheno_wo_kmer_expected
        ],
        1
    )

    test_results_file.write(
        kmer + "\t%.2f\t%.2E\t" % chisquare_results
        + str(no_samples_w_kmer) + "\t| " + " ".join(samples_w_kmer) + "\n"
    )
    pvalue = chisquare_results[1]
    return pvalue


def get_samples_distribution_for_chisquared( kmers_presence_vector, samples_w_kmer, samples):
    no_samples_wo_kmer = 0
    with_pheno_with_kmer = 0
    with_pheno_without_kmer = 0
    without_pheno_with_kmer = 0
    without_pheno_without_kmer = 0
    for index, sample in enumerate(samples):
        # print(index, sample)#
        # print(kmers_presence_vector)#
        # print('-------------')
        if sample.phenotypes[self.name] == 1:
            if (kmers_presence_vector[index] != "0"):
                with_pheno_with_kmer += sample.weight
                samples_w_kmer.append(sample.name)
            else:  #
                with_pheno_without_kmer += sample.weight
                no_samples_wo_kmer += 1
        elif sample.phenotypes[self.name] == 0:
            if (kmers_presence_vector[index] != "0"):
                without_pheno_with_kmer += sample.weight
                samples_w_kmer.append(sample.name)
            else:
                without_pheno_without_kmer += sample.weight
                no_samples_wo_kmer += 1
    return (
        with_pheno_with_kmer, with_pheno_without_kmer,
        without_pheno_with_kmer, without_pheno_without_kmer,
        no_samples_wo_kmer
    )
def get_totals_in_classes(w_pheno_w_kmer, w_pheno_wo_kmer, wo_pheno_w_kmer, wo_pheno_wo_kmer):
    w_pheno = (w_pheno_w_kmer + w_pheno_wo_kmer)
    wo_pheno = (wo_pheno_w_kmer + wo_pheno_wo_kmer)
    w_kmer = (w_pheno_w_kmer + wo_pheno_w_kmer)
    wo_kmer = (w_pheno_wo_kmer + wo_pheno_wo_kmer)
    total = w_pheno + wo_pheno
    return w_pheno, wo_pheno, w_kmer, wo_kmer, total


def filter_kmer_tr(level, k, species,anti, cv,n_jobs):
    #building kmer_presence_vector in the form of : kmer as index, the other columns are sample ID
    _, meta_txt, _ = amr_utility.name_utility.Pts_GETname(level, species, anti, '')
    names_of_samples_list = np.genfromtxt(meta_txt + '_Test_' + str(cv) + '_id2', dtype="str")
    for
        for line in :






def extract_info(s,level,cv,f_test,n_jobs):
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
    for species, anti in zip(df_species, antibiotics):
        if f_test:
            pass
        else:
            filter_kmer_tr(level,species,anti, n_jobs)#todo


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--k',default=13, type=int, required=True,
    #                     help='Kmer size')
    parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
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
    extract_info(parsedArgs.species,parsedArgs.level,parsedArgs.cv,parsedArgs.f_test,parsedArgs.n_jobs)


