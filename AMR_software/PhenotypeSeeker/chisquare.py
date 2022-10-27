#!/usr/bin/env/python
__author__ = "Erki Aun"
__version__ = "0.7.3"
# @Last Modified by:   Khu
# @Last Modified time:
import sys
import os
# sys.path.append('../')
sys.path.insert(0, os.getcwd())
import pandas as pd
import numpy as np
from src.amr_utility import name_utility
import argparse
from scipy import stats
import collections





def conduct_chi_squared_test( meta_txt,cv,N_samples,kmer, kmer_presence, test_results_file,dic_pheno,names_of_samples_list):#only for trainng set.
    samples_w_kmer = []  #
    ( w_pheno_w_kmer, w_pheno_wo_kmer, wo_pheno_w_kmer, wo_pheno_wo_kmer,
        no_samples_wo_kmer) =  get_samples_distribution_for_chisquared(meta_txt,cv,kmer_presence, samples_w_kmer,dic_pheno,names_of_samples_list)


    no_samples_w_kmer = len(samples_w_kmer)
    min_samples=2#all met.
    max_samples=N_samples-2#max number of samples to consider a kmer in the ML

    if no_samples_w_kmer <  min_samples or no_samples_wo_kmer < 2  or no_samples_w_kmer >  max_samples:

        return None
    # else:
        # print('Not filtered.')
    (w_pheno, wo_pheno, w_kmer, wo_kmer, total) = get_totals_in_classes(
        w_pheno_w_kmer, w_pheno_wo_kmer, wo_pheno_w_kmer, wo_pheno_wo_kmer
    )

    (
        w_pheno_w_kmer_expected, w_pheno_wo_kmer_expected,
        wo_pheno_w_kmer_expected, wo_pheno_wo_kmer_expected
    ) = get_expected_distribution(
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


def get_samples_distribution_for_chisquared(meta_txt, cv,kmers_presence_vector, samples_w_kmer,dic_pheno,names_of_samples_list):
    no_samples_wo_kmer = 0
    with_pheno_with_kmer = 0
    with_pheno_without_kmer = 0
    without_pheno_with_kmer = 0
    without_pheno_without_kmer = 0

    weight_dic = np.load(meta_txt+'_temp/CV_tr'+str(cv)+'/weight.npy', allow_pickle='TRUE').item()



    for index, sample in enumerate(names_of_samples_list):

        if dic_pheno[sample] == 1:#phenotye
            # print(index, kmers_presence_vector[index])
            if (kmers_presence_vector[index] != "0"):#kmer presence
                with_pheno_with_kmer += weight_dic[sample]
                samples_w_kmer.append(sample)
            else:  #
                with_pheno_without_kmer += weight_dic[sample]
                no_samples_wo_kmer += 1
        elif dic_pheno[sample] == 0:#
            if (kmers_presence_vector[index] != "0"):
                without_pheno_with_kmer += weight_dic[sample]
                samples_w_kmer.append(sample)
            else:
                without_pheno_without_kmer += weight_dic[sample]
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

def get_expected_distribution(w_pheno, wo_pheno, w_kmer, wo_kmer, total):
    w_pheno_w_kmer_expected = ((w_pheno * w_kmer)/ float(total))
    w_pheno_wo_kmer_expected = ((w_pheno * wo_kmer)/ float(total))
    wo_pheno_w_kmer_expected  = ((wo_pheno * w_kmer)/ float(total))
    wo_pheno_wo_kmer_expected = ((wo_pheno * wo_kmer)/ float(total))
    return(
        w_pheno_w_kmer_expected, w_pheno_wo_kmer_expected,
        wo_pheno_w_kmer_expected, wo_pheno_wo_kmer_expected
        )

def filter_kmer_tr(level,species,anti, cv,f_phylotree,f_kma,temp_path):
    #for training set.
    #building kmer_presence_vector in the form of : kmer as index, the other columns are sample ID

    _,_,meta_txt,_ = name_utility.GETname_model2('phenotypeseeker',level, species, anti,'',temp_path,f_kma,f_phylotree)
    names_of_samples_list = np.genfromtxt(meta_txt + '_Train_' + str(cv) + '_id2', dtype="str")
    names_of_samples_list = names_of_samples_list.tolist()
    N_samples=len(names_of_samples_list)
    vectors_as_multiple_input=[]
    for sample in names_of_samples_list:
        vectors_as_multiple_input.append(meta_txt+'_temp/CV_tr'+str(cv)+'/'+sample+'_mapped' )

    Chi_results_file = open(meta_txt+'_temp/CV_tr'+str(cv)+ "/chi-squared_test_results.txt", "w")

    #preparing for pheno infor.
    dic_pheno = collections.defaultdict(list)  # Note: the order are not the same as final matrix.
    pheno_matrix =pd.read_csv(meta_txt+ '_Train_' + str(cv) + '_data.pheno', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    for sample in names_of_samples_list:

        pheno=pheno_matrix[pheno_matrix['genome_id'] == str(sample)].iat[0,1]


        dic_pheno[sample]=pheno


    for line in zip(*[open(item) for item in vectors_as_multiple_input]):

        kmer_count = line[0].split()[0]
        kmer_presence_vector = [j.split()[1].strip() for j in line]#in the same order as kmer_count
        # print(kmer_count)
        # print(kmer_presence_vector)
        conduct_chi_squared_test(meta_txt,cv,N_samples,
            kmer_count, kmer_presence_vector,
            Chi_results_file,dic_pheno,names_of_samples_list)
    Chi_results_file.close()






def extract_info(s,anti,level,cv,f_phylotree,f_kma,temp_path):
    '''extract feature w.r.t. each species and drug
    k: k mer feature
    '''
    s = [str(i.replace("_", " ")) for i in s]
    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    data = data.loc[s, :]
    # data.at['Mycobacterium tuberculosis', 'modelling antibiotics']=['capreomycin', 'ciprofloxacin']
    # --------------------------------------------------------
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()

    for species, antibiotics in zip(df_species, antibiotics):

        print(species,anti)
        filter_kmer_tr(level,species,anti,cv,f_phylotree,f_kma,temp_path)


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                        help='Directory to store temporary files.')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                        help='kma based cv folders.')
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


    #parser.set_defaults(canonical=True)
    # parser.print_help()
    parsedArgs=parser.parse_args()

    extract_info(parsedArgs.species,parsedArgs.anti,parsedArgs.level,parsedArgs.cv,parsedArgs.f_phylotree,parsedArgs.f_kma,parsedArgs.temp_path)


