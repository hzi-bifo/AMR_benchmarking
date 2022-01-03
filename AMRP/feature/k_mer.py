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
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import confusion_matrix
import subprocess
import multiprocessing as mp
import amr_utility.name_utility
import amr_utility.math_utility
# from amr_utility import name_utility
import warnings
import argparse
import os
import logging
import sys




def kmer(species,antibiotics,k,canonical,vocab):

    '''
    summerise antibiotic
    data_sub: data w.r.t. selected species , with selected antibiotics(>1000 strains).
    data_sub_uniqueID: unique ID on top of data_sub
    Output: saved h5 files of kmer features, w.r.t. each species.
    '''

    antibiotics_selected = ast.literal_eval(antibiotics)
    print(species)
    print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)
    data_sub=[]#initialize the dataframe for kmer extraction.
    save_mame_kmc,save_name_kmer=amr_utility.name_utility.save_name_kmer(species,canonical,k)
    for anti in antibiotics_selected: #make the kmer matrix for all the antibiotic selected for one species.
        save_name_meta,save_name_modelID,save_name_val=amr_utility.name_utility.GetSaveName(species,anti,canonical,k)
        data_sub_anti=pd.read_csv(save_name_modelID + '.txt', index_col=0,dtype={'genome_id': object},sep="\t")
        data_sub.append(data_sub_anti)
    data_sub=pd.concat(data_sub)
    data_sub_uniqueID = data_sub.groupby(by="genome_id").count()
    ID = data_sub_uniqueID.index.tolist()
    print('Number of strains:', len(ID))

    # initialize!
    # k mer features
    data = np.zeros((len(vocab), 1), dtype='uint16')
    # data = np.zeros((len(vocab), len(id)), dtype='uint16')
    feature_matrix = pd.DataFrame(data, index=vocab, columns=['initializer'])  # delete later
    # feature_matrix = pd.DataFrame(data, index=vocab, columns=data_sub['genome_id'])
    feature_matrix.index.name = 'feature'
    # print(feature_matrix)
    l = 0
    for i in ID:
        l += 1
        if (l % 1000 == 0):
            print(l,species)
        # map feature txt from stored data(KMC tool processed) to feature matrix.
        f = pd.read_csv(save_mame_kmc + str(i) + '.txt',
                        names=['combination', str(i)],dtype={'genome_id': object}, sep="\t")
        #print('f.shape(just check)!!!!!!!!!!')
        #print(f.shape)

        f = f.set_index('combination')
        # feature_matrix.update(f)
        feature_matrix = pd.concat([feature_matrix, f], axis=1)
        # print(feature_matrix[str(i)])
        # print(feature_matrix)
    # delete initializer column and save the feature matrix
    # make feature matrix 65536*n
    feature_matrix = feature_matrix.drop(['initializer'], axis=1)
    # print(feature_matrix)
    # nan_values = feature_matrix[feature_matrix[ID[0]].isnull()]
    # print('just check nan values..',nan_values)#check. select nan data
    # replace nan data with o.
    feature_matrix = feature_matrix.fillna(0)

    #feature_matrix = feature_matrix.T
    #print(feature_matrix)
    feature_matrix.to_hdf(save_name_kmer, key='df', mode='w', complevel=9)#overwriting.
    print(feature_matrix)
    # feature_matrix.to_scv.t(save_name_kmer+'.txt', sep="\t")
    # return feature_matrix
    # ==================================================================================================




def extract_info(k,canonical):
    '''extract feature w.r.t. each species and drug
    k: k mer feature
    '''

    data = pd.read_csv('metadata/Species_antibiotic_FineQuality.csv', index_col=0,dtype={'genome_id': object}, sep="\t")

    data=data[data['number'] != 0]#drop the species with 0 in column 'number'.

    #delete later:
    #data=data.loc[['Escherichia coli'],:]
    #data = data.loc[['Pseudomonas aeruginosa', 'Escherichia coli'], :]
    #-------------

    df_species=data.index.tolist()
    antibiotics=data['modelling antibiotics'].tolist()
    print(data)
    print(k,canonical)

    vocab=amr_utility.math_utility.vocab_build(canonical,k)# all the kmer combinations in list

    pool = mp.Pool(processes=8)
    pool.starmap(kmer, zip(df_species,antibiotics,repeat(k),repeat(canonical),repeat(vocab)))
    #kmer('Pseudomonas aeruginosa',str(['ceftazidime', 'ciprofloxacin', 'meropenem']),k,canonical)


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k',default=8, type=int, required=True,
                        help='Kmer size')
    parser.add_argument('-c','--canonical', dest='canonical',action='store_true',
                        help='Canonical kmer or not: True')
    parser.add_argument('-n','--non_canonical',dest='canonical',action='store_false',
                        help='Canonical kmer or not: False')
    #parser.set_defaults(canonical=True)
    parsedArgs=parser.parse_args()
    print(parsedArgs)
    extract_info(parsedArgs.k,parsedArgs.canonical)
