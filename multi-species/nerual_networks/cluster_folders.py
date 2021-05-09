#!/usr/bin/python
#Python 3.6
#nested CV, modified by khu based on Derya Aytan's work.
#https://bitbucket.org/deaytan/neural_networks/src/master/Neural_networks.py
#Note: this scritp should be used as a module, otherwise the storage will be disrupted.
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import sys
sys.path.append('../')
sys.path.insert(0, os.getcwd())

import numpy as np
import getopt
import sys
import warnings
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,confusion_matrix,classification_report,f1_score
import collections
import random
from sklearn import utils
from sklearn import preprocessing
import argparse
import ast
import amr_utility.name_utility
import itertools
import statistics

import pickle
import copy

def cluster_split_old(dict_cluster, Random_State, cv):
    # Custom k fold cross validation
    # cross validation method divides the clusters and adds to the partitions.
    # Samples are not divided directly due to sample similarity issue
    all_data_splits_pre = []  # cluster order
    all_data_splits = []  # cluster order
    all_available_data = range(0, len(dict_cluster))  # all the clusters had
    clusters_n = len(dict_cluster)  # number of clusters
    all_samples = []  # all the samples had in the clusters
    for i in dict_cluster:
        for each in dict_cluster[i]:
            all_samples.append(each)

    # Shuffle the clusters and divide them
    shuffled = list(utils.shuffle(list(dict_cluster.keys()),
                                  random_state=Random_State))  # shuffled cluster names

    # Divide the clusters equally
    r = int(len(shuffled) / cv)  # batches size,e.g. 105/5=21

    a = 0
    b = r
    for i in range(cv):  # 5

        all_data_splits_pre.append(shuffled[a:b])

        if i != cv - 2:  # 1 =0,1,2,4
            a = b
            b = b + r

        else:
            a = b
            b = len(shuffled)

    # Extract the samples inside the clusters
    # If the number of samples are lower than the expected number of samples per partition, get an extra cluster

    totals = []
    all_extra = []
    for i in range(len(all_data_splits_pre)):  # 5.#cluster order
        if i == 0:
            m_fromprevious = i + 1  # 1
        else:
            m_fromprevious = np.argmax(totals)  # the most samples CV index
        tem_Nsamples = []  # number of samples in the selected clusters
        extracted = list(set(all_data_splits_pre[i]) - set(all_extra))  # order of cluster, w.r.t dict_cluster

        for e in extracted:
            elements = dict_cluster[str(e)]
            tem_Nsamples.append(len(elements))
        sum_tem = sum(tem_Nsamples)

        print('sum_tem', sum_tem)  # all_samples in that folder: val,train,test.
        print('average',len(all_samples) / float(cv))
        print('len(all_samples)',len(all_samples))
        a = 0
        while sum_tem + 200 < len(all_samples) / float(cv):  # all_samples: val,train,test
            extra = list(utils.shuffle(
                all_data_splits_pre[m_fromprevious], random_state=Random_State))[a]  # cluster order
            extracted = extracted + [extra]  # cluster order
            all_extra.append(extra)
            a = a + 1
            tem_Nsamples = []
            print('extracted', extracted)
            for e in extracted:  # every time add one cluster order
                elements = dict_cluster[str(e)]
                tem_Nsamples.append(len(elements))  # sample number
            sum_tem = sum(tem_Nsamples)
            for item in range(len(all_data_splits)):
                all_data_splits[item] = list(
                    set(all_data_splits[item]) - set([extra]))  # rm previous cluster order, because it's moved to this fold.
        totals.append(sum_tem)#wrong totals
        all_data_splits.append(extracted)  ##cluster order for each CV
    return all_data_splits


def cluster_split(dict_cluster, Random_State, cv):#khu: modified
    # Custom k fold cross validation
    # cross validation method divides the clusters and adds to the partitions.
    # Samples are not divided directly due to sample similarity issue
    # all_data_splits_pre = []  # cluster order
    all_data_splits = []  # cluster order
    all_available_data = range(0, len(dict_cluster))  # all the clusters had
    clusters_n = len(dict_cluster)  # number of clusters
    all_samples = []  # all the samples had in the clusters
    for i in dict_cluster:
        for each in dict_cluster[i]:
            all_samples.append(each)

    # Shuffle the clusters and divide them
    shuffled = list(utils.shuffle(list(dict_cluster.keys()),
                                  random_state=Random_State))  # shuffled cluster names

    # Divide the clusters equally
    r = int(len(shuffled) / cv)  # batches size,e.g. 105/5=21

    a = 0
    b = r
    for i in range(cv):  # 5

        all_data_splits.append(shuffled[a:b])

        if i != cv - 2:  # 1 =0,1,2,4
            a = b
            b = b + r

        else:
            a = b
            b = len(shuffled)


    totals = []
    for folder_cluster in all_data_splits:  # 5.#cluster order
        tem=[]
        for e in folder_cluster:
            elements = dict_cluster[str(e)]
            tem.append(len(elements))
        totals.append(sum(tem))
    print('all samples',sum(totals),totals,len(all_samples) / float(cv))


    # all_data_splits_pre=copy.deepcopy(all_data_splits)
    print('Re_sampling...........')

    for i in range(len(all_data_splits)):  # 5.#cluster order

        extracted = list(set(all_data_splits[i]))  # order of cluster, w.r.t dict_cluster
        sum_sub = totals[i]
        print(sum_sub)
        print('totals====================================',totals)
        # print(all_data_splits)
        b=0
        while (sum_sub < 0.2*(len(all_samples) / float(cv))) and b<100:  # all_samples: val,train,test
            b+=1

            m_from = np.argmax(totals)  # the most samples CV index
            extra = list(utils.shuffle(all_data_splits[m_from], random_state=Random_State))[0]  # cluster order

            a = 0
            while len(dict_cluster[extra]) >= 1.0*(len(all_samples) / float(cv)) and a < 5:  # in case one cluster contain a lot of samples
                m_from = np.argmax(totals)  # the most samples CV index
                extra = list(utils.shuffle(all_data_splits[m_from], random_state=a))[0]  # shuffle again, and try
                a += 1



            a=0
            totals_sub = copy.deepcopy(totals)
            while len(dict_cluster[extra]) >= 1.0*(len(all_samples) / float(cv) ) and a < 5:#in case one cluster contain a lot of samples
                totals_sub[m_from]=0
                m_from = np.argmax(totals_sub)
                extra = list(utils.shuffle(all_data_splits[m_from], random_state=a))[0]  # shuffle again, and try

                a+=1
                # print(a)
            #==========make sure the folder giving out the cluster still have enough samples left.
            tem_Nsamples = []
            for e in  list(set(all_data_splits[m_from]) - set([extra])):  # every time add one cluster order
                elements = dict_cluster[str(e)]
                tem_Nsamples.append(len(elements))  # sample number
            sum_from = sum(tem_Nsamples)

            extracted = extracted + [extra]  # cluster order
            tem_Nsamples = []

            for e in extracted:  # every time add one cluster order
                elements = dict_cluster[str(e)]
                tem_Nsamples.append(len(elements))  # sample number
            sum_sub = sum(tem_Nsamples)
            if len(dict_cluster[extra]) < 1.0*(len(all_samples) / float(cv)) and sum_from > sum_sub:

                print('extracted', extracted)
                print('sum_sub', sum_sub, 'draw from:', m_from,'extra',extra)
                totals[i] = sum_sub
                totals[m_from] = sum_from
                all_data_splits[i]=extracted
                all_data_splits[m_from] = list(set(all_data_splits[m_from]) - set([extra]))  # rm previous cluster order, because it's moved to this fold.
                print('totals=========', totals)

    return all_data_splits
def prepare_cluster(fileDir, p_clusters):
    # prepare a dictionary for clusters, the keys are cluster numbers, items are sample names.
    # cluster index collection.
    # cluster order list
    filename = os.path.join(fileDir, p_clusters)
    filename = os.path.abspath(os.path.realpath(filename))
    output_file_open = open(filename, "r")
    cluster_summary = output_file_open.readlines()
    dict_cluster = collections.defaultdict(list)  # key: starting from 0

    for each_cluster in cluster_summary:
        splitted = each_cluster.split()
        if splitted != []:
            if splitted[0].isdigit():
                dict_cluster[str(int(splitted[0]) - 1)].append(splitted[3])
            if splitted[0] == "Similar":
                splitted = each_cluster.split()
                splitted_2 = each_cluster.split(":")
                dict_cluster[str(int(splitted_2[1].split()[0]) - 1)
                ].append(splitted[6])
    # print("dict_cluster: ", dict_cluster)
    return dict_cluster
def prepare_sample_name(fileDir, p_names):
    # sample name list

    filename = os.path.join(fileDir, p_names)
    filename = os.path.abspath(os.path.realpath(filename))
    names_open = open(filename, "r")
    names_read = names_open.readlines()
    # sample names collection.
    names = []
    for each in range(len(names_read)):
        names.append(names_read[each].replace("\n", ""))  # correct the sample names #no need w.r.t. Patric data
    return names

def prepare_folders(cv, Random_State, p_names, p_clusters,f_version):

    fileDir = os.path.dirname(os.path.realpath('__file__'))
    names=prepare_sample_name(fileDir, p_names)

    dict_cluster = prepare_cluster(fileDir, p_clusters)
    if f_version=='original':
        all_data_splits = cluster_split_old(dict_cluster, Random_State,cv)  # split cluster into cv Folds. len(all_data_splits)=5
    else:
        all_data_splits = cluster_split(dict_cluster, Random_State,cv)  # split cluster into cv Folds. len(all_data_splits)=5


    folders_sample = []  # collection of samples for each split
    for out_cv in range(cv):
        folders_sample_sub = []
        iter_clusters = all_data_splits[out_cv]  # clusters included in that split
        for cl_ID in iter_clusters:
            for element in dict_cluster[cl_ID]:
                folders_sample_sub.append(names.index(element))  # extract cluster ID from the rest folders. 4*(cluster_N)
        folders_sample.append(folders_sample_sub)

    totals = []
    for folder_cluster in all_data_splits:  # 5.#cluster order
        tem = []
        for e in folder_cluster:
            elements = dict_cluster[str(e)]
            tem.append(len(elements))
        totals.append(sum(tem))

    return folders_sample,totals


