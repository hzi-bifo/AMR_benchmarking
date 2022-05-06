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
import pickle
import numpy as np
import getopt
import sys
import warnings
import matplotlib
matplotlib.use('Agg')
import collections
import random
from sklearn import utils
import copy

'''
Note;
This script prepares CV folders in accordance with the id list in meta/model/loose/species_anti for single-species model;
./log/temp/loose/multi-species/merge_name_id(CV training) for multi-concat model.
'./log/temp/' + str(level) + '/multi_species/' + merge_name + '/' + 'ID' for discrete multi-s model.
'''



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
        #
        # print('sum_tem', sum_tem)  # all_samples in that folder: val,train,test.
        # print('average',len(all_samples) / float(cv))
        # print('len(all_samples)',len(all_samples))
        a = 0
        # while sum_tem + 200 < len(all_samples) / float(cv):  # all_samples: val,train,test
        while sum_tem + 1000 < len(all_samples) / float(cv):  # all_samples: val,train,test
            extra = list(utils.shuffle(
                all_data_splits_pre[m_fromprevious], random_state=Random_State))[a]  # cluster order
            extracted = extracted + [extra]  # cluster order
            all_extra.append(extra)
            a = a + 1
            tem_Nsamples = []
            # print('extracted', extracted)
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
    # print('all samples',sum(totals),totals,len(all_samples) / float(cv))


    # all_data_splits_pre=copy.deepcopy(all_data_splits)
    # print('Re_sampling...........')

    for i in range(len(all_data_splits)):  # 5.#cluster order

        # extracted = list(set(all_data_splits[i]))  # order of cluster, w.r.t dict_cluster
        extracted = all_data_splits[i]
        sum_sub = totals[i]
        # print(sum_sub)
        # print('totals====================================',totals)
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
                # print('!!!!',m_from,extra)
                a+=1
                # print(a)


            #==========make sure the folder giving out the cluster still have enough samples left.
            tem_Nsamples = []
            for e in sorted(list(set(all_data_splits[m_from]) - set([extra]))):  # every time add one cluster order
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

                # print('extracted', extracted)
                # print('sum_sub', sum_sub, 'draw from:', m_from,'extra',extra)
                totals[i] = sum_sub
                totals[m_from] = sum_from
                all_data_splits[i]=extracted
                all_data_splits[m_from] = sorted(list(set(all_data_splits[m_from]) - set([extra]))) # rm previous cluster order, because it's moved to this fold.
                # print('totals=========', totals)

    return all_data_splits

# def prepare_cluster(fileDir, p_clusters):
#     # prepare a dictionary for clusters, the keys are cluster numbers, items are sample names.
#     # cluster index collection.
#     # cluster order list
#
#     filename = os.path.join(fileDir, p_clusters)
#     filename = os.path.abspath(os.path.realpath(filename))
#     output_file_open = open(filename, "r")
#     cluster_summary = output_file_open.readlines()
#
#     # starting..........................................................
#     dict_cluster = collections.defaultdict(list)  # key: starting from 0
#
#     for each_cluster in cluster_summary:
#         splitted = each_cluster.split()
#         if splitted != []:
#             if splitted[0].isdigit():
#                 dict_cluster[str(int(splitted[0]) - 1)].append(splitted[3])
#             if splitted[0] == "Similar":
#                 splitted = each_cluster.split()
#                 splitted_2 = each_cluster.split(":")
#                 dict_cluster[str(int(splitted_2[1].split()[0]) - 1)
#                 ].append(splitted[6])
#     print("dict_cluster size: ", len(dict_cluster))
#     return dict_cluster

def prepare_cluster(fileDir, p_clusters):
    # prepare a dictionary for clusters, the keys are cluster numbers, items are sample names.
    # cluster index collection.
    # cluster order list
    if type(p_clusters) == list:#multi_species, & multi-output.
        dict_cluster=[]#D: number of speceis
        for each_cluster in p_clusters:
            filename = os.path.join(fileDir, each_cluster)
            filename = os.path.abspath(os.path.realpath(filename))
            output_file_open = open(filename, "r")
            cluster_summary = output_file_open.readlines()
            # starting..........................................................
            dict_cluster_sub = collections.defaultdict(list)  # key: starting from 0

            for each_cluster in cluster_summary:
                splitted = each_cluster.split()
                if splitted != []:
                    if splitted[0].isdigit():
                        dict_cluster_sub[str(int(splitted[0]) - 1)].append(splitted[3])
                    if splitted[0] == "Similar":
                        splitted = each_cluster.split()
                        splitted_2 = each_cluster.split(":")
                        dict_cluster_sub[str(int(splitted_2[1].split()[0]) - 1)
                        ].append(splitted[6])
                    # if splitted[1]=='Indexing':#meaning starting next species ina multi-species setting.
            dict_cluster.append(dict_cluster_sub)
            # count=dict_cluster_sub.keys()[-1]+1
            # # count+=1



    else:
        filename = os.path.join(fileDir, p_clusters)
        filename = os.path.abspath(os.path.realpath(filename))
        output_file_open = open(filename, "r")
        cluster_summary = output_file_open.readlines()


        # starting..........................................................
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
    # print("dict_cluster size: ", len(dict_cluster))
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

# def prepare_folders(cv, Random_State, p_names, p_clusters,f_version):
#     '''
#     :param p_names: list of sample names, in the same order of day_x, day_y.
#     :param p_clusters: kma cluster output, with a suffix of '_clustered_90.txt'.
#     :param f_version: "new": khu; original: Derya Aytan's work.
#     :return: folders_sample: Dimension: cv*(sample_n in each split(it varies)). Elements: index of sampels w.r.t. data_x, data_y
#             totals: sample number in each folder.
#     '''
#     fileDir = os.path.dirname(os.path.realpath('__file__'))
#     names=prepare_sample_name(fileDir, p_names)
#
#     dict_cluster = prepare_cluster(fileDir, p_clusters)
#
#     if type(dict_cluster)==collections.defaultdict:
#         if f_version=='original':
#             all_data_splits = cluster_split_old(dict_cluster, Random_State,cv)  # split cluster into cv Folds. len(all_data_splits)=5
#         else:
#             all_data_splits = cluster_split(dict_cluster, Random_State,cv)  # split cluster into cv Folds. len(all_data_splits)=5
#
#
#         folders_sample = []  # collection of samples for each split
#         for out_cv in range(cv):
#             folders_sample_sub = []
#             iter_clusters = all_data_splits[out_cv]  # clusters included in that split
#             for cl_ID in iter_clusters:
#                 for element in dict_cluster[cl_ID]:
#                     folders_sample_sub.append(names.index(element))  # extract cluster ID from the rest folders. 4*(cluster_N)
#             folders_sample.append(folders_sample_sub)
#
#         totals = []
#         for folder_cluster in all_data_splits:  # 5.#cluster order
#             tem = []
#             for e in folder_cluster:
#                 elements = dict_cluster[str(e)]
#                 tem.append(len(elements))
#             totals.append(sum(tem))
#
#     elif type(dict_cluster)==list:#multi-species
#         folders_sample=[]#D:n_cv* sample numbers
#         totals=[]
#         for out_cv in range(cv):
#             folders_sample.append([])
#
#
#         for  dict_cluster_sub in dict_cluster:
#             if f_version == 'original':
#                 all_data_splits = cluster_split_old(dict_cluster_sub, Random_State,
#                                                     cv)  # split cluster into cv Folds. len(all_data_splits)=5
#             else:
#                 all_data_splits = cluster_split(dict_cluster_sub, Random_State,
#                                                 cv)  # split cluster into cv Folds. len(all_data_splits)=5
#             # folders_sample_single_species=[]
#             for out_cv in range(cv):
#                 # folders_sample_sub = []
#                 iter_clusters = all_data_splits[out_cv]  # clusters included in that split
#                 for cl_ID in iter_clusters:
#                     for element in dict_cluster_sub[cl_ID]:
#                         folders_sample[out_cv].append(
#                             names.index(element))  # extract cluster ID from the rest folders. 4*(cluster_N)
#             totals_sub = []
#             for folder_cluster in all_data_splits:  # 5.#cluster order
#                 tem = []
#                 for e in folder_cluster:
#                     elements = dict_cluster_sub[str(e)]
#                     tem.append(len(elements))
#                 totals_sub.append(sum(tem))
#             totals.append(totals_sub)# in the order of species , the oder comes from list: p_cluster.
#
#     return folders_sample,totals


def prepare_folders(cv, Random_State, p_names, p_clusters,f_version):
    '''
    Based on kma clusers.
    :param p_names: list of sample names, in the same order of day_x, day_y.
    :param p_clusters: kma cluster output, with a suffix of '_clustered_90.txt'.
    :param f_version: "new": khu; original: Derya Aytan's work.
    :return: folders_sample: Dimension: cv*(sample_n in each split(it varies)). Elements: index of sampels w.r.t. data_x, data_y
            totals: sample number in each folder. folders_sampleName: sample names in each folder.
    '''
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    names=prepare_sample_name(fileDir, p_names)

    dict_cluster = prepare_cluster(fileDir, p_clusters)

    if type(dict_cluster)==collections.defaultdict:
        if f_version=='original':
            all_data_splits = cluster_split_old(dict_cluster, Random_State,cv)  # split cluster into cv Folds. len(all_data_splits)=5
        else:
            all_data_splits = cluster_split(dict_cluster, Random_State,cv)  # split cluster into cv Folds. len(all_data_splits)=5


        folders_sample = []  # collection of samples for each split
        folders_sampleName=[] # only for G2P use
        for out_cv in range(cv):
            folders_sample_sub = []
            folders_sampleName_sub=[]
            iter_clusters = all_data_splits[out_cv]  # clusters included in that split
            for cl_ID in iter_clusters:
                for element in dict_cluster[cl_ID]:
                    folders_sample_sub.append(names.index(element))  # extract cluster ID from the rest folders. 4*(cluster_N)
                    folders_sampleName_sub.append('iso_'+element)
            folders_sample.append(folders_sample_sub)
            folders_sampleName.append(folders_sampleName_sub)

        totals = []
        for folder_cluster in all_data_splits:  # 5.#cluster order
            tem = []
            for e in folder_cluster:
                elements = dict_cluster[str(e)]
                tem.append(len(elements))
            totals.append(sum(tem))

    elif type(dict_cluster)==list:#multi-species
        folders_sample=[]#D:n_cv* sample numbers
        folders_sampleName = []
        totals=[]
        for out_cv in range(cv):
            folders_sample.append([])
            folders_sampleName.append([])# only for G2P use

        for dict_cluster_sub in dict_cluster:
            if f_version == 'original':
                all_data_splits = cluster_split_old(dict_cluster_sub, Random_State,
                                                    cv)  # split cluster into cv Folds. len(all_data_splits)=5
            else:
                all_data_splits = cluster_split(dict_cluster_sub, Random_State,
                                                cv)  # split cluster into cv Folds. len(all_data_splits)=5
            # folders_sample_single_species=[]
            for out_cv in range(cv):
                # folders_sample_sub = []
                iter_clusters = all_data_splits[out_cv]  # clusters included in that split
                for cl_ID in iter_clusters:
                    for element in dict_cluster_sub[cl_ID]:
                        folders_sample[out_cv].append(
                            names.index(element))  # extract cluster ID from the rest folders. 4*(cluster_N)
                        folders_sampleName[out_cv].append(
                            'ISO_'+element)
            totals_sub = []
            for folder_cluster in all_data_splits:  # 5.#cluster order
                tem = []
                for e in folder_cluster:
                    elements = dict_cluster_sub[str(e)]
                    tem.append(len(elements))
                totals_sub.append(sum(tem))
            totals.append(totals_sub)# in the order of species , the order comes from list: p_cluster.
    print(totals)
    return folders_sample,totals,folders_sampleName

def prepare_folders_tree(cv,species,anti,p_names,f_multi):
    '''
    Based on phylo-trees.
    :return:  index of sampels w.r.t. data_x, data_y, according to p_names
    '''
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    # main_path='/net/sgi/metagenomics/nobackup/prot/ecoli_res/'+ str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+\
    #           '/res-all/classification/cv/ecoli_'+str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+\
    #           '_tree/tree/6mers-std-tree_resistant_phenotype/cv_folds.txt'
    main_path='cv_folders/'+'loose'+'/'+str(species.replace(" ", "_"))+'/cv_tree_'+ str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'.txt'
    mapping_file='cv_folders/'+'loose'+'/'+str(species.replace(" ", "_"))+'/mapping_2.npy'
    tree_names=[]
    with open(main_path) as f:
        lines = f.readlines()
        for i in lines:
            # print(i.split('\t'))
            tree_names_sub=[]


            for each in i.split('\t'):
                each=each.replace("\n", "")
                # decode the md5 name to iso names
                mapping_dic = np.load(mapping_file, allow_pickle='TRUE').item()
                decoder_name = mapping_dic[each]
                iso_name=decoder_name[0]
                #------------------------------------
                tree_names_sub.append(iso_name.replace("iso_", "").replace("\n", ""))
                # print(tree_names_sub)
            tree_names.append(tree_names_sub)
    # print(tree_names)
    names = prepare_sample_name(fileDir, p_names)
    # print(names)
    if f_multi:
        pass
        #todo
    else:#single-species model
        folders_sample = []  # collection of samples for each split

        for out_cv in range(cv):
            folders_sample_sub = []

            tree_names_split = tree_names[out_cv]  # list: names included in that split
            for cl_ID in tree_names_split:

                folders_sample_sub.append(
                    names.index(cl_ID))  # extract cluster ID from the rest folders. 4*(cluster_N)

            folders_sample.append(folders_sample_sub)
    # print(folders_sample)
    return folders_sample


def prepare_folders_random(cv,species,anti,p_names,f_multi):
    '''
    Based on phylo-trees.
    :return:  index of sampels w.r.t. data_x, data_y, according to p_names
    '''
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    if species != "Mycobacterium tuberculosis":

        # main_path='/net/sgi/metagenomics/nobackup/prot/ecoli_res/'+ str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+\
        #           '/res-all/classification/cv/ecoli_'+str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+\
        #           '_tree/tree/6mers-std-tree_resistant_phenotype/cv_folds.txt'
        main_path='cv_folders/'+'loose'+'/'+str(species.replace(" ", "_"))+'/cv_random_'+ str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'.txt'
        mapping_file='cv_folders/'+'loose'+'/'+str(species.replace(" ", "_"))+'/mapping_2.npy'
        Random_names=[]
        with open(main_path) as f:
            lines = f.readlines()
            for i in lines:
                # print(i.split('\t'))
                Random_names_sub=[]
                for each in i.split('\t'):
                    each=each.replace("\n", "")
                    # decode the md5 name to iso names
                    mapping_dic = np.load(mapping_file, allow_pickle='TRUE').item()
                    decoder_name = mapping_dic[each]
                    iso_name=decoder_name[0]
                    #------------------------------------
                    Random_names_sub.append(iso_name.replace("iso_", "").replace("\n", ""))
                    # print(tree_names_sub)
                Random_names.append(Random_names_sub)
        # print(tree_names)
        names = prepare_sample_name(fileDir, p_names)

    else:#only for the case of "Mycobacterium tuberculosis"
        main_path='cv_folders/'+'loose'+'/'+str(species.replace(" ", "_"))+"/"+ \
              str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_random_cv.pickle"

        Random_names = pickle.load(open(main_path, "rb"))
        names = prepare_sample_name(fileDir, p_names)
    if f_multi:
        pass
        #todo
    else:#single-species model
        folders_sample = []  # collection of samples for each split

        for out_cv in range(cv):
            folders_sample_sub = []
            tree_names_split = Random_names[out_cv]  # list: names included in that split
            for cl_ID in tree_names_split:

                folders_sample_sub.append(
                    names.index(cl_ID))  # extract cluster ID from the rest folders. 4*(cluster_N)

            folders_sample.append(folders_sample_sub)
    # print(folders_sample)
    return folders_sample


def prepare_folders_multiAnti(cv,species,antibiotics,p_names,f_random,f_tree,level):
    #todo
    # Based on pickle files
    # :return:  index of sampels w.r.t. data_x, data_y, according to p_names

    fileDir = os.path.dirname(os.path.realpath('__file__'))
    #
    # main_path='cv_folders/'+'loose'+'/'+str(species.replace(" ", "_"))+"/"+ \
    #           str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_random_cv.pickle"
    multi_log='./log/temp/' + str(level) + '/multi_anti/' + str(species.replace(" ", "_")) + '/'
    
    if f_random:
        # main_path=multi_log+str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_random_raw.pickle"
        main_path2=multi_log+ "kma.pickle"
    elif f_tree:
        # main_path=multi_log+str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_tree_raw.pickle"
        main_path2=multi_log+ "kma.pickle"
    else:#f_kma
        # main_path=multi_log+str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_kma_raw.pickle"
        main_path2=multi_log+ "kma.pickle"
        print('kma folds!')


    Random_names = pickle.load(open(main_path2, "rb"))
    names = prepare_sample_name(fileDir, p_names)
    # print(names)
    folders_sample = []  # collection of samples for each split
    for out_cv in range(cv):
        folders_sample_sub = []
        tree_names_split = Random_names[out_cv]  # list: names included in that split
        for cl_ID in tree_names_split:
            folders_sample_sub.append(
                names.index(cl_ID))  # extract cluster ID from the rest folders. 4*(cluster_N)
        folders_sample.append(folders_sample_sub)
    # # print(folders_sample)
    # # --------------------
    # # --------------------
    # folders_sample_test = []  # collection of samples for each split
    # for out_cv in range(cv):
    #     for anti in antibiotics:
    #         if f_random:
    #             main_path=multi_log+str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_random_raw.pickle"
    #             # main_path2=multi_log+ "kma.pickle"
    #         elif f_tree:
    #             main_path=multi_log+str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_tree_raw.pickle"
    #             # main_path2=multi_log+ "kma.pickle"
    #         else:#f_kma
    #             main_path=multi_log+str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_kma_raw.pickle"
    #             # main_path2=multi_log+ "kma.pickle"
    #         Random_names = pickle.load(open(main_path, "rb"))
    #         names = prepare_sample_name(fileDir, p_names)
    #         # print(names)
    #
    #         folders_sample_sub = []
    #         tree_names_split = Random_names[out_cv]  # list: names included in that split
    #         for cl_ID in tree_names_split:
    #             folders_sample_sub.append(
    #                 names.index(cl_ID))  # extract cluster ID from the rest folders. 4*(cluster_N)
    #         folders_sample_test.append(folders_sample_sub)

    return folders_sample
'''
def prepare_folders_random_mt(cv,species,anti,p_names,f_multi):
    
    # Based on phylo-trees.
    # :return:  index of sampels w.r.t. data_x, data_y, according to p_names
    
    fileDir = os.path.dirname(os.path.realpath('__file__'))

    main_path='cv_folders/'+'loose'+'/'+str(species.replace(" ", "_"))+"/"+ \
              str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_random_cv.pickle"
    Random_names=[]
    Random_names = pickle.load(open(main_path, "rb"))
    # with open(main_path) as f:
    #     lines = f.readlines()
    #     for i in lines:
    #         # print(i.split('\t'))
    #         Random_names_sub=[]
    #         for each in i.split('\t'):
    #             each=each.replace("\n", "")
    #             # print(each)
    #             # print('-')
    #             Random_names_sub.append(each)
    #             # print(tree_names_sub)
    #         Random_names.append(Random_names_sub)
    names = prepare_sample_name(fileDir, p_names)
    # print(names)
    if f_multi:
        pass
        #todo
    else:#single-species model
        folders_sample = []  # collection of samples for each split

        for out_cv in range(cv):
            folders_sample_sub = []

            tree_names_split = Random_names[out_cv]  # list: names included in that split
            for cl_ID in tree_names_split:

                folders_sample_sub.append(
                    names.index(cl_ID))  # extract cluster ID from the rest folders. 4*(cluster_N)

            folders_sample.append(folders_sample_sub)
    # print(folders_sample)
    return folders_sample
'''
