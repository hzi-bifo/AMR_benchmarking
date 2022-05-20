import os
import argparse
import pickle
import amr_utility.name_utility
import amr_utility.graph_utility
import amr_utility.file_utility
import amr_utility.load_data
import numpy as np
import pandas as pd
import subprocess
import shutil
from pathlib import Path
import cv_folders.cluster_folders
import collections

'''This script extracts the KMA_based CV folders that are not counted in scores w.r.t. each speceis and anti, because of only one phenotype in the folder'''

def extract_info( s, f_all, cv,level,f_phylotree,f_kma):
    fileDir = os.path.dirname(os.path.realpath('__file__'))

    data = pd.read_csv('metadata/' + str(level) + '_Species_antibiotic_FineQuality.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    # for training model on part of the dataset:-------------
    # data=data.loc[['Escherichia coli'],:]
    if f_all == False:
        data = data.loc[s, :]
    # --------------------------------------------------------
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()

    Ignore_dict = collections.defaultdict(list)
    for fscore in ['f1_macro','f1_positive','f1_negative']:
    # for fscore in ['f1_negative']:
        for species, antibiotics in zip(df_species, antibiotics):
            print(species)
            antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)

            # antibiotics, ID, Y=[antibiotics[10]], [ID[10]], [Y[10]]
            i_anti = 0

            for anti in antibiotics:
                # prepare features for each training and testing sets, to prevent information leakage
                # id_all = ID[i_anti]#sample name list, e.g. [1352.10013,1352.10014,1354.10,1366.10]
                # id_all =np.array(id_all)
                y_all = Y[i_anti]
                i_anti+=1
                # id_all =np.array(id_all)
                y_all = np.array(y_all)
                print(anti)

                save_name_meta, p_names = amr_utility.name_utility.GETsave_name_modelID(level, species, anti)
                Random_State = 42
                p_clusters = amr_utility.name_utility.GETname_folder(species, anti, level)

                if f_phylotree:  # phylo-tree based cv folders
                    folders_index = cv_folders.cluster_folders.prepare_folders_tree(cv, species, anti, p_names,
                                                                                      False)
                elif f_kma:  # kma cluster based cv folders
                    folders_index, _, _ = cv_folders.cluster_folders.prepare_folders(cv, Random_State, p_names,
                                                                                           p_clusters,
                                                                                           'new')
                else:#random
                    folders_index = cv_folders.cluster_folders.prepare_folders_random(cv, species, anti, p_names,
                                                                                          False)

                for out_cv in range(cv):
                    test_samples_index = folders_index[out_cv]# a list of index
                    # id_test = id_all[test_samples_index]#sample name list
                    y_test = y_all[test_samples_index]
                    y_test=np.array(y_test)
                    y_check=np.unique(y_test)
                    # print(out_cv)
                    # print(y_test)
                    if fscore== 'f1_macro':
                        # print(out_cv,' , phenotype:', y_check, ', N_sample: ',y_test )
                        if len(y_check)==1:
                            print(out_cv,' , phenotype:', y_check, ', N_sample: ',y_test.shape)
                            Ignore_dict[str(species.replace(" ", "_"))+'/'+str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))].append(out_cv)

                    elif fscore== 'f1_negative':
                        if len(y_check)==1 and y_check[0]==1:
                            print(out_cv,' , phenotype:', y_check, ', N_sample: ',y_test.shape)
                            Ignore_dict[str(species.replace(" ", "_"))+'/'+str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))].append(out_cv)
                    elif fscore== 'f1_positive':
                        if len(y_check)==1 and y_check[0]==0:
                            print(out_cv,' , phenotype:', y_check, ', N_sample: ',y_test.shape)
                            Ignore_dict[str(species.replace(" ", "_"))+'/'+str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))].append(out_cv)





        np.save('cv_folders/'+str(level)+'/igore_list'+'_kma_'+str(f_kma)+'_tree_'+str(f_phylotree)+'_'+fscore, Ignore_dict)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--level', default='loose', type=str,
                        help='Quality control: strict or loose')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument('-f_prepare_meta', '--f_prepare_meta', dest='f_prepare_meta', action='store_true',
                        help='Prepare the list files for S2G.')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                        help='kma based cv folders.')
    parser.add_argument("-cv", "--cv", default=10, type=int,
                        help='CV splits number')
    # parser.add_argument('-fscore', '--fscore', default='f1_macro', type=str, required=False,
    #                     help='the score used to choose the best classifier for each antibiotic. Can be f1_pos'
    #                          'f1_neg, accuracy.')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'Pseudomonas aeruginosa\' \
                    \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
                    \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parsedArgs = parser.parse_args()
    extract_info( parsedArgs.species,parsedArgs.f_all, parsedArgs.cv, parsedArgs.level,parsedArgs.f_phylotree,parsedArgs.f_kma)
