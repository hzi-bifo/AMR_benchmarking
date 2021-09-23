import os
import argparse
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
import itertools



def extract_info(path_sequence,s,f_all,f_prepare_meta,f_tree,cv,level,n_jobs,f_ml,f_phylotree):

    # if path_sequence=='/net/projects/BIFO/patric_genome':
    #     path_large_temp='/net/sgi/metagenomics/data/khu/benchmarking/s2g2p'#todo, may need a change
    # else:
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    # path_large_temp = os.path.join(fileDir, 'large_temp')
    # # print(path_large_temp)
    #
    # amr_utility.file_utility.make_dir(path_large_temp)

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
    # print(data)


    for species in df_species:
        amr_utility.file_utility.make_dir('log/temp/' + str(level) + '/'+str(species.replace(" ", "_")))
        amr_utility.file_utility.make_dir('log/results/' + str(level) +'/'+ str(species.replace(" ", "_")))


    if f_prepare_meta:
        #prepare the dna list accroding to S2G format.
        for species, antibiotics in zip(df_species, antibiotics):

            antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
            i_anti = 0

            run_file = open('log/temp/' + str(level) +'/'+ str(species.replace(" ", "_"))+'/'+'getfeature.sh', "w")
            run_file.write("#!/bin/bash")
            run_file.write("\n")
            # if path_sequence == '/vol/projects/BIFO/patric_genome':
            if Path(fileDir).parts[1] == 'vol':
                run_file = amr_utility.file_utility.hzi_cpu_header4(run_file,
                                                                    str(species.replace(" ", "_"))+'seeker',
                                                                    8, 'PhenotypeSeeker','all.q')

            run_file.write("export PYTHONPATH=/vol/projects/khu/amr/PhenotypeSeeker")
            run_file.write("\n")
            for anti in antibiotics:
                # prepare features for each training and testing sets, to prevent information leakage
                id_all = ID[i_anti]
                id_all = np.array(id_all)
                for out_cv in range(cv):

                    #1. exrtact CV folders----------------------------------------------------------------
                    save_name_meta, p_names = amr_utility.name_utility.GETsave_name_modelID(level, species, anti)
                    Random_State = 42
                    p_clusters = amr_utility.name_utility.GETname_folder(species, anti, level)
                    if f_phylotree:  # phylo-tree based cv folders
                        folders_index = cv_folders.cluster_folders.prepare_folders_tree(cv, species, antibiotics,
                                                                                        p_names,
                                                                                        False)
                    else:  # kma cluster based cv folders
                        folders_index, _, _ = cv_folders.cluster_folders.prepare_folders(cv, Random_State, p_names,
                                                                                         p_clusters,
                                                                                         'new')
                    test_samples_index = folders_index[out_cv]
                    train_val_train_index = folders_index[:out_cv] + folders_index[out_cv + 1:]
                    id_val_train = id_all[list(itertools.chain.from_iterable(train_val_train_index))]# sample name list
                    id_test = id_all[test_samples_index]

                    #2. prepare meta files for this round of training samples-------------------

                    name,meta_txt=amr_utility.name_utility.Pts_GETname(level, species, anti)
                    name_list = pd.read_csv(name, index_col=0, dtype={'genome_id': object}, sep="\t")

                    if Path(fileDir).parts[1] == 'vol':
                        # path_list=np.genfromtxt(path, dtype="str")
                        name_list['Addresses'] = '/vol/projects/BIFO/patric_genome/' + name_list['genome_id'].astype(str)+'.fna'
                    else:
                        name_list['Addresses']= '/net/projects/BIFO/patric_genome/' + name_list['genome_id'].astype(str)+'.fna'
                    # only retain those in the training and validataion CV folders
                    name_list_train=name_list.loc[name_list['genome_id'].isin(id_val_train)]

                    name_list_train['ID'] = 'iso_' + name_list_train['genome_id'].astype(str)

                    name_list_train.rename(columns={'resistant_phenotype': anti}, inplace=True)
                    name_list_train = name_list_train.loc[:, ['ID', 'Addresses',anti]]
                    name_list_train.to_csv(meta_txt +'_Train_'+ str(out_cv) + '_data.pheno', sep="\t", index=False,
                                             header=True)

                    # 3. prepare meta files for this round of testing samples-------------------


                    # only retain those in the training and validataion CV folders
                    name_list_test = name_list.loc[name_list['genome_id'].isin(id_test)]

                    name_list_test['ID'] = 'iso_' + name_list_test['genome_id'].astype(str)

                    name_list_test.rename(columns={'resistant_phenotype': anti}, inplace=True)
                    name_list_test = name_list_test.loc[:, ['ID', 'Addresses', anti]]
                    name_list_test.to_csv(meta_txt + '_Test_' + str(out_cv) + '_data.pheno', sep="\t", index=False,
                                     header=True)# note: when running this set, set pval_limit==1.0. --pvalue 1.0


                    #4. prepare bash

                    cmd = 'python /vol/projects/khu/amr/PhenotypeSeeker/scripts/phenotypeseeker_new modeling  ' \
                          '--no_assembly --cv_K %s %s' % (out_cv,str(
        anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'_Train_'+ str(out_cv) + '_data.pheno')
                    run_file.write(cmd)
                    run_file.write("\n")
                    cmd = 'python /vol/projects/khu/amr/PhenotypeSeeker/scripts/phenotypeseeker_new modeling  ' \
                          '--no_assembly --no_weights --cv_K %s %s' % (out_cv, str(
        anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'_Test_' + str(out_cv) + '_data.pheno')
                    run_file.write(cmd)
                    run_file.write("\n")

    if f_tree == True:

            pass


    if f_ml: #ML
        pass





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-path_sequence', '--path_sequence', default= '/net/projects/BIFO/patric_genome', type=str, required=False,
    #                     help='path of sequence,another option: \'/vol/projects/BIFO/patric_genome\'')
    # parser.add_argument('-path_large_temp', '--path_large_temp', default='/net/sgi/metagenomics/data/khu/benchmarking/phylo', type=str,
    #                     required=False, help='path for large temp files/folders, another option: \'/vol/projects/khu/amr/benchmarking/large_temp\'')
    parser.add_argument('-path_sequence', '--path_sequence', default='/vol/projects/BIFO/patric_genome', type=str,
                        required=False,
                        help='path of sequence,another option: \'/net/projects/BIFO/patric_genome\'')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    parser.add_argument('-l', '--level', default='loose', type=str,
                        help='Quality control: strict or loose')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument('-f_prepare_meta', '--f_prepare_meta', dest='f_prepare_meta', action='store_true',
                        help='Prepare the list files for S2G.')
    parser.add_argument('-f_ml', '--f_ml', dest='f_ml', action='store_true',
                        help='prepare bash')
    parser.add_argument("-cv", "--cv", default=10, type=int,
                        help='CV splits number')
    parser.add_argument('-f_tree', '--f_tree', dest='f_tree', action='store_true',
                        help='Kma cluster')  # c program
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'Pseudomonas aeruginosa\' \
                    \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
                    \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parsedArgs = parser.parse_args()
    extract_info(parsedArgs.path_sequence, parsedArgs.species, parsedArgs.f_all, parsedArgs.f_prepare_meta,
                 parsedArgs.f_tree, parsedArgs.cv, parsedArgs.level, parsedArgs.n_jobs, parsedArgs.f_ml, parsedArgs.f_phylotree)
