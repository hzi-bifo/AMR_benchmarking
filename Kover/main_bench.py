import amr_utility.name_utility
import amr_utility.graph_utility
import amr_utility.file_utility
import amr_utility.load_data
import cv_folders.cluster_folders
import argparse,itertools,os
from pathlib import Path
import numpy as np
import pandas as pd

"For preparing meta files to run Kover 2.0."

def extract_info(path_sequence, s,kmer,f_all, f_prepare_meta,cv, level, n_jobs, f_ml,f_phylotree,f_kma,f_qsub):
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
    # print(data)


    for species in df_species:
        amr_utility.file_utility.make_dir('log/temp/' + str(level) + '/'+str(species.replace(" ", "_")))
        amr_utility.file_utility.make_dir('log/results/' + str(level) +'/'+ str(species.replace(" ", "_")))

    if f_prepare_meta:
        # prepare the anti list and id list for each species, antibiotic, and CV folders.
        for species, antibiotics in zip(df_species, antibiotics):

            antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
            i_anti = 0


            antibiotics_=[]
            for anti in antibiotics:
                id_all = ID[i_anti]  # sample names
                i_anti += 1
                id_all = np.array(id_all)
                print(anti)
                print(id_all.shape)
                print('-------------')

                name, meta_txt, _ = amr_utility.name_utility.Pts_GETname(level, species, anti,'')
                name_list = pd.read_csv(name, index_col=0, dtype={'genome_id': object}, sep="\t")
                name_list.loc[:,'ID'] = 'iso_' + name_list['genome_id'].astype(str)
                if Path(fileDir).parts[1] == 'vol':
                    # path_list=np.genfromtxt(path, dtype="str")
                    name_list['path'] = '/vol/projects/BIFO/patric_genome/' + name_list['genome_id'].astype(str)+'.fna'
                else:
                    name_list['path']= '/net/projects/BIFO/patric_genome/' + name_list['genome_id'].astype(str)+'.fna'
                name_list1 = name_list.loc[:, ['ID', 'path']]
                name_list1.to_csv(meta_txt + '_data', sep="\t", index=False,header=False)
                name_list2 = name_list.loc[:, ['ID', 'resistant_phenotype']]
                name_list2.to_csv(meta_txt + '_pheno', sep="\t", index=False,header=False)
                name_list['ID'].to_csv(meta_txt + '_id', sep="\t", index=False, header=False)
                for out_cv in range(cv):

                    # 1. exrtact CV folders----------------------------------------------------------------
                    _, p_names = amr_utility.name_utility.GETsave_name_modelID(level, species, anti)
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
                    test_samples_index = folders_index[out_cv]
                    train_val_train_index = folders_index[:out_cv] + folders_index[out_cv + 1:]
                    id_val_train = id_all[list(itertools.chain.from_iterable(train_val_train_index))]  # sample name list
                    id_test = id_all[test_samples_index]

                    # 2. prepare meta files for this round of training samples-------------------



                    # only retain those in the training and validataion CV folders
                    name_list_train = name_list.loc[name_list['genome_id'].isin(id_val_train)]
                    # name_list_train['genome_id'].to_csv(meta_txt + '_Train_' + str(out_cv) + '_id2', sep="\t", index=False, header=False)
                    name_list_train.loc[:,'ID'] = 'iso_' + name_list_train['genome_id'].astype(str)
                    name_list_train['ID'].to_csv(meta_txt +  '_Train_' + str(out_cv) + '_id', sep="\t", index=False, header=False)
                    #
                    # name_list_train1 = name_list_train.loc[:, ['ID', 'path']]
                    # name_list_train1.to_csv(meta_txt + '_Train_' + str(out_cv) + '_data', sep="\t", index=False,header=False)
                    # name_list_train2 = name_list_train.loc[:, ['ID', 'resistant_phenotype']]
                    # name_list_train2.to_csv(meta_txt + '_Train_' + str(out_cv) + '_pheno', sep="\t", index=False,header=False)


                    # 3. prepare meta files for this round of testing samples-------------------

                    # only retain those in the training and validataion CV folders
                    name_list_test = name_list.loc[name_list['genome_id'].isin(id_test)]
                    # name_list_test['genome_id'].to_csv(meta_txt + '_Test_' + str(out_cv) + '_id2', sep="\t", index=False,header=False)
                    name_list_test.loc[:,'ID'] = 'iso_' + name_list_test['genome_id'].astype(str)
                    name_list_test['ID'].to_csv(meta_txt + '_Test_' + str(out_cv) + '_id', sep="\t", index=False, header=False)

                    # name_list_test1 =name_list_test.loc[:, ['ID', 'path']]
                    # name_list_test1.to_csv(meta_txt + '_Test_' + str(out_cv) + '_data', sep="\t", index=False,header=False)
                    # name_list_test2 = name_list_test.loc[:, ['ID', 'resistant_phenotype']]
                    # name_list_test2.to_csv(meta_txt + '_Test_' + str(out_cv) + '_pheno', sep="\t", index=False,header=False)


                anti=str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
                antibiotics_.append(anti)
            # save the list to a txt file.
            anti_list = 'log/temp/' + str(level) + '/' + str(species.replace(" ", "_")) + '/anti_list'
            pd.DataFrame(antibiotics_).to_csv(anti_list, sep="\t", index=False, header=False)





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('-path_sequence', '--path_sequence', default= '/net/projects/BIFO/patric_genome', type=str, required=False,
    #                     help='path of sequence,another option: \'/vol/projects/BIFO/patric_genome\'')
    # parser.add_argument('-path_large_temp', '--path_large_temp', default='/net/sgi/metagenomics/data/khu/benchmarking/phylo', type=str,
    #                     required=False, help='path for large temp files/folders, another option: \'/vol/projects/khu/amr/benchmarking/large_temp\'')
    parser.add_argument('-path_sequence', '--path_sequence', default='/vol/projects/BIFO/patric_genome', type=str,
                        required=False,
                        help='path of sequence,another option: \'/net/projects/BIFO/patric_genome\'')
    parser.add_argument('-k', '--kmer', default=31, type=int,
                        help='k-mer')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                        help='kma based cv folders.')
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
    # parser.add_argument('-f_tree', '--f_tree', dest='f_tree', action='store_true',
    #                     help='Kma cluster')
    parser.add_argument('-f_qsub', '--f_qsub', dest='f_qsub',
                        help='Prepare scriptd for qsub.', action='store_true', )
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'Pseudomonas aeruginosa\' \
                    \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
                    \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parsedArgs = parser.parse_args()
    extract_info(parsedArgs.path_sequence, parsedArgs.species, parsedArgs.kmer,parsedArgs.f_all, parsedArgs.f_prepare_meta,
                 parsedArgs.cv, parsedArgs.level, parsedArgs.n_jobs, parsedArgs.f_ml,
                 parsedArgs.f_phylotree,parsedArgs.f_kma,parsedArgs.f_qsub)
