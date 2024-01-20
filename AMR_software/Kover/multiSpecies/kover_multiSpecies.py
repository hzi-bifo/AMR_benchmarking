#!/usr/bin/python
from src.amr_utility import name_utility, file_utility,load_data
import argparse,itertools,os
import numpy as np
import pandas as pd
from src.cv_folds import name2index
import json
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

"For preparing meta files to run Kover over LOSO. " \
"Each time one species-antibiotic combination is used as testing set. " \
"The other combinations sharing the samne antibiotics are used as training set. "

def extract_info(path_sequence, list_species,f_all, f_prepare_meta ,level, temp_path):

    merge_name = []
    data = pd.read_csv('./data/PATRIC/meta/'+str(level)+'_multi-species_summary.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")

    if f_all:
        list_species = data.index.tolist()[:-1]
        data = data.loc[list_species, :]
    else:
        print('Warning. You are not using all the possible data.')
        data = data.loc[list_species, :]
        data = data.loc[:, (data.sum() > 1)]
    data = data.loc[:, (data != 0).any(axis=0)]
    df_anti = data.dot(data.columns + ';').str.rstrip(';')
    for n in list_species:
        merge_name.append(n[0] + n.split(' ')[1][0])


    if f_prepare_meta:
        # prepare the anti list and id list for each species, antibiotic, and CV folders.
        count = 0
        for species_testing in list_species:
            Potential_species_training=list_species[:count] + list_species[count+1 :]
            count += 1
            antibiotics_test=df_anti[species_testing].split(';') #involoved antibiotics
            for anti_test in antibiotics_test:
                print('the species-anti combination to test: ', species_testing, '_',anti_test)
                #########
                ##1. define train & test species,antibiotic. Each time only one combination as test.
                ########
                list_species_training=[]
                for each_species in Potential_species_training:
                    # print(each_species)
                    antibiotics_temp= df_anti[each_species].split(';')
                    # print(antibiotics_temp)
                    if anti_test in antibiotics_temp:
                        list_species_training.append(each_species)
                print('the species for training: ',list_species_training)
                list_species_training_test=list_species_training+[species_testing]
                #########
                ##2. all meta data
                ########

                name_list = pd.DataFrame(columns=['genome_id', 'resistant_phenotype'])
                for each_species in list_species_training_test:
                    _,id_name,_,_ = name_utility.GETname_model3('kover',level, each_species, anti_test,'',temp_path)
                    name_each = pd.read_csv(id_name, index_col=0, dtype={'genome_id': object}, sep="\t")
                    name_list = name_list.append(name_each, ignore_index=True)



                anti_save,_,meta_save,_ = name_utility.GETname_model3('kover',level, species_testing, anti_test,'',temp_path)
                file_utility.make_dir(os.path.dirname(meta_save))
                name_list.loc[:,'ID'] = 'iso_' + name_list['genome_id'].astype(str)
                name_list['path']=str(path_sequence) +'/'+ name_list['genome_id'].astype(str)+'.fna'
                name_list1 = name_list.loc[:, ['ID', 'path']]
                name_list1.to_csv(meta_save + '_data', sep="\t", index=False,header=False)
                name_list2 = name_list.loc[:, ['ID', 'resistant_phenotype']]
                name_list2.to_csv(meta_save + '_pheno', sep="\t", index=False,header=False)
                name_list['ID'].to_csv(meta_save + '_id', sep="\t", index=False, header=False)
                #########
                ##3. training & test set meta data
                ########
                name_list_train = pd.DataFrame(columns=['genome_id', 'resistant_phenotype'])
                for each_species in list_species_training:
                    _,id_name_train,_,_ = name_utility.GETname_model3('kover',level, each_species, anti_test,'',temp_path)
                    name_each = pd.read_csv(id_name_train, index_col=0, dtype={'genome_id': object}, sep="\t")
                    name_list_train = name_list_train.append(name_each, ignore_index=True)
                name_list_train.loc[:,'ID'] = 'iso_' + name_list_train['genome_id'].astype(str)
                name_list_train['ID'].to_csv(meta_save +  '_Train_id', sep="\t", index=False, header=False)



                _,id_name_test,_,_ = name_utility.GETname_model3('kover',level, species_testing, anti_test,'',temp_path)
                name_list_test = pd.read_csv(id_name_test, index_col=0, dtype={'genome_id': object}, sep="\t")
                name_list_test.loc[:,'ID'] = 'iso_' + name_list_test['genome_id'].astype(str)
                name_list_test['ID'].to_csv(meta_save +  '_Test_id', sep="\t", index=False, header=False)

            antibiotics_test_=[str(anti.translate(str.maketrans({'/': '_', ' ': '_'}))) for anti in antibiotics_test]
            pd.DataFrame(antibiotics_test_).to_csv(anti_save, sep="\t", index=False, header=False) #todo re-consider location

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-path_sequence', '--path_sequence', type=str, required=False,
                        help='Path of the directory with PATRIC sequences')
    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                        help='Directory to store temporary files.')
    # parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
    #                     help=' phylo-tree based cv folders.')
    # parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
    #                     help='kma based cv folders.')
    parser.add_argument('-l', '--level', default='loose', type=str,
                        help='Quality control: strict or loose')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument('-f_prepare_meta', '--f_prepare_meta', dest='f_prepare_meta', action='store_true',
                        help='Prepare the list files for S2G.')
    # parser.add_argument("-cv", "--cv", default=10, type=int,
    #                     help='CV splits number')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'Pseudomonas aeruginosa\' \
                    \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
                    \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parsedArgs = parser.parse_args()
    extract_info(parsedArgs.path_sequence, parsedArgs.species, parsedArgs.f_all, parsedArgs.f_prepare_meta,parsedArgs.level,parsedArgs.temp_path)
