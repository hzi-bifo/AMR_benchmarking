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
"For preparing meta files to run Kover 2.0."

def extract_info(path_sequence, s,f_all, f_prepare_meta,cv, level,f_phylotree,f_kma,temp_path):
    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    if f_all == False:
        data = data.loc[s, :]
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()

    if f_prepare_meta:
        # prepare the anti list and id list for each species, antibiotic, and CV folders.
        for species, antibiotics in zip(df_species, antibiotics):
            print(species)
            antibiotics, ID, _ =  load_data.extract_info(species, False, level)
            i_anti = 0
            antibiotics_=[]
            for anti in antibiotics:
                id_all = ID[i_anti]  # sample names
                i_anti += 1
                id_all = np.array(id_all)
                _,name,meta_txt,_ = name_utility.GETname_model2('kover',level, species, anti,'',temp_path,f_kma,f_phylotree)

                file_utility.make_dir(os.path.dirname(meta_txt))
                name_list = pd.read_csv(name, index_col=0, dtype={'genome_id': object}, sep="\t")
                name_list.loc[:,'ID'] = 'iso_' + name_list['genome_id'].astype(str)
                name_list['path']=str(path_sequence) +'/'+ name_list['genome_id'].astype(str)+'.fna'
                name_list1 = name_list.loc[:, ['ID', 'path']]
                name_list1.to_csv(meta_txt + '_data', sep="\t", index=False,header=False)
                name_list2 = name_list.loc[:, ['ID', 'resistant_phenotype']]
                name_list2.to_csv(meta_txt + '_pheno', sep="\t", index=False,header=False)
                name_list['ID'].to_csv(meta_txt + '_id', sep="\t", index=False, header=False)
                for out_cv in range(cv):
                    # 1. exrtact CV folders---------------------------------------------------------------
                    p_names = name_utility.GETname_meta(species,anti,level)
                    folds_txt=name_utility.GETname_folds(species,anti,level,f_kma,f_phylotree)
                    folders_sample = json.load(open(folds_txt, "rb"))
                    folders_index=name2index.Get_index(folders_sample,p_names) # CV folds


                    test_samples_index = folders_index[out_cv]
                    train_val_train_index = folders_index[:out_cv] + folders_index[out_cv + 1:]
                    id_val_train = id_all[list(itertools.chain.from_iterable(train_val_train_index))]  # sample name list
                    id_test = id_all[test_samples_index]

                    # 2. prepare meta files for this round of training samples-------------------
                    # only retain those in the training and validataion CV folders
                    name_list_train = name_list.loc[name_list['genome_id'].isin(id_val_train)]
                    name_list_train.loc[:,'ID'] = 'iso_' + name_list_train['genome_id'].astype(str)
                    name_list_train['ID'].to_csv(meta_txt +  '_Train_' + str(out_cv) + '_id', sep="\t", index=False, header=False)

                    # 3. prepare meta files for this round of testing samples-------------------

                    # only retain those in the training and validataion CV folders
                    name_list_test = name_list.loc[name_list['genome_id'].isin(id_test)]
                    name_list_test.loc[:,'ID'] = 'iso_' + name_list_test['genome_id'].astype(str)
                    name_list_test['ID'].to_csv(meta_txt + '_Test_' + str(out_cv) + '_id', sep="\t", index=False, header=False)

                anti=str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
                antibiotics_.append(anti)
            # save the list to a txt file.
            anti_list,_,_,_= name_utility.GETname_model2('kover',level, species, '','',temp_path,f_kma,f_phylotree)
            pd.DataFrame(antibiotics_).to_csv(anti_list, sep="\t", index=False, header=False)





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-path_sequence', '--path_sequence', type=str, required=False,
                        help='Path of the directory with PATRIC sequences')
    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                        help='Directory to store temporary files.')
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
    parser.add_argument("-cv", "--cv", default=10, type=int,
                        help='CV splits number')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'Pseudomonas aeruginosa\' \
                    \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
                    \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parsedArgs = parser.parse_args()
    extract_info(parsedArgs.path_sequence, parsedArgs.species, parsedArgs.f_all, parsedArgs.f_prepare_meta,
                 parsedArgs.cv, parsedArgs.level, parsedArgs.f_phylotree,parsedArgs.f_kma,parsedArgs.temp_path)
