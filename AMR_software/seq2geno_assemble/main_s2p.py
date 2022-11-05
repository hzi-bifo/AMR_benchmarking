#!/usr/bin/python
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import numpy as np
from src.amr_utility import name_utility, file_utility, load_data
import argparse
import itertools
import pandas as pd
# import hyper_range
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import auc,roc_curve,matthews_corrcoef,confusion_matrix,f1_score,precision_recall_fscore_support,classification_report
# import pickle,json
# from sklearn import preprocessing
# from src.cv_folds import name2index


def extract_info(path_sequence,temp_path,s,f_all,f_prepare_meta, f_phylotree, f_kma,level,f_ml,cv,n_jobs):
    software_name='seq2geno'

    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]
    if f_all == False:
        data = data.loc[s, :]
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    file_utility.make_dir(temp_path+'log/software/'+software_name+'/software_output/cano6mer/temp') #for kmers




    if f_prepare_meta:
        fileDir = os.path.dirname(os.path.realpath('__file__'))
        #prepare the dna list accroding to S2G format.
        for species, antibiotics in zip(df_species, antibiotics):
            antibiotics, _, _ =  load_data.extract_info(species, False, level)
            ALL=[]
            for anti in antibiotics:
                name,_,_ = name_utility.GETname_model(software_name,level, species, anti,'',temp_path)
                name_list = pd.read_csv(name, index_col=0, dtype={'genome_id': object}, sep="\t")
                name_list['path']=str(path_sequence) +'/'+ name_list['genome_id'].astype(str)+'.fna'
                name_list['genome_id'] = 'iso_' + name_list['genome_id'].astype(str)
                pseudo=np.empty(len(name_list.index.to_list()),dtype='object')
                pseudo.fill (fileDir+'/AMR_software/'+software_name+'/example/CH2500.1.fastq.gz,'+fileDir+'/AMR_software/'+software_name+'/example/CH2500.2.fastq.gz')
                name_list['path_pseudo'] = pseudo
                ALL.append(name_list)

            _,dna_list, assemble_list, yml_file, run_file_name,wd ,pseudo_args=  name_utility.GETname_S2Gfeature( species, temp_path,6)
            file_utility.make_dir(os.path.dirname(dna_list))

            #combine the list for all antis
            species_dna=ALL[0]
            for i in ALL[1:]:
                species_dna = pd.merge(species_dna, i, how="outer", on=["genome_id",'path','path_pseudo'])# merge antibiotics within one species
            # print(species_dna)
            species_dna_final=species_dna.loc[:,['genome_id','path']]
            species_dna_final.to_csv(assemble_list, sep="\t", index=False,header=False)
            species_pseudo = species_dna.loc[:, ['genome_id', 'path_pseudo']]
            species_pseudo.to_csv(dna_list, sep="\t", index=False, header=False)


            #prepare yml files based on a basic version at working directory


            wd_results = wd + 'results'
            file_utility.make_dir(wd_results)

            # # fileDir = os.path.dirname(os.path.realpath('__file__'))
            # # wd_results = os.path.join(fileDir, wd_results)
            # # assemble_list=os.path.join(fileDir, assemble_list)
            # # dna_list=os.path.join(fileDir, dna_list)

            # #modify the yml file
            a_file = open('./AMR_software/'+software_name+'/seq2geno_inputs.yml', "r")
            list_of_lines = a_file.readlines()
            list_of_lines[12] = "    100000\n"
            list_of_lines[14] = "    %s\n" % dna_list
            list_of_lines[26] = "    %s\n" % wd_results
            list_of_lines[28] = "    %s\n" % assemble_list


            list_of_lines[16] = "    %s\n" % pseudo_args[0]
            list_of_lines[18] = "    %s\n" % pseudo_args[1]
            list_of_lines[20] = "    %s\n" % pseudo_args[2]
            list_of_lines[22] = "    %s\n" % pseudo_args[3]
            list_of_lines[24] = "    %s\n" % pseudo_args[4]




            a_file = open(yml_file, "w")
            a_file.writelines(list_of_lines)
            a_file.close()




def create_generator(nFolds):
    for idx in range(len(nFolds)):
        test =nFolds[idx]
        train = list(itertools.chain(*[fold for idy, fold in enumerate(nFolds) if idy != idx]))
        yield train, test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-path_sequence', '--path_sequence', type=str, required=False,
                        help='Path of the directory with PATRIC sequences')
    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                        help='Directory to store temporary files.')
    parser.add_argument('-l', '--level', default='loose', type=str,
                        help='Quality control: strict or loose')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species.')
    parser.add_argument('-f_prepare_meta', '--f_prepare_meta', dest='f_prepare_meta', action='store_true',
                        help='Prepare the list files for S2G.')
    parser.add_argument('-f_finished', '--f_finished', dest='f_finished', action='store_true',
                        help='delete large unnecessary tempt files')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folds.')
    parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                        help='kma based cv folds.')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'Pseudomonas aeruginosa\' \
                \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
                \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('-f_ml', '--f_ml', dest='f_ml', action='store_true',
                        help='ML')
    parser.add_argument("-cv", "--cv_number", default=10, type=int,
                        help='CV splits number. Default=10 ')
    parser.add_argument('-n_jobs','--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parsedArgs = parser.parse_args()
    extract_info(parsedArgs.path_sequence,parsedArgs.temp_path,parsedArgs.species,parsedArgs.f_all,parsedArgs.f_prepare_meta,
                parsedArgs.f_phylotree,parsedArgs.f_kma,parsedArgs.level,parsedArgs.f_ml,parsedArgs.cv_number,parsedArgs.n_jobs)
