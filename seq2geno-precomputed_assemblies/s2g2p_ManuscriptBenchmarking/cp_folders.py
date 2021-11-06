#!/usr/bin/env/python
# @Author: kxh
# @Date:   Nov 5 2021
# @Last Modified by:   
# @Last Modified time:
'''
This script cp or folders from Ehsan 's directory
'''
import argparse,os,ast
import pandas as pd
import numpy as np
from shutil import copyfile
import amr_utility.file_utility,amr_utility.load_data
# '/net/sgi/metagenomics/nobackup/prot/new_experiments/ecoli/patric_data/results/res_amoxicillin/classification/cv/ECOLI_amoxicillin/treecv/gpaindel_amoxicillin_resistance'

def model(l,df_species,cl, antibiotics,cv,n_jobs,f_phylotree,f_kma):
    pass

def extract_info(l,s,cv,n_jobs):
    data = pd.read_csv('metadata/' + str(l) + '_Species_antibiotic_FineQuality.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    # data = pd.read_csv('metadata/Species_antibiotic_FineQuality.csv', index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]# drop the species with 0 in column 'number'.
    # for training model on part of the dataset:-------------
    # data=data.loc[['Escherichia coli'],:]
    data = data.loc[s, :]
    #data.at['Mycobacterium tuberculosis', 'modelling antibiotics']=['capreomycin', 'ciprofloxacin']
    # --------------------------------------------------------
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    print(data)
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    for species,antibiotics in zip(df_species, antibiotics):

        temp=fileDir+'/temp_folders/'+ str(species.replace(" ", "_"))
        amr_utility.file_utility.make_dir(temp)#@ BIFO
        antibiotics_selected = ast.literal_eval(antibiotics)
        print(species)
        print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)
        antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, l)

        for anti in antibiotics:
            #todo need to change after ecoli

            original='/net/sgi/metagenomics/nobackup/prot/new_experiments/ecoli/patric_data/results/res_'+ str(
            anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'/classification/cv/ECOLI_'+ str(
            anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'/treecv/gpaindel_'+ str(
            anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'_resistance/cv_folds.txt'
            des=temp+'/cv_folds_'+ str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'.txt'
            print(original)
            copyfile(original, des)



if __name__== '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-clf', '--Model_classifier', default='svm', type=str, required=True,
    #                     help='svm,logistic,lsvm,all')
    parser.add_argument('--l', '--level', default='loose', type=str, required=False,
                        help='Quality control: strict or loose')
    parser.add_argument("-cv", "--cv_number", default=10, type=int,
                        help='CV splits number')
    parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parser.add_argument('-s','--species', default=[], type=str,nargs='+',help='species to run: e.g.\'seudomonas aeruginosa\' \
     \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
     \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')

    #parser.set_defaults(canonical=True)
    parsedArgs=parser.parse_args()
    # parser.print_help()
    print(parsedArgs)
    extract_info(parsedArgs.l,parsedArgs.species,parsedArgs.cv_number,parsedArgs.n_jobs)
