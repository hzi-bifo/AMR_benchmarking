

import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
from subprocess import PIPE, run
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score, KFold,cross_val_predict,cross_validate
from sklearn import svm,preprocessing
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import multiprocessing as mp
import time
import pickle
import argparse






def model(species,antibiotics,level):
    path = "/net/projects/BIFO/patric_genome/"
    print(species)
    logDir = os.path.join("Results/Res_results_"+str(species.replace(" ", "_"))+"/")
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
            print("Can't create logging directory:", logDir)
    logDir = os.path.join("Results/Point_results_"+str(species.replace(" ", "_"))+"/")
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
            print("Can't create logging directory:", logDir)
    # antibiotics_selected = ast.literal_eval(antibiotics)

    # print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)

    # for anti in ['mupirocin', 'penicillin', 'rifampin', 'tetracycline', 'vancomycin']:
    # for anti in antibiotics_selected:
    # save_name_meta, save_name_modelID = amr_utility.name_utility.save_name_modelID(level, species, anti)

    save_name_speciesID = 'metadata/model/id_' + str(species.replace(" ", "_")) + '.list'
    data_sub_anti = pd.read_csv(save_name_speciesID , index_col=0, dtype={'genome_id': object}, sep="\t")
    data_sub_anti = data_sub_anti.drop_duplicates()
    for strain_ID in data_sub_anti['genome_id'].to_list():
        cmd_acquired = ("python3 run_resfinder.py"
                        + " -ifa " + path+ str(strain_ID)+".fna"
                        + " -o  Res_results_"+str(species.replace(" ", "_"))+"/" + str(strain_ID)
                        + " -s \'"+ str(species)+"\'"
                        + " --min_cov 0.6"
                        + " -t 0.8"
                        + " --acquired"
                        + " --db_path_res /home/khu/AMR/benchmarking/resfinder/db_resfinder"
                        + " --blastPath /usr/bin/blastn")
        procs = run(cmd_acquired, shell=True, stdout=PIPE, stderr=PIPE,
                    check=True)

        cmd_acquired = ("python3 run_resfinder.py"
                        + " -ifa " + path + str(strain_ID) + ".fna"
                        + " -o  Point_results_"+str(species.replace(" ", "_"))+"/" + str(strain_ID)
                        + " -s \'"+ str(species)+"\'"
                        + " --min_cov 0.6"
                        + " -t 0.8"
                        + " --point"
                        + " --db_path_point /home/khu/AMR/benchmarking/resfinder/db_pointfinder"
                        + " --blastPath /usr/bin/blastn")
        procs = run(cmd_acquired, shell=True, stdout=PIPE, stderr=PIPE,
                    check=True)

        # cmd_acquired = ("python3 run_resfinder_modified_kma.py"
        #                 + " -ifa " + path + str(strain_ID) + ".fna"
        #                 + " -o  Point_kma_results/" + str(strain_ID)
        #                 + " -s \'" + str(species) + "\'"
        #                 + " --min_cov 0.6"
        #                 + " -t 0.8"
        #                 + " --point"
        #                 + " --db_path_point /home/khu/AMR/benchmarking/resfinder/db_pointfinder"
        #                 + " --kmaPath  /home/khu/AMR/benchmarking/resfinder/cge/kma/kma")
        # procs = run(cmd_acquired, shell=True, stdout=PIPE, stderr=PIPE,
        #             check=True)



def extract_info(s,l):

    data = pd.read_csv('metadata/'+str(l)+'_Species_antibiotic_FineQuality.csv', index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]# drop the species with 0 in column 'number'.
    # for training model on part of the dataset:-------------
    # data=data.loc[['Escherichia coli'],:]
    data = data.loc[s, :]
    #data.at['Mycobacterium tuberculosis', 'modelling antibiotics']=['capreomycin', 'ciprofloxacin']
    # --------------------------------------------------------
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    print(data)

    for df_species,antibiotics in zip(df_species, antibiotics):
        model(df_species,antibiotics,l)

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--l','--level',default=None, type=str, required=True,
                        help='Quality control: strict or loose')
    # parser.add_argument('-b', '--balance', dest='balance',
    #                     help='use downsampling or not ', action='store_true', )

    parser.add_argument('--s','--species', default=[], type=str,nargs='+',help='species to run: e.g.\'seudomonas aeruginosa\' \
     \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
     \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')

    #parser.set_defaults(canonical=True)
    parsedArgs=parser.parse_args()
    # parser.print_help()
    print(parsedArgs)
    extract_info(parsedArgs.s, parsedArgs.l)
    # extract_info(parsedArgs.s,parsedArgs.b,parsedArgs.l)







