

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
from os import walk
from itertools import repeat
import psutil

def cmd(path, path_results, point_database_path,res_database_path, kma_path,strain_ID, species,threshold_point,min_cov_point):
    cmd_acquired = ("python3 ./resfinder/run_resfinder_kma.py"
                    + " -ifa " + path + '/' +str(strain_ID) + ".fna"
                    + " -o  " + path_results+  '/' +str(strain_ID)
                    + " -s \'" + str(species) + "\'"
                    + " --min_cov 0.6"
                    + " -t 0.8"
                    + " --min_cov_point " + str(min_cov_point)
                    + " -t_p " + str(threshold_point)
                    + " --point"
                    + " --db_path_point " + point_database_path
                    + " --acquired"
                    + " --db_path_res " + res_database_path
                    + " --kmaPath " + kma_path
                    + " -u")

    return cmd_acquired


def cmd_res(path, path_results, point_database_path,res_database_path, kma_path,strain_ID, species,threshold_point,min_cov_point):
    cmd_acquired = ("python3 ./resfinder/run_resfinder_kma.py"
                    + " -ifa " + path + '/' +str(strain_ID) + ".fna"
                    + " -o  " + path_results+ '/' +str(strain_ID)
                    + " -s \'" + str(species) + "\'"
                    + " --min_cov 0.6"
                    + " -t 0.8"
                    # + " --min_cov_point " + str(min_cov_point)
                    # + " -t_p " + str(threshold_point)
                    # + " --point"
                    # + " --db_path_point " + point_database_path
                    + " --acquired"
                    + " --db_path_res " + res_database_path
                    + " --kmaPath " + kma_path
                    + " -u")

    return cmd_acquired



def run_Res(path,path_results,strain_ID,species,threshold_point,min_cov_point):
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    point_database_path = os.path.join(fileDir, 'resfinder/db_pointfinder')
    point_database_path = os.path.abspath(os.path.realpath(point_database_path))
    res_database_path = os.path.join(fileDir, 'resfinder/db_resfinder')
    res_database_path = os.path.abspath(os.path.realpath(res_database_path))
    kma_path = os.path.join(fileDir, 'resfinder/cge/kma/kma')
    kma_path = os.path.abspath(os.path.realpath(kma_path))

    try:
        if species in ['Klebsiella pneumoniae','Escherichia coli','Staphylococcus aureus','Mycobacterium tuberculosis','Salmonella enterica',
                       'Neisseria gonorrhoeae','Enterococcus faecium','Campylobacter jejuni']:

            cmd_acquired =cmd(path, path_results, point_database_path,res_database_path,kma_path,strain_ID, species,threshold_point,min_cov_point)
            procs = run(cmd_acquired, shell=True, stdout=PIPE, stderr=PIPE,
                        check=True)

        else:# PointFinder not possible for other species.
            cmd_acquired = cmd_res(path, path_results, point_database_path, res_database_path, kma_path, strain_ID, species,
                               threshold_point, min_cov_point)


            procs = run(cmd_acquired, shell=True, stdout=PIPE, stderr=PIPE,
                        check=True)

    except:
        print("Error, not finished: ",strain_ID)
        print(cmd_acquired)


def make_dir(name):
    logDir = os.path.join(name)
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
            print("Can't create logging directory:", logDir)

def determination(species,path_data,n_jobs,threshold_point,min_cov_point):
    # path_data = "/net/projects/BIFO/patric_genome"
    print(species)
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    path_results = os.path.join(fileDir,'large_temp/resfinder_results',str(species.replace(" ", "_")))
    make_dir(path_results)


    save_name_speciesID = 'metadata/model/id_' + str(species.replace(" ", "_")) + '.list'
    data_sub_anti = pd.read_csv(save_name_speciesID , index_col=0, dtype={'genome_id': object}, sep="\t")
    data_sub_anti = data_sub_anti.drop_duplicates()
    # output_folder="/net/sgi/metagenomics/data/khu/benchmarking/resfinder_results/" + str(species.replace(" ", "_")) + "/"
    id_list = data_sub_anti['genome_id'].to_list()

    pool = mp.Pool(processes=n_jobs)
    pool.starmap(run_Res, zip(repeat(path_data),repeat(path_results),id_list,repeat(species),repeat(threshold_point),repeat(min_cov_point)))


def extract_info(s,f_all,path_sequence,n_jobs,threshold_point,min_cov_point):

    data = pd.read_csv('metadata/loose_Species_antibiotic_FineQuality.csv', index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]# drop the species with 0 in column 'number'.
    # for training model on part of the dataset:-------------
    # data=data.loc[['Escherichia coli'],:]
    if f_all==False:
        data = data.loc[s, :]
    #data.at['Mycobacterium tuberculosis', 'modelling antibiotics']=['capreomycin', 'ciprofloxacin']
    # --------------------------------------------------------
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    print(data)
    # pool = mp.Pool(processes=5)
    # pool.starmap(determination, zip(df_species,repeat(l),repeat(n_jobs)))
    for species in df_species:
        determination(species,path_sequence,n_jobs,threshold_point,min_cov_point)

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_sequence', '--path_sequence', default='/net/projects/BIFO/patric_genome/', type=str,
                        required=False,
                        help='path of sequence,another option: \'/vol/projects/BIFO/patric_genome\'')
    parser.add_argument('-t_p', '--threshold_point', default=0.8, type=float,
                        help='Threshold for identity of Pointfinder. ')
    parser.add_argument('-l_p', '--min_cov_point', default=0.6, type=float,
                        help=' Minimum (breadth-of) coverage of Pointfinder. ')
    parser.add_argument('-s','--species', default=[], type=str,nargs='+',help='species to run: e.g.\'seudomonas aeruginosa\' \
     \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
     \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')

    #parser.set_defaults(canonical=True)
    parsedArgs=parser.parse_args()
    # parser.print_help()
    # print(parsedArgs)
    extract_info(parsedArgs.species,parsedArgs.f_all,parsedArgs.path_sequence,parsedArgs.n_jobs,parsedArgs.threshold_point,parsedArgs.min_cov_point)










