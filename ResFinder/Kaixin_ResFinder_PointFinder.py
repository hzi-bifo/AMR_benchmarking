

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

def cmd(path,path_results,point_database_path,res_database_path,strain_ID,species):

    cmd_acquired = ("python3 run_resfinder.py"
                    + " -ifa " + path + str(strain_ID) + ".fna"
                    + " -o  " + path_results+ str(
                species.replace(" ", "_")) + "/" + str(strain_ID)
                    + " -s \'" + str(species) + "\'"
                    + " --min_cov 0.6"
                    + " -t 0.8"
                    + " --point"
                    + " --db_path_point " + point_database_path
                    + " --acquired"
                    + " --db_path_res "+ res_database_path
                    # + " --blastPath /usr/bin/blastn"
                    + " -u")

    return cmd_acquired


def has_handle(fpath):
    for proc in psutil.process_iter():
        try:
            for item in proc.open_files():
                if fpath == item.path:
                    return True
        except Exception:
            pass

    return False
def run_Res(path,path_results,strain_ID,species,check,check_miss):
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    point_database_path = os.path.join(fileDir, 'db_pointfinder')
    point_database_path = os.path.abspath(os.path.realpath(point_database_path))
    res_database_path = os.path.join(fileDir, 'db_resfinder')
    res_database_path = os.path.abspath(os.path.realpath(res_database_path))

    try:
        if species in ['Klebsiella pneumoniae','Escherichia coli','Staphylococcus aureus','Mycobacterium tuberculosis','Salmonella enterica',
                       'Neisseria gonorrhoeae','Enterococcus faecium','Campylobacter jejuni']:
            if check_miss==True:

                # already =os.listdir("/net/sgi/metagenomics/data/khu/benchmarking/resfinder_results/" + str(species.replace(" ", "_")) )
                # if strain_ID  not in already:
                if not has_handle((path_results + str(species.replace(" ", "_"))+'/'+str(strain_ID))):
                    already =os.listdir(path_results+ str(species.replace(" ", "_"))+'/'+str(strain_ID))
                    if 'pheno_table.txt' not in already:
                        print('missing table, nowing working on: ',strain_ID)
                        cmd_acquired = cmd(path,path_results,point_database_path,res_database_path,strain_ID,species)
                        procs = run(cmd_acquired, shell=True, stdout=PIPE, stderr=PIPE,
                                    check=True)
                        print('finished: ',strain_ID)
            else:
                cmd_acquired = cmd(path,path_results,point_database_path,res_database_path,strain_ID,species)
                procs = run(cmd_acquired, shell=True, stdout=PIPE, stderr=PIPE,
                            check=True)

        else:# PointFinder not possible for other species.
            cmd_acquired = ("python3 run_resfinder.py"
                            + " -ifa " + path + str(strain_ID) + ".fna"
                            + " -o  "+ path_results + str(
                        species.replace(" ", "_")) + "/" + str(strain_ID)
                            # + " -s \'" + str(species) + "\'"
                            + " --min_cov 0.6"
                            + " -t 0.8"
                            + " --acquired"
                            + " --db_path_res "+res_database_path
                            # + " --blastPath /usr/bin/blastn"
                            + " -u")
            procs = run(cmd_acquired, shell=True, stdout=PIPE, stderr=PIPE,
                        check=True)

    except:
        if check==True:
            print(cmd(path,strain_ID,species))
        else:
            print("Error, not finished: ",strain_ID)
            print(cmd_acquired)


def make_dir(name):
    logDir = os.path.join(name)
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
            print("Can't create logging directory:", logDir)

def determination(species,path_data,path_results,n_jobs,check,check_miss):
    # path_data = "/net/projects/BIFO/patric_genome/"
    print(species)
    logDir = os.path.join(path_results+str(species.replace(" ", "_"))+"/")
    make_dir(logDir)

    save_name_speciesID = 'metadata/model/id_' + str(species.replace(" ", "_")) + '.list'
    data_sub_anti = pd.read_csv(save_name_speciesID , index_col=0, dtype={'genome_id': object}, sep="\t")
    data_sub_anti = data_sub_anti.drop_duplicates()
    output_folder="/net/sgi/metagenomics/data/khu/benchmarking/resfinder_results/" + str(species.replace(" ", "_")) + "/"
    id_list = data_sub_anti['genome_id'].to_list()

    pool = mp.Pool(processes=n_jobs)
    pool.starmap(run_Res, zip(repeat(path_data),repeat(path_results),id_list,repeat(species),repeat(check),repeat(check_miss)))


def extract_info(s,path_sequence,path_results,n_jobs,check,check_missing):

    data = pd.read_csv('metadata/loose_Species_antibiotic_FineQuality.csv', index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]# drop the species with 0 in column 'number'.
    # for training model on part of the dataset:-------------
    # data=data.loc[['Escherichia coli'],:]
    data = data.loc[s, :]
    #data.at['Mycobacterium tuberculosis', 'modelling antibiotics']=['capreomycin', 'ciprofloxacin']
    # --------------------------------------------------------
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    print(data)
    # pool = mp.Pool(processes=5)
    # pool.starmap(determination, zip(df_species,repeat(l),repeat(n_jobs)))
    for species in df_species:
        determination(species,path_sequence,path_results,n_jobs,check,check_missing)

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--l','--level',default=None, type=str, required=True,
    #                     help='Quality control: strict or loose')
    # parser.add_argument('-b', '--balance', dest='balance',
    #                     help='use downsampling or not ', action='store_true', )
    parser.add_argument('-path_sequence', '--path_sequence', default='/net/projects/BIFO/patric_genome/', type=str,
                        required=False,
                        help='path of sequence,another option: \'/vol/projects/BIFO/patric_genome\'')
    parser.add_argument('-path_results', '--path_results',
                        default='/net/sgi/metagenomics/data/khu/benchmarking/resfinder_results/', type=str,
                        required=False,
                        help='another option: \'/vol/projects/khu/amr/benchmarking/large_temp/resfinder_results/\'')

    parser.add_argument('--s','--species', default=[], type=str,nargs='+',help='species to run: e.g.\'seudomonas aeruginosa\' \
     \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
     \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parser.add_argument('--check', dest='check', help='debug ', action='store_true', )
    parser.add_argument('--check_miss', dest='check_miss', help='process those still missing results. ', action='store_true', )
    #parser.set_defaults(canonical=True)
    parsedArgs=parser.parse_args()
    # parser.print_help()
    # print(parsedArgs)
    extract_info(parsedArgs.s,parsedArgs.path_sequence,parsedArgs.path_results,parsedArgs.n_jobs,parsedArgs.check,parsedArgs.check_miss)
    # extract_info(parsedArgs.s,parsedArgs.b,parsedArgs.l)









