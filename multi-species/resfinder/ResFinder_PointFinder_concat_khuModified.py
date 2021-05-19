

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

def cmd_point(path, path_results, point_database_path, strain_ID, species):
    cmd_acquired = ("python3 ./resfinder/run_resfinder_kma.py"
                    + " -ifa " + path + '/' +str(strain_ID) + ".fna"
                    + " -o  " + path_results + str(species.replace(" ", "_")) + "/" + str(strain_ID)
                    + " -s \'" + str(species) + "\'"
                    + " --min_cov 0.6"
                    + " -t 0.8"
                    + " --point"
                    + " --db_path_point " + point_database_path
                    # + " --acquired"
                    # + " --db_path_res " + res_database_path
                    # + " --blastPath /usr/bin/blastn"
                    + " -u")

    return cmd_acquired

# def cmd(path,path_results,point_database_path,res_database_path,strain_ID,species):
#
#     cmd_acquired = ("python3 run_resfinder.py"
#                     + " -ifa " + path + str(strain_ID) + ".fna"
#                     + " -o  " + path_results+ str(
#                 species.replace(" ", "_")) + "/" + str(strain_ID)
#                     + " -s \'" + str(species) + "\'"
#                     + " --min_cov 0.6"
#                     + " -t 0.8"
#                     + " --point"
#                     + " --db_path_point " + point_database_path
#                     + " --acquired"
#                     + " --db_path_res "+ res_database_path
#                     # + " --blastPath /usr/bin/blastn"
#                     + " -u")
#
#     return cmd_acquired

def make_dir(name):
    logDir = os.path.join(name)
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
            print("Can't create logging directory:", logDir)
def has_handle(fpath):
    for proc in psutil.process_iter():
        try:
            for item in proc.open_files():
                if fpath == item.path:
                    return True
        except Exception:
            pass

    return False

def run_Res(path,path_results,strain_ID,species):
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    point_database_path = os.path.join(fileDir, 'resfinder/db_pointfinder')
    point_database_path = os.path.abspath(os.path.realpath(point_database_path))
    # res_database_path = os.path.join(fileDir, 'resfinder/db_resfinder')
    # res_database_path = os.path.abspath(os.path.realpath(res_database_path))

    try:
        cmd_acquired = cmd_point(path, path_results, point_database_path,strain_ID, species)
        procs = run(cmd_acquired, shell=True, stdout=PIPE, stderr=PIPE,
                    check=True)

    except:
        print("Error, not finished: ",strain_ID)
        print(cmd_acquired)




def determination(species,path_data,path_results,id_list,n_jobs):


    pool = mp.Pool(processes=n_jobs)
    pool.starmap(run_Res, zip(repeat(path_data),repeat(path_results),id_list,repeat(species)))






