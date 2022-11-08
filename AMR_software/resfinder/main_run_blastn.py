#!/usr/bin/env python3

import os
import sys
# sys.path.append('../../')
sys.path.insert(0, os.getcwd())
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from subprocess import PIPE, run
import pandas as pd
import numpy as np
import multiprocessing as mp
import argparse, tqdm
from itertools import repeat
from src.amr_utility import name_utility, file_utility

def cmd(path, path_results, point_database_path,res_database_path,strain_ID, species,threshold_point,min_cov_point):
    cmd_acquired = ("python3 ./AMR_software/resfinder/run_resfinder_blastn.py"
                    + " -ifa " + path + '/' +str(strain_ID) + ".fna"
                    + " -o  " + path_results+  str(species.replace(" ", "_"))+ '/' +str(strain_ID)
                    + " -s \'" + str(species) + "\'"
                    + " --min_cov 0.6"
                    + " -t 0.8"
                    + " --min_cov_point " + str(min_cov_point)
                    + " -t_p " + str(threshold_point)
                    + " --point"
                    + " --db_path_point " + point_database_path
                    + " --acquired"
                    + " --db_path_res " + res_database_path
                    # + " --kmaPath " + kma_path
                    + " -u")

    return cmd_acquired


def cmd_res(path, path_results, point_database_path,res_database_path,strain_ID, species,threshold_point,min_cov_point):
    cmd_acquired = ("python3 ./AMR_software/resfinder/run_resfinder_blastn.py"
                    + " -ifa " + path + '/' +str(strain_ID) + ".fna"
                    + " -o  " + path_results+ str(species.replace(" ", "_"))+ '/' +str(strain_ID)
                    + " -s \'" + str(species) + "\'"
                    + " --min_cov 0.6"
                    + " -t 0.8"
                    # + " --min_cov_point " + str(min_cov_point)
                    # + " -t_p " + str(threshold_point)
                    # + " --point"
                    # + " --db_path_point " + point_database_path
                    + " --acquired"
                    + " --db_path_res " + res_database_path
                    # + " --kmaPath " + kma_path
                    + " -u")

    return cmd_acquired



def run_Res(path,path_results,strain_ID,species,threshold_point,min_cov_point):
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    point_database_path = os.path.join(fileDir, 'AMR_software/resfinder/db_pointfinder')
    point_database_path = os.path.abspath(os.path.realpath(point_database_path))
    res_database_path = os.path.join(fileDir, 'AMR_software/resfinder/db_resfinder')
    res_database_path = os.path.abspath(os.path.realpath(res_database_path))
    # kma_path = os.path.join(fileDir, 'AMR_software/resfinder/cge/kma/kma')
    # kma_path = os.path.abspath(os.path.realpath(kma_path))
    path_results=path_results+'software_output/'
    try:
        if species in ['Klebsiella pneumoniae','Escherichia coli','Staphylococcus aureus','Mycobacterium tuberculosis','Salmonella enterica',
                       'Neisseria gonorrhoeae','Enterococcus faecium','Campylobacter jejuni']:

            cmd_acquired =cmd(path, path_results, point_database_path,res_database_path,strain_ID, species,threshold_point,min_cov_point)
            procs = run(cmd_acquired, shell=True, stdout=PIPE, stderr=PIPE,
                        check=True)

        else:# PointFinder not possible for other species.
            cmd_acquired = cmd_res(path, path_results, point_database_path, res_database_path, strain_ID, species,
                               threshold_point, min_cov_point)


            procs = run(cmd_acquired, shell=True, stdout=PIPE, stderr=PIPE,
                        check=True)

    except:
        print("Error, not finished: ",strain_ID)
        print(cmd_acquired)



def determination(species,path_data,n_jobs,temp_path,threshold_point,min_cov_point):
    path_results=temp_path
    save_name_speciesID = './data/PATRIC/meta/by_species_bq/id_' + str(species.replace(" ", "_"))
    id=np.genfromtxt(save_name_speciesID, dtype="str")
    data_sub_anti = pd.DataFrame(data=id, columns=['genome_id'],index=np.array(range(len(id))))
    data_sub_anti = data_sub_anti.drop_duplicates()
    id_list = data_sub_anti['genome_id'].to_list()

    pool = mp.Pool(processes=n_jobs)
    total_number=len(id_list)
    inputs=zip(repeat(path_data),repeat(path_results),id_list,repeat(species),repeat(threshold_point),repeat(min_cov_point))
    pool.starmap(run_Res, tqdm.tqdm(inputs, total=total_number),chunksize=1)
    pool.close()
    pool.join()


def extract_info(s,f_all,path_sequence,n_jobs,level,temp_path,threshold_point,min_cov_point):
    temp_path=temp_path+'log/software/resfinder_b/'
    file_utility.make_dir(temp_path)
    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")

    if f_all==False:
        data = data.loc[s, :]
    print(data)
    df_species = data.index.tolist()
    for species in df_species:
        determination(species,path_sequence,n_jobs,temp_path,threshold_point,min_cov_point)

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_sequence', '--path_sequence', type=str, required=True,
                        help='Path of the directory with PATRIC sequences.')
    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                        help='Directory to store temporary files.')
    parser.add_argument('-t_p', '--threshold_point', default=0.8, type=float,
                        help='Threshold for identity of Pointfinder. ')
    parser.add_argument('-l_p', '--min_cov_point', default=0.6, type=float,
                        help=' Minimum coverage of Pointfinder. ')
    parser.add_argument('-s','--species', default=[], type=str,nargs='+',help='species to run: e.g.\'seudomonas aeruginosa\' \
     \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
     \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='All the possible species')
    parser.add_argument('-l', '--level', default='loose', type=str, required=False,
                        help='Quality control: strict or loose')
    #parser.set_defaults(canonical=True)
    parsedArgs=parser.parse_args()
    # parser.print_help()
    # print(parsedArgs)
    extract_info(parsedArgs.species,parsedArgs.f_all,parsedArgs.path_sequence,parsedArgs.n_jobs,parsedArgs.level,parsedArgs.temp_path,parsedArgs.threshold_point,parsedArgs.min_cov_point)










