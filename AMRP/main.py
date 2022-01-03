import os
import argparse
import amr_utility.name_utility
import amr_utility.graph_utility
import amr_utility.file_utility
import amr_utility.load_data
import numpy as np
import pandas as pd
import subprocess
import shutil
from pathlib import Path

def make_dir():
    logDir = os.path.join('metadata/model/strict')
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
             print("Can't create logging directory:", logDir)
    logDir = os.path.join('metadata/model/loose')
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
            print("Can't create logging directory:", logDir)
    logDir = os.path.join('metadata/quality/')
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
            print("Can't create logging directory:", logDir)
    logDir = os.path.join('metadata/balance/strict')
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
            print("Can't create logging directory:", logDir)
    logDir = os.path.join('metadata/balance/loose')
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
            print("Can't create logging directory:", logDir)
    logDir = os.path.join('log/feature/kmer')
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
             print("Can't create logging directory:", logDir)
    logDir = os.path.join('kmc')
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
            print("Can't create logging directory:", logDir)
    logDir = os.path.join('log/validation_results')
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
             print("Can't create logging directory:", logDir)
    logDir = os.path.join('log/feature/odh')
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
            print("Can't create logging directory:", logDir)
    logDir = os.path.join('log/results')
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
            print("Can't create logging directory:", logDir)

    # for path_f in ['kmc/cano8mer','kmc/non_cano8mer','kmc/cano6mer','kmc/non_cano6mer','kmc/cano10mer','kmc/non_cano10mer']:
    #     logDir = os.path.join(path_f)#for kmc tool
    #
    #     if not os.path.exists(logDir):
    #         try:
    #             os.makedirs(logDir)
    #
    #         except OSError:
    #             print("Can't create logging directory:", logDir)

def extract_info( level, s,path_sequence, f_all, f_cv_folder,f_qsub):
    # if path_sequence=='/net/projects/BIFO/patric_genome':
    #     path_large_temp='/net/sgi/metagenomics/data/khu/benchmarking/s2g2p'#todo, may need a change
    # else:
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    path_large_temp = os.path.join(fileDir, 'large_temp')
    print(path_large_temp)

    amr_utility.file_utility.make_dir(path_large_temp)

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
    if f_cv_folder:
        for species, antibiotics in zip(df_species, antibiotics):

            antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)

            for anti in antibiotics:

                p_clusters_already='/vol/projects/khu/amr/benchmarking2_kma/log/temp/loose/'+str(species.replace(" ", "_"))+'/'+str(
            anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'_clustered_90.txt'
                p_clusters= amr_utility.name_utility.GETname_folder(species,anti,level)
                amr_utility.file_utility.make_dir(os.path.dirname(p_clusters))
                shutil.copyfile(p_clusters_already, p_clusters)
    if f_qsub:
        for species, antibiotics in zip(df_species, antibiotics):
            amr_utility.file_utility.make_dir('log/qsub')
            run_file_name='log/qsub/'+str(species.replace(" ", "_"))+'_kmer.sh'
            run_file = open(run_file_name, "w")
            run_file.write("#!/bin/bash")
            run_file.write("\n")
            if path_sequence == '/vol/projects/BIFO/patric_genome':
                run_file = amr_utility.file_utility.hzi_cpu_header4(run_file,
                                                                    str(species.replace(" ", "_")),
                                                                    20, 'amr')
            cmd = 'python model_cv.py --n_jobs 20 -s \'%s\'' % species
            run_file.write(cmd)
            run_file.write("\n")



if __name__ == '__main__':
    make_dir()
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--level', default='loose', type=str, required=False,
                        help='Quality control: strict or loose')
    parser.add_argument('-f_cv_folder', '--f_cv_folder', dest='f_cv_folder',
                        help='Prepare CV folders, w.r.t. kma.', action='store_true', )
    parser.add_argument('-f_qsub', '--f_qsub', dest='f_qsub',
                        help='Prepare scriptd for qsub.', action='store_true', )
    parser.add_argument('-path_sequence', '--path_sequence', default='/vol/projects/BIFO/patric_genome', type=str,
                        required=False,
                        help='path of sequence,another option: \'/net/projects/BIFO/patric_genome\'')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
         \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
         \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parsedArgs = parser.parse_args()
    # parser.print_help()
    print(parsedArgs)
    extract_info(parsedArgs.level,parsedArgs.species,parsedArgs.path_sequence,parsedArgs.f_all,parsedArgs.f_cv_folder,parsedArgs.f_qsub)