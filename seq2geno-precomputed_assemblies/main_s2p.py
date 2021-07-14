#!/usr/bin/python
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import numpy as np
import ast
import statistics
import operator
import time
import amr_utility.name_utility
import amr_utility.graph_utility
import amr_utility.file_utility
import argparse
import amr_utility.load_data
import pandas as pd
from itertools import repeat
import multiprocessing as mp
import subprocess
import csv
import pickle
import cv_folders.cluster_folders



def extract_info(path_sequence,s,f_all,f_prepare_meta,f_tree,cv,level,n_jobs,f_finished):

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


    for species in df_species:
        amr_utility.file_utility.make_dir('log/temp/' + str(level) + '/'+str(species.replace(" ", "_")))
        amr_utility.file_utility.make_dir('log/results/' + str(level) +'/'+ str(species.replace(" ", "_")))


    if f_prepare_meta:
        #prepare the dna list accroding to S2G format.
        for species, antibiotics in zip(df_species, antibiotics):

            antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
            ALL=[]
            for anti in antibiotics:
                name,path,_,_,_,_,_=amr_utility.name_utility.s2g_GETname(level, species, anti)
                name_list = pd.read_csv(name, index_col=0, dtype={'genome_id': object}, sep="\t")
                name_list['genome_id'] = 'iso_' + name_list['genome_id'].astype(str)
                path_list=np.genfromtxt(path, dtype="str")
                name_list['path']=path_list
                pseudo=np.empty(path_list.shape[0],dtype='object')
                pseudo.fill('/vol/projects/khu/amr/seq2geno/example_sg_dataset/reads_subset/dna/CH2500.1.fastq.gz,/vol/projects/khu/amr/seq2geno/example_sg_dataset/reads_subset/dna/CH2500.2.fastq.gz')
                name_list['path_pseudo'] = pseudo
                ALL.append(name_list)
                # print(name_list)
            _, _, dna_list, assemble_list, yml_file, run_file_name,wd = amr_utility.name_utility.s2g_GETname(level, species, '')

            #combine the list for all antis
            species_dna=ALL[0]
            for i in ALL[1:]:
                species_dna = pd.merge(species_dna, i, how="outer", on=["genome_id",'path','path_pseudo'])# merge antibiotics within one species
            print(species_dna)
            species_dna_final=species_dna.loc[:,['genome_id','path']]
            species_dna_final.to_csv(assemble_list, sep="\t", index=False,header=False)
            species_pseudo = species_dna.loc[:, ['genome_id', 'path_pseudo']]
            species_pseudo.to_csv(dna_list, sep="\t", index=False, header=False)


            #prepare yml files based on a basic version at working directory

            # cmd_cp='cp seq2geno_inputs.yml %s' %wd
            # subprocess.run(cmd_cp, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            wd_results = wd + '/results'
            amr_utility.file_utility.make_dir(wd_results)

            fileDir = os.path.dirname(os.path.realpath('__file__'))
            wd_results = os.path.join(fileDir, wd_results)
            assemble_list=os.path.join(fileDir, assemble_list)
            dna_list=os.path.join(fileDir, dna_list)
            # #modify the yml file
            a_file = open("seq2geno_inputs.yml", "r")
            list_of_lines = a_file.readlines()
            list_of_lines[14] = "    %s\n" % dna_list
            list_of_lines[26] = "    %s\n" % wd_results
            list_of_lines[28] = "    %s\n" % assemble_list


            a_file = open(yml_file, "w")
            a_file.writelines(list_of_lines)
            a_file.close()


            #prepare bash scripts
            run_file = open(run_file_name, "w")
            run_file.write("#!/bin/bash")
            run_file.write("\n")
            if path_sequence == '/vol/projects/BIFO/patric_genome':
                run_file = amr_utility.file_utility.hzi_cpu_header4(run_file,
                                                                   str(species.replace(" ", "_")),
                                                                   20,'snakemake_env')
            cmd='seq2geno -f ./%s' % yml_file
            run_file.write(cmd)
            run_file.write("\n")


    if f_tree == True:

        #kma cluster: use the results in multi-species model.
        #phylo-tree: build a tree w.r.t. each species. use the roary results after the s2g is finished.
        #todo add names


        for species in df_species:
            _, _, dna_list, assemble_list, yml_file, run_file_name, wd = amr_utility.name_utility.s2g_GETname(level,
                                                                                                              species,
                                                                                                              '')

            aln=wd+ '/results/denovo/roary/core_gene_alignment.aln'
            tree=wd+ '/results/denovo/roary/nj_tree.newick'
            aln=amr_utility.file_utility.get_full_d(aln)
            tree = amr_utility.file_utility.get_full_d(tree)
            cmd='Rscript --vanilla phylo_tree.r -f %s -o %s' %(aln,tree)
            subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


    if f_finished:# delete large unnecessary tempt files(folder:spades)
        for species in df_species:
            _, _, dna_list, assemble_list, yml_file, run_file_name, wd = amr_utility.name_utility.s2g_GETname(level,
                                                                                                              species,
                                                                                                              '')
            spades_cp=amr_utility.file_utility.get_full_d(wd)+'/results/denovo/spades'
            cmd = 'rm -r %s' % (spades_cp)
            subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-path_sequence', '--path_sequence', default= '/net/projects/BIFO/patric_genome', type=str, required=False,
    #                     help='path of sequence,another option: \'/vol/projects/BIFO/patric_genome\'')
    # parser.add_argument('-path_large_temp', '--path_large_temp', default='/net/sgi/metagenomics/data/khu/benchmarking/phylo', type=str,
    #                     required=False, help='path for large temp files/folders, another option: \'/vol/projects/khu/amr/benchmarking/large_temp\'')
    parser.add_argument('-path_sequence', '--path_sequence', default='/vol/projects/BIFO/patric_genome', type=str,
                        required=False,
                        help='path of sequence,another option: \'/net/projects/BIFO/patric_genome\'')

    parser.add_argument('-l', '--level', default='loose', type=str,
                        help='Quality control: strict or loose')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument('-f_prepare_meta', '--f_prepare_meta', dest='f_prepare_meta', action='store_true',
                        help='Prepare the list files for S2G.')
    parser.add_argument('-f_finished', '--f_finished', dest='f_finished', action='store_true',
                        help='delete large unnecessary tempt files')
    parser.add_argument("-cv", "--cv", default=10, type=int,
                        help='CV splits number')
    parser.add_argument('-f_tree', '--f_tree', dest='f_tree', action='store_true',
                        help='Kma cluster')  # c program
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'Pseudomonas aeruginosa\' \
                \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
                \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parsedArgs = parser.parse_args()
    extract_info(parsedArgs.path_sequence,parsedArgs.species,parsedArgs.f_all,parsedArgs.f_prepare_meta,parsedArgs.f_tree,parsedArgs.cv,parsedArgs.level,parsedArgs.n_jobs,parsedArgs.f_finished)