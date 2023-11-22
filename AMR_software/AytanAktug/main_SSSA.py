#!/usr/bin/env python3
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
# sys.path.append('../../')
sys.path.insert(0, os.getcwd())
import argparse
from src.amr_utility import name_utility, file_utility,load_data
import pandas as pd
from itertools import repeat
import multiprocessing as mp
from AMR_software.AytanAktug.data_preparation import merge_scaffolds,scored_representation_blast,ResFinder_analyser_blast,merge_resfinder_pointfinder,merge_input_output_files,merge_resfinder
from AMR_software.AytanAktug.nn import nn_SSSA
import json


def MergeScaffolds(path_sequence,species,anti,temp_path,level):
    save_name_ID,save_name_meta,anti_list,path_cluster,path_cluster_results,path_cluster_temp,path_metadata_pheno,path_res_result,path_point_repre_results,\
                   path_res_repre_results,path_mutation_gene_results,path_x_y,path_x,path_y,path_name=\
                    name_utility.GETname_AAfeatureSSSA('AytanAktug',level,species, anti,temp_path)
    merge_scaffolds.extract_info(path_sequence,save_name_ID,path_cluster,16) #saved to path_temp_cluster + 'cluster'


def Res(species,anti,temp_path,level):
    save_name_ID,save_name_meta,anti_list,path_cluster,path_cluster_results,path_cluster_temp,path_metadata_pheno,path_res_result,path_point_repre_results,\
                   path_res_repre_results,path_mutation_gene_results,path_x_y,path_x,path_y,path_name=\
                    name_utility.GETname_AAfeatureSSSA('AytanAktug',level,species, anti,temp_path)

    if species in ['Klebsiella pneumoniae','Escherichia coli','Staphylococcus aureus','Mycobacterium tuberculosis','Salmonella enterica',
                   'Neisseria gonorrhoeae','Enterococcus faecium','Campylobacter jejuni']:
        scored_representation_blast.extract_info(path_res_result,save_name_ID,path_point_repre_results,True,True)#SNP, the last para not zip format.

    ResFinder_analyser_blast.extract_info(path_res_result,save_name_ID,path_res_repre_results,True)#GPA,the last para means not zip format.

def MergeMutionGene(species,anti,temp_path,level):
    save_name_ID,save_name_meta,anti_list,path_cluster,path_cluster_results,path_cluster_temp,path_metadata_pheno,path_res_result,path_point_repre_results,\
                   path_res_repre_results,path_mutation_gene_results,path_x_y,path_x,path_y,path_name=\
                    name_utility.GETname_AAfeatureSSSA('AytanAktug',level,species, anti,temp_path)

    if species in ['Klebsiella pneumoniae','Escherichia coli','Staphylococcus aureus','Mycobacterium tuberculosis','Salmonella enterica',
                   'Neisseria gonorrhoeae','Enterococcus faecium','Campylobacter jejuni']:
        merge_resfinder_pointfinder.extract_info(path_point_repre_results,path_res_repre_results,path_mutation_gene_results)
    else:#only AMR gene feature
        merge_resfinder.extract_info(save_name_ID,path_res_repre_results,path_mutation_gene_results)

def MatchingIo(species,anti,temp_path,level):
    save_name_ID,save_name_meta,anti_list,path_cluster,path_cluster_results,path_cluster_temp,path_metadata_pheno,path_res_result,path_point_repre_results,\
                   path_res_repre_results,path_mutation_gene_results,path_x_y,path_x,path_y,path_name=\
                    name_utility.GETname_AAfeatureSSSA('AytanAktug',level,species, anti,temp_path)
    merge_input_output_files.extract_info(save_name_ID,path_mutation_gene_results,path_metadata_pheno,path_x_y)

def RedirectPrint(file):
    sys.stdout = open(file, 'w')

def Evaluation(species, anti,learning, epochs, f_fixed_threshold, f_nn_base, f_optimize_score,temp_path,cv,f_scaler,f_phylotree,f_kma,level,f_cpu):
    save_name_ID,save_name_meta,anti_list,path_cluster,path_cluster_results,path_cluster_temp,path_metadata_pheno,path_res_result,path_point_repre_results,\
                           path_res_repre_results,path_mutation_gene_results,path_x_y,path_x,path_y,path_name=\
                            name_utility.GETname_AAfeatureSSSA('AytanAktug',level,species, anti,temp_path)

    save_name_score,save_name_weights, save_name_loss = name_utility.GETname_AAscoreSSSA('AytanAktug',species, anti,learning, epochs,
           f_fixed_threshold, f_nn_base, f_optimize_score,temp_path,f_kma,f_phylotree)#if learning=0.0, and epoch = 0, it means hyper parameter selection mode.

    file_utility.make_dir(os.path.dirname(save_name_score))
    file_utility.make_dir(os.path.dirname(save_name_weights))
    file_utility.make_dir(os.path.dirname(save_name_loss))

    RedirectPrint(save_name_loss)
    score=nn_SSSA.eval(species, anti, level, path_x,path_y, path_name,cv,
                         f_scaler, f_fixed_threshold, f_nn_base,f_phylotree,f_kma, f_optimize_score,
                         save_name_weights)# hyperparmeter selection in inner loop of nested CV
    with open(save_name_score, 'w') as f:
        json.dump(score, f)


def extract_info(path_sequence,s,level,f_initialize,f_pre_cluster,f_res,f_merge_mution_gene,
                 f_matching_io,f_nn,cv, epochs, learning,f_scaler,f_fixed_threshold,
                 f_nn_base,f_phylotree,f_kma,f_optimize_score,n_jobs,f_all,temp_path,f_cpu):

    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    if f_all == False:
        data = data.loc[s, :]
    df_species = data.index.tolist()

    for species in df_species:

        antibiotics, _, _ =  load_data.extract_info(species, False, level)
        if f_initialize:

            antibiotics_=[]
            for anti in antibiotics:
                anti=str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
                antibiotics_.append(anti)
                save_name_ID,save_name_meta,anti_list,path_cluster,path_cluster_results,path_cluster_temp,path_metadata_pheno,\
                path_res_result,path_point_repre_results,path_res_repre_results,path_mutation_gene_results,path_x_y,path_x,path_y,path_name=\
                    name_utility.GETname_AAfeatureSSSA('AytanAktug',level,species, anti,temp_path)
                file_utility.make_dir(os.path.dirname(path_metadata_pheno))
                file_utility.make_dir(os.path.dirname(path_cluster))
                meta_pheno = pd.read_csv(save_name_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
                meta_pheno.to_csv(path_metadata_pheno, sep="\t", index=False,header=False)

            pd.DataFrame(antibiotics_).to_csv(anti_list, sep="\t", index=False, header=False)#prepare anti_list


        if f_pre_cluster:

            pool = mp.Pool(processes=n_jobs)
            pool.starmap(MergeScaffolds,zip(repeat(path_sequence),repeat(species),antibiotics,repeat(temp_path),repeat(level)))
            pool.close()
            pool.join()



        if f_res:
            pool = mp.Pool(processes=n_jobs)
            pool.starmap(Res,zip(repeat(species),antibiotics,repeat(temp_path),repeat(level)))
            pool.close()
            pool.join()



        if f_merge_mution_gene:
            pool = mp.Pool(processes=n_jobs)
            pool.starmap(MergeMutionGene,zip(repeat(species),antibiotics,repeat(temp_path),repeat(level)))
            pool.close()
            pool.join()


        if f_matching_io:
            pool = mp.Pool(processes=n_jobs)
            pool.starmap(MatchingIo,zip(repeat(species),antibiotics,repeat(temp_path),repeat(level)))
            pool.close()
            pool.join()



        if f_nn or f_nn_base:
            if f_cpu:

                pool = mp.Pool(processes=n_jobs)
                pool.starmap(Evaluation,zip(repeat(species),antibiotics,repeat(learning),repeat(epochs),repeat(f_fixed_threshold),repeat(f_nn_base),
                                            repeat(f_optimize_score),repeat(temp_path),repeat(cv),repeat(f_scaler),repeat(f_phylotree),repeat(f_kma),repeat(level),repeat(f_cpu)))
                pool.close()
                pool.join()

            else: #GPU

                for anti in antibiotics:
                    Evaluation(species, anti,learning, epochs, f_fixed_threshold, f_nn_base, f_optimize_score,temp_path,cv,f_scaler,f_phylotree,f_kma,level,f_cpu)


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    #parameters for fearure building
    parser.add_argument('-f_initialize', '--f_initialize', dest='f_initialize', action='store_true',
                        help='Prepare some metadata files in requested forms.')
    parser.add_argument('-f_pre_cluster', '--f_pre_cluster', dest='f_pre_cluster', action='store_true',
                        help='Kma cluster bash generating')
    parser.add_argument('-f_res', '--f_res', dest='f_res', action='store_true',
                        help='Analyse Point/ResFinder results')
    parser.add_argument('-f_merge_mution_gene', '--f_merge_mution_gene', dest='f_merge_mution_gene', action='store_true',
                        help=' Merging ResFinder and PointFinder results')
    parser.add_argument('-f_matching_io', '--f_matching_io', dest='f_matching_io', action='store_true',
                        help='Matching input and output results')
    parser.add_argument('-l', '--level', default='loose', type=str,
                        help='Quality control: strict or loose')

    #parameters for nn  CV
    parser.add_argument('-f_nn', '--f_nn', dest='f_nn', action='store_true',
                        help='Run the NN model')
    parser.add_argument("-cv", "--cv_number", default=10, type=int,
                        help='CV splits number')
    parser.add_argument("-e", "--epochs", default=1000, type=int,
                        help='epochs (only for output names purpose).  0 indicate using the hyperparameter optimization.')
    parser.add_argument("-learning", "--learning", default=0.001, type=float,
                         help='learning rate (only for output names purpose).  0.0 indicate using the hyperparameter optimization.')
    parser.add_argument('-f_scaler', '--f_scaler', dest='f_scaler', action='store_true',
                        help='normalize the data')
    parser.add_argument('-f_fixed_threshold', '--f_fixed_threshold', dest='f_fixed_threshold', action='store_true',
                        help='set a fixed threshod:0.5.')
    parser.add_argument('-f_nn_base', '--f_nn_base', dest='f_nn_base', action='store_true',
                        help='AytanAktug default model setttings.')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                        help='kma based cv folders.')
    parser.add_argument('-f_optimize_score', '--f_optimize_score', default='f1_macro', type=str,
                        help='Optimize score for choosing the best estimator in inner loop.')

    #other parameters
    parser.add_argument('-path_sequence', '--path_sequence', type=str,required=False,
                        help='Path of the directory with PATRIC sequences.')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='All the possible species.')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
            \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
            \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('-n_jobs','--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parser.add_argument('-f_cpu','--f_cpu',  dest='f_cpu', action='store_true',help='NN model in CPU mode, so parallelization.')
    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                    help='Directory to store temporary files.')
    # parser.add_argument('-o', '--output_path', default='./', type=str, required=False,
    #                 help='Results folder.')
    parsedArgs = parser.parse_args()
    # parser.print_help()
    # print(parsedArgs)
    extract_info(parsedArgs.path_sequence,parsedArgs.species,parsedArgs.level,parsedArgs.f_initialize,parsedArgs.f_pre_cluster,parsedArgs.f_res,
                 parsedArgs.f_merge_mution_gene,parsedArgs.f_matching_io,parsedArgs.f_nn,parsedArgs.cv_number,parsedArgs.epochs,
                 parsedArgs.learning,parsedArgs.f_scaler,parsedArgs.f_fixed_threshold,parsedArgs.f_nn_base,parsedArgs.f_phylotree,parsedArgs.f_kma,
                 parsedArgs.f_optimize_score,parsedArgs.n_jobs,parsedArgs.f_all,parsedArgs.temp_path,parsedArgs.f_cpu)
