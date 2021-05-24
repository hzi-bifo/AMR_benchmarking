#!/usr/bin/python
import sys
import os
sys.path.append('../')
# sys.path.insert(0, os.getcwd())
import pandas as pd
import numpy as np
from pathlib import Path
def GETsave_quality(species,level):
    save_quality='quality/GenomeFineQuality_' +str(level)+'_'+ str(species.replace(" ", "_")) + '.txt'
    save_all_quality="quality/"+str(species.replace(" ", "_"))+".csv"
    return save_all_quality,save_quality

def GETsave_name_speciesID(species,f=False):#f: flag for the location of calling this function. Flag for metadata.py
    #All the strains before quality control
    if f ==  True:
        save_name_speciesID = 'model/id_' + str(species.replace(" ", "_")) + '.list'
    else:
        save_name_speciesID='metadata/model/id_' + str(species.replace(" ", "_")) + '.list'
    return save_name_speciesID


def GETsave_name_modelID(level,species,anti,f=False):#f: flag for the location of calling this function. Flag for metadata.py
    if f == True:
        save_name_meta = 'balance/'+str(level)+'/log_' + str(species.replace(" ", "_")) + '_' + \
                         str(anti.translate(str.maketrans({'/': '_', ' ': '_'}))) + '_'
        save_name_modelID = 'model/'+str(level)+'/Data_' + str(species.replace(" ", "_")) + '_' + str(
            anti.translate(str.maketrans({'/': '_', ' ': '_'})))

    else:
        save_name_meta = 'metadata/balance/'+str(level)+'/log_' + str(species.replace(" ", "_")) + '_' + \
                         str(anti.translate(str.maketrans({'/': '_', ' ': '_'}))) + '_'
        save_name_modelID = 'metadata/model/'+str(level)+'/Data_' + str(species.replace(" ", "_")) + '_' + str(
            anti.translate(str.maketrans({'/': '_', ' ': '_'})))

    return save_name_meta,save_name_modelID

def GETsave_name_odh(species,anti,k,m,d):#for odh
    save_name_odh ='log/feature/odh/log_' + str(species.replace(" ", "_"))+'.k{}m{}d{}.hd5'.format(k, m, d)
    save_name_odh_score='../log/validation_results/log_' + str(species.replace(" ", "_"))+ str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'.k{}m{}d{}.hd5'.format(k, m, d)
    return save_name_odh,save_name_odh_score

def GETsave_name_kmer(species,canonical,k):#for kmer
    if canonical == True:
        save_name_kmer = 'log/feature/kmer/cano_' + str(species.replace(" ", "_")) + '_' + str(k) + '_mer.h5'
        save_mame_kmc = 'kmc/cano' + str(k) + 'mer/merge_'+str(k)+'mers_'

    else:
        save_name_kmer = 'log/feature/kmer/non_cano_' + str(species.replace(" ", "_")) + '_' + str(k) + '_mer.h5'
        save_mame_kmc = 'kmc/non_cano' + str(k) + 'mer/merge_'+str(k)+'mers_'

    return save_mame_kmc,save_name_kmer

def GETsave_name_score(species,anti,canonical,k):#for kmer
    if canonical == True:
        save_name_score = 'log/validation_results/cano_' + str(k) + '_mer_' +str(species.replace(" ", "_")) + '_' + str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
    else:
        save_name_score = 'log/validation_results/non_cano_' + str(k) + '_mer_'  + str(species.replace(" ", "_")) + '_' + str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
    return save_name_score



def load_metadata(SpeciesFile):#for metadata.py
    '''
    :param SpeciesFile: species list
    :return: Meta data for each strain, which belongs to the species in the parameter file
    '''
    data = pd.read_csv('PATRIC_genomes_AMR.txt', dtype={'genome_id': object, 'genome_name': object}, sep="\t")
    data['genome_name'] = data['genome_name'].astype(str)

    data['species'] = data.genome_name.apply(lambda x: ' '.join(x.split(' ')[0:2]))
    data = data.loc[:, ("genome_id", 'species', 'antibiotic', 'resistant_phenotype')]
    df_species = pd.read_csv(SpeciesFile, dtype={'genome_id': object}, sep="\t", header=0)
    info_species = df_species['species'].tolist()  # 10 species that should be modelled!
    data = data.loc[data['species'].isin(info_species)]
    data = data.dropna()
    data = data.reset_index(drop=True)  # new index now

    return data, info_species

#Note: this names may still useful for main_nn_analysis.py for the still running version of nn.
# def GETname_multi_bench_folder(species,level,learning,epochs,f_fixed_threshold):
#     #only for mkdir folders at the beginning
#     name_weights_folder='log/temp/' +str(level)+ '/'+ str(species.replace(" ", "_")) +'/lr_'+ str(learning)+'_ep_'+str(epochs)+'_fixT_'+str(f_fixed_threshold)
#     return name_weights_folder
# def GETname_multi_bench_weight(species,antibiotics,level,cv,innerCV,learning,epochs,f_fixed_threshold):
#
#     name_weights='log/temp/' +str(level)+ '/'+ str(species.replace(" ", "_")) +'/lr_'+ str(learning)+'_ep_'+str(epochs)+'_fixT_'+str(f_fixed_threshold)+'/'+\
#                  str(antibiotics.translate(str.maketrans({'/': '_', ' ': '_'}))) + '_weights_' + str(cv) + str(innerCV)
#     return name_weights
#
#
# def GETname_multi_bench_save_name_score(species,antibiotics,level,learning,epochs,f_fixed_threshold):
#     save_name_score = str(level)+'/'+str(species.replace(" ", "_"))  +'/'+\
#                       str(antibiotics.translate(str.maketrans({'/': '_', ' ': '_'})))+'_lr_'+ str(learning)+'_ep_'+str(epochs)+'_fixT_'+str(f_fixed_threshold)
#     return save_name_score
#
#
# def GETname_multi_bench_save_name_final(species,level,learning,epochs,f_fixed_threshold):
#     save_name_score_f = str(level) + '/' + str(species.replace(" ", "_")) + '/' + 'lr_' + str(
#         learning) + '_ep_' + str(epochs) + '_fixT_' + str(f_fixed_threshold)
#     return save_name_score_f
def GETname_multi_bench_folder_multi(species,level,learning,epochs,f_fixed_threshold,f_nn_base,f_optimize_score):
    #only for mkdir folders at the beginning
    if f_optimize_score=='auc':
        name_weights_folder = 'log/temp/' + str(level) + '/multi_species/' + str(species.replace(" ", "_")) + '/lr_' + str(
            learning) + '_ep_' + str(epochs)  +'_base_'+str(f_nn_base)+ '_ops_' + f_optimize_score
    else:
        name_weights_folder='log/temp/' +str(level)+ '/multi_species/'+ str(species.replace(" ", "_")) +'/lr_'+ str(learning)+'_ep_'+str(epochs)+'_fixT_'+str(f_fixed_threshold)
    return name_weights_folder
def GETname_multi_bench_folder(species,level,learning,epochs,f_fixed_threshold,f_nn_base,f_optimize_score):
    #only for mkdir folders at the beginning
    if f_optimize_score=='auc':
        name_weights_folder = 'log/temp/' + str(level) + '/' + str(species.replace(" ", "_")) + '/lr_' + str(
            learning) + '_ep_' + str(epochs)  +'_base_'+str(f_nn_base)+ '_ops_' + f_optimize_score
    else:
        name_weights_folder='log/temp/' +str(level)+ '/'+ str(species.replace(" ", "_")) +'/lr_'+ str(learning)+'_ep_'+str(epochs)+'_fixT_'+str(f_fixed_threshold)
        # in the future
        # name_weights_folder='log/temp/' +str(level)+ '/'+ str(species.replace(" ", "_")) +'/lr_'+ str(learning)+'_ep_'+str(epochs)+'_base_'+str(f_nn_base)+'_fixT_'+str(f_fixed_threshold)

    return name_weights_folder
def GETname_multi_bench_weight(species,antibiotics,level,cv,innerCV,learning,epochs,f_fixed_threshold,f_nn_base,f_optimize_score):

    if antibiotics=='all_possible_anti' or type(antibiotics)==list:#multi_species/output.
        folder = GETname_multi_bench_folder_multi(species, level, learning, epochs, f_fixed_threshold, f_nn_base,
                                            f_optimize_score)
        if type(antibiotics)==list:
            antibiotics='_'.join(antibiotics)#no use so far. maybe in the future, the user can choose antibiotics to envolve.

    else:
        folder = GETname_multi_bench_folder(species, level, learning, epochs, f_fixed_threshold, f_nn_base,
                                            f_optimize_score)
    if f_optimize_score=='auc':
        name_weights = folder + '/' + str(antibiotics.translate(str.maketrans({'/': '_', ' ': '_'}))) + '_weights_' + str(cv) + str(innerCV)
    else:
        name_weights=folder+'/'+str(antibiotics.translate(str.maketrans({'/': '_', ' ': '_'}))) + '_weights_' + str(cv) + str(innerCV)
    return name_weights


def GETname_multi_bench_save_name_score(species,antibiotics,level,learning,epochs,f_fixed_threshold,f_nn_base,f_optimize_score):
    if antibiotics=='all_possible_anti' or type(antibiotics)==list:#multi_species/output.
        folder ='log/temp/' +  str(level) +'/multi_species/' + str(species.replace(" ", "_"))

        if type(antibiotics)==list:
            antibiotics='_'.join(antibiotics)#no use so far. maybe in the future, the user can choose antibiotics to envolve.
    else:
        folder ='log/temp/' + str(level) + '/' + str(species.replace(" ", "_"))


    if f_optimize_score=='auc':
        save_name_score =folder + '/' + str(antibiotics.translate(str.maketrans({'/': '_', ' ': '_'}))) + '_lr_' + str(
            learning) + '_ep_' + str(epochs) + '_base_'+str(f_nn_base)+ '_ops_' + f_optimize_score
    else:
        save_name_score = folder +'/'+str(antibiotics.translate(str.maketrans({'/': '_', ' ': '_'})))+'_lr_'+ str(learning)+'_ep_'+str(epochs)+'_fixT_'+str(f_fixed_threshold)

        #in the future
        # save_name_score = folder + '/' + str(antibiotics.translate(str.maketrans({'/': '_', ' ': '_'}))) + '_lr_' + str(
        #     learning) + '_ep_' + str(epochs) + '_base_'+str(f_nn_base)+  '_fixT_' + str(f_fixed_threshold)
    return save_name_score


def GETname_multi_bench_save_name_final(species,antibiotics,level,learning,epochs,f_fixed_threshold,f_nn_base,f_optimize_score):
    if antibiotics == 'all_possible_anti' or type(antibiotics) == list:  # multi_species/output.
        folder ='log/results/' + str(level) + '/multi_species/' + str(species.replace(" ", "_"))
        if type(antibiotics) == list:
            antibiotics = '_'.join(
                antibiotics)  # no use so far. maybe in the future, the user can choose antibiotics to envolve.

    else:
        folder = 'log/results/' + str(level) + '/' + str(species.replace(" ", "_"))



    if f_optimize_score=='auc':
        save_name_score_f = folder +'/' + 'lr_' + str(
            learning) + '_ep_' + str(epochs) + '_base_' + str(f_nn_base) + '_ops_' + f_optimize_score
    else:
        save_name_score_f =folder + '/' + 'lr_' + str(
        learning) + '_ep_' + str(epochs) + '_fixT_' + str(f_fixed_threshold)
        # in the future
        # ave_name_score_f = str(level) + '/' + str(species.replace(" ", "_")) + '/' + 'lr_' + str(
            # learning) + '_ep_' + str(epochs) +  '_base_' + str(f_nn_base) + '_fixT_' + str(f_fixed_threshold)
    return save_name_score_f


def GETname_multi_bench_main_feature(level, species, anti,path_large_temp):
    #names of temp files
    save_name_meta,save_name_modelID = GETsave_name_modelID(level, species, anti, f=False)
    # save_name_species=str(level)+'/'+str(species.replace(" ", "_"))
    save_name_anti=str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
    name_species_anti = str(species.replace(" ", "_")) + '_' + str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
    '''
    save_name_modelID='metadata/model/'+str(level)+'/Data_' + str(species.replace(" ", "_")) + '_' + str(
            anti.translate(str.maketrans({'/': '_', ' ': '_'})))
    '''
    path_feature = './log/temp/' + str(level)+'/'+str(species.replace(" ", "_")) # all feature temp data(except large files)
    # path_res_result='/net/flashtest/scratch/khu/benchmarking/Results/'+str(species.replace(" ", "_"))#old
    if Path(path_large_temp).parts[0]=='vol': #only because on different servers
        path_res_result = '/vol/projects/khu/amr/benchmarking2_kma/large_temp/resfinder_results/' + str(
            species.replace(" ", "_"))
    else:
        path_res_result = '/net/sgi/metagenomics/data/khu/benchmarking/resfinder_results/' + str(species.replace(" ", "_"))
    path_metadata='./'+save_name_modelID #anti,species
    # path_large_temp='/net/flashtest/scratch/khu/benchmarking/Results/clustering/'+name_species_anti+'all_strains_assembly.txt'#
    path_large_temp_kma = path_large_temp+'/clustering/' + name_species_anti + 'all_strains_assembly.txt'

    path_large_temp_prokka =path_large_temp+'/prokka/'+str(species.replace(" ", "_"))
    path_large_temp_roary=path_large_temp+'/roary/' + str(level) +'/'+ name_species_anti
    path_metadata_prokka='metadata/model/id_' + str(species.replace(" ", "_"))
    path_cluster_temp=path_feature+'/clustered_90_'+save_name_anti# todo check!!! checked
    path_metadata_pheno=path_metadata+'resfinder'
    #temp results
    path_roary_results=path_large_temp+'/results_roary/'+str(level) +'/'+ name_species_anti
    path_cluster_results=path_feature+'/'+save_name_anti+'_clustered_90.txt'
    path_point_repre_results=path_feature+'/'+save_name_anti+'_mutations.txt'
    path_res_repre_results = path_feature + '/' + save_name_anti + '_acquired_genes.txt'
    path_mutation_gene_results=path_feature + '/' + save_name_anti + '_res_point.txt'
    path_x_y = path_feature + '/' + save_name_anti + '_final_'

    path_x = path_x_y+'data_x.txt'
    path_y = path_x_y+'data_y.txt'
    path_name = path_x_y + 'data_names.txt'
    return path_feature,path_res_result,path_metadata, path_large_temp_kma,path_large_temp_prokka,path_large_temp_roary,\
           path_metadata_prokka,path_cluster_temp,path_metadata_pheno,path_roary_results,path_cluster_results,path_point_repre_results,\
           path_res_repre_results,path_mutation_gene_results,path_x_y,path_x,path_y,path_name

#========================
# multi
# =======================

def GETname_multi_bench_multi(level,path_large_temp,merge_name):

    # path_large_temp_kma_multi='/net/sgi/metagenomics/data/khu/benchmarking/phylo' or '/vol/projects/khu/amr/benchmarking/large_temp'
    multi_log='./log/temp/' + str(level) + '/multi_species/' + merge_name + '/'
    path_metadata_multi = multi_log + 'ID'
    path_metadata_pheno_multi = multi_log + 'pheno'
    run_file_kma='./cv_folders/' + str(level) + '/multi_species/' + merge_name + "_kma.sh"
    run_file_roary1 ='./cv_folders/' + str(level) + '/multi_species/' +merge_name+ "_roary1.sh"
    run_file_roary2  = './cv_folders/' + str(level) + '/multi_species/' + merge_name + "_roary2.sh"
    run_file_roary3 = './cv_folders/' + str(level) + '/multi_species/' + merge_name + "_roary3.sh"
    path_x = multi_log + 'data_x.txt'
    path_y = multi_log + 'data_y.txt'
    path_name = multi_log + 'data_names.txt'

    return multi_log,path_metadata_multi,path_metadata_pheno_multi,run_file_kma,run_file_roary1,run_file_roary2,run_file_roary3,path_x,path_y,path_name

def GETname_multi_bench_multi_species(level,path_large_temp,merge_name,s):
    # multi_log,_,_,_,_,_,_=GETname_multi_bench_multi(level,path_large_temp,merge_name)
    path_large_temp_kma_multi = path_large_temp + '/clustering/' +str(level) +'/multi_species/'+ merge_name + '/' + str(s.replace(" ", "_")) + '_all_strains_assembly.txt'  # to change if anti choosable
    path_feature_multi = './log/temp/' + str(level) + '/multi_species/' + merge_name + '/'  # all feature temp data(except large files)

    path_cluster_temp_multi=path_feature_multi+'clustered_90_'+str(s.replace(" ", "_"))  # to change if anti choosable. will be removed after finishing
    path_cluster_results_multi = path_feature_multi  + str(s.replace(" ", "_")) + '_clustered_90.txt'
    # path_large_temp_prokka_multi = path_large_temp+'/prokka/multi/'+ +merge_name+'/'+str(s.replace(" ", "_"))
    path_large_temp_roary_multi = path_large_temp + '/roary/' + str(level) + '/multi_species/' +merge_name+'/'+ str(s.replace(" ", "_")) # to change if anti choosable

    path_roary_results_multi=path_large_temp+'/results_roary/'+str(level) +'/multi_species/'+merge_name+'/'+ str(s.replace(" ", "_"))

    path_metadata_s_multi = path_feature_multi + str(s.replace(" ", "_")) + '_id'
    path_metadata_pheno_s_multi=path_feature_multi+str(s.replace(" ", "_"))+'_meta'
    #-----
    # The same as single-species model name
    path_large_temp_prokka = path_large_temp + '/prokka/' + str(s.replace(" ", "_"))
    # path_metadata_prokka_multi = path_feature_multi + str(s.replace(" ", "_")) + '_id'

    if Path(path_large_temp).parts[0]=='vol': #only because on different servers
        path_res_result = '/vol/projects/khu/amr/benchmarking2_kma/large_temp/resfinder_results/' + str(
            s.replace(" ", "_"))
    else:
        path_res_result = '/net/sgi/metagenomics/data/khu/benchmarking/resfinder_results/' + str(
            s.replace(" ", "_"))
    # -----

    path_point_repre_results_multi = path_feature_multi + str(s.replace(" ", "_")) + '_mutations.txt'# to change if anti choosable
    path_res_repre_results_multi = path_feature_multi  + str(s.replace(" ", "_")) + '_acquired_genes.txt'
    path_mutation_gene_results_multi = path_feature_multi  + str(s.replace(" ", "_")) + '_res_point.txt'



    return path_large_temp_kma_multi,path_cluster_temp_multi,path_cluster_results_multi,path_large_temp_roary_multi,\
           path_roary_results_multi,path_metadata_s_multi,path_metadata_pheno_s_multi,path_large_temp_prokka,path_res_result,path_point_repre_results_multi,\
           path_res_repre_results_multi,path_mutation_gene_results_multi,path_feature_multi


#========================
# multi & concat
# =======================
def GETname_multi_bench_concat(level,path_large_temp,merge_name,threshold_point,min_cov_point):

    # path_large_temp_kma_multi='/net/sgi/metagenomics/data/khu/benchmarking/phylo' or '/vol/projects/khu/amr/benchmarking/large_temp'
    multi_log='./log/temp/' + str(level) + '/multi_concat/' + merge_name + '/'
    path_metadata_multi=multi_log + 'ID'
    path_metadata_pheno_multi=multi_log + 'pheno'
    # run_file_kma='./cv_folders/' + str(level) + '/multi_concat/' + merge_name + "_kma.sh"
    # run_file_roary1='./cv_folders/' + str(level) + '/multi_concat/' +merge_name+ "_roary1.sh"
    # run_file_roary2 = './cv_folders/' + str(level) + '/multi_concat/' + merge_name + "_roary2.sh"
    # run_file_roary3 = './cv_folders/' + str(level) + '/multi_concat/' + merge_name + "_roary3.sh"
    path_res_concat = path_large_temp + '/resfinder_results/merge_species_t_' + str(threshold_point) + '_cov_' + str(
        min_cov_point) + '/'

    path_point_repre_results = multi_log + merge_name + '_mutations.txt'  #
    path_res_repre_results = multi_log + merge_name + '_acquired_genes.txt'
    path_mutation_gene_results=multi_log + merge_name + '_res_point.txt'
    # path_x = multi_log + 'data_x.txt'
    # path_y = multi_log + 'data_y.txt'
    # path_name = multi_log + 'data_names.txt'

    return multi_log,path_metadata_multi,path_metadata_pheno_multi,path_res_concat,path_point_repre_results,path_res_repre_results,path_mutation_gene_results

def GETname_multi_bench_concat_species(level,path_large_temp,merge_name,merge_name_train):
    # path_large_temp_kma_multi='/net/sgi/metagenomics/data/khu/benchmarking/phylo' or '/vol/projects/khu/amr/benchmarking/large_temp'
    multi_log = './log/temp/' + str(level) + '/multi_concat/' + merge_name + '/'
    path_id_multi = multi_log + merge_name_train + '_id'
    path_metadata_multi = multi_log + merge_name_train + '_pheno'


    path_point_repre_results = multi_log + merge_name_train + '_mutations.txt'  #
    path_res_repre_results = multi_log + merge_name_train + '_acquired_genes.txt'
    path_mutation_gene_results = multi_log + merge_name_train + '_res_point.txt'

    path_x_y = multi_log + merge_name_train
    path_x = path_x_y + 'data_x.txt'
    path_y = path_x_y + 'data_y.txt'
    path_name = path_x_y + 'data_names.txt'

    return path_id_multi, path_metadata_multi, path_point_repre_results, path_res_repre_results,path_mutation_gene_results,path_x_y,path_x, path_y, path_name

def GETname_multi_bench_folder_concat(species,merge_name_test,level,learning,epochs,f_fixed_threshold,f_nn_base,f_optimize_score):
    #only for mkdir folders at the beginning

    name_weights_folder = 'log/temp/' + str(level) + '/multi_concat/' + str(species.replace(" ", "_")) + '/'+merge_name_test+'_lr_' + str(
        learning) + '_ep_' + str(epochs)  +'_base_'+str(f_nn_base)+ '_ops_' + f_optimize_score+'_fixT_'+str(f_fixed_threshold)

    return name_weights_folder

def GETname_multi_bench_save_name_score_concat(merge_name,merge_name_test,level,learning,epochs,f_fixed_threshold,f_nn_base,f_optimize_score):

    folder ='log/temp/' +  str(level) +'/multi_concat/' + merge_name
    # if type(antibiotics)==list:
    #     antibiotics='_'.join(antibiotics)#no use so far. maybe in the future, the user can choose antibiotics to envolve.
    save_name_score =folder + '/' + merge_name_test + '_lr_' + str(
        learning) + '_ep_' + str(epochs) + '_base_'+str(f_nn_base)+ '_ops_' + f_optimize_score+'_fixT_'+str(f_fixed_threshold)

    return save_name_score

def GETname_multi_bench_save_name_concat_final(merge_name,merge_name_test,level,learning,epochs,f_fixed_threshold,f_nn_base,f_optimize_score):

    folder ='log/results/' + str(level) + '/multi_concat/' + merge_name
    # if type(antibiotics) == list:
    #     antibiotics = '_'.join(antibiotics)  # no use so far. maybe in the future, the user can choose antibiotics to envolve.

    save_name_score_f = folder + '/' + merge_name_test +'_lr_' + str(learning) + '_ep_' + str(epochs) +'_base_' + \
                        str(f_nn_base) +  '_ops_' + f_optimize_score+'_fixT_' + str(f_fixed_threshold)
    return save_name_score_f


'''
def old(species,anti,canonical,k):
 
    if canonical == True:

        save_name_score_old = 'log/validation_results/cano_' +str(
            species.replace(" ", "_")) + '_' + str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))

    else:
       save_name_score_old = 'log/validation_results/non_cano_' + str(
            species.replace(" ", "_")) + '_' + str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))

    return save_name_score_old
def GetSaveName(species,anti,canonical,k):

    # if '/' in anti:
    save_name = 'log/meta/log_' + str(species.replace(" ", "_")) + '_' + str(anti.translate(str.maketrans({'/': '_', ' ': '_'}))) + '_'
    save_name_model = 'log/model/Data_' + str(species.replace(" ", "_")) + '_' + str(anti.replace("/", "_"))
    if canonical== True:
        save_name_kmer='log/feature/kmer/cano_'+ str(species.replace(" ", "_")) + '_' + str(k) + '_mer.txt'
        save_name_val= 'log/validation_results/cano_' + str(species.replace(" ", "_")) + '_' + str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
    else:
        save_name_kmer = 'log/feature/kmer/non_cano_' + str(species.replace(" ", "_")) + '_' + str(k) + '_mer.txt'
        save_name_val =  'log/validation_results/non_cano_' + str(species.replace(" ", "_")) + '_'+ str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
    # else:
    #     save_name = 'log/meta/log_' + str(species.replace(" ", "_")) + '_' + str(anti) + '_'
    #     save_name_model = 'log/model/Data_' + str(species.replace(" ", "_")) + '_' + str(anti)
    #     if canonical== True:
    #         save_name_kmer='log/feature/kmer/cano_'+ str(species.replace(" ", "_")) + '_' + str(k) + '_mer.txt'
    #         save_name_val  = 'log/validation_results/cano' + str(species.replace(" ", "_")) + '_' + str(anti)
    #     else:
    #         save_name_kmer = 'log/feature/kmer/non_cano_' + str(species.replace(" ", "_")) + '_' + str(k) + '_mer.txt'
    #         save_name_val = 'log/validation_results/non_cano' + str(species.replace(" ", "_")) + '_' + str(anti)
    return save_name,save_name_model,save_name_kmer,save_name_val
'''

# def GetSaveName(species,anti,canonical,k):
#     '''
#
#       :param species: str
#       :param anti: str
#       :param canonical: Boolean
#       :return: save_name: the metadata w.r.t. species and antibiotic. No use and just check
#       save_name_model: the id and pheotype w.r.t. species and antibiotic.
#       save_kmer_name: the kmer w.r.t. spexies.
#       save_name_val: model validation results.
#       '''
#     save_name_meta = 'log/meta/log_' + str(species.replace(" ", "_")) + '_' + str(anti.translate(str.maketrans({'/': '_', ' ': '_'}))) + '_'
#     save_name_modelID = 'log/model/Data_' + str(species.replace(" ", "_")) + '_' + str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
#
#     if canonical == True:
#
#         save_name_score = 'log/validation_results/cano_' + str(k) + '_mer_' +str(species.replace(" ", "_")) + '_' + str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
#     else:
#
#         save_name_score = 'log/validation_results/non_cano_' + str(k) + '_mer_'  + str(species.replace(" ", "_")) + '_' + str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
#     return save_name_meta,save_name_modelID,save_name_score