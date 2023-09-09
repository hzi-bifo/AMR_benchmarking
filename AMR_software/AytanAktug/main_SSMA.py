#!/usr/bin/env python3

import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
# sys.path.append('../../')
sys.path.insert(0, os.getcwd())
from src.amr_utility import name_utility, file_utility,load_data
from AMR_software.AytanAktug.data_preparation import merge_scaffolds,scored_representation_blast,ResFinder_analyser_blast,merge_resfinder_pointfinder,merge_input_output_files,merge_resfinder
import argparse,json
import pandas as pd
import numpy as np
from AMR_software.AytanAktug.nn import nn_SSMA
from src.cv_folds import cluster2folds

'''Aytan-Aktug single-species , multi-anti model'''



def match_feature(species, temp_path, antibiotics, level):
    '''match features with metadata (Y), i.e. preparing X and Y for ML.'''
    _,_,path_ID_multi,path_metadata_multi,_,_,_,_,_,_,path_mutation_gene_results,path_x ,path_y,path_name=\
                    name_utility.GETname_AAfeatureSSMA('AytanAktug',level,species, '',temp_path)


    meta_s = pd.read_csv(path_metadata_multi, sep="\t", header=0, index_col=0, dtype={'id': object, 'pheno': int})
    feature_s=np.genfromtxt(path_mutation_gene_results, dtype="str")
    n_feature_s=feature_s.shape[1]-1#number of features for this species
    df_feature_s=pd.DataFrame(feature_s, index=None, columns=np.insert(np.array(np.arange(n_feature_s),dtype='object'), 0, 'id'))#,dtype={'id': object}
    df_feature_s.set_index(df_feature_s.columns[0],inplace=True)
    df_feature_s.index.to_series().to_csv(path_name,header=False, index=False,sep="\t")
    df_feature_s.to_csv(path_x,index=False,header=False, sep="\t")
    #Merge meta and pheno to make sure the use the same id list(order).
    df_feature_s_f = pd.merge(df_feature_s, meta_s, how="outer", on=["id"])
    df_phenotype_final=df_feature_s_f.loc[:, antibiotics]
    df_phenotype_final=df_phenotype_final.fillna(-1)
    df_phenotype_final.to_csv(path_y,index=False,header=False, sep="\t")



def prepare_meta(species,antibiotics,level,temp_path):
    '''
    return: each species' metadata of selected antibiotics. combined metadata of all selected species(all antibiotics).
    '''
    _,_,path_ID_multi,path_metadata_multi,_,_,_,_,_,_,_,path_x ,path_y,path_name=\
                    name_utility.GETname_AAfeatureSSMA('AytanAktug',level,species, '',temp_path)
    file_utility.make_dir(os.path.dirname(path_metadata_multi))
    metadata_pheno_all_sub=[]
    for anti in antibiotics:

        path_metadata_pheno,path_metadata_multi_temp,_,_,_ ,_,_,_,_,_,_,_,_,_=\
                    name_utility.GETname_AAfeatureSSMA('AytanAktug',level,species, anti,temp_path)

        meta_pheno_temp = pd.read_csv(path_metadata_pheno, index_col=0, dtype={'genome_id': object}, sep="\t")
        meta_pheno_temp.to_csv(path_metadata_multi_temp, sep="\t", index=False,header=False)
        metadata_pheno = pd.read_csv(path_metadata_multi_temp,  sep="\t",header=None,names=['id',anti],dtype={'id': object,'pheno':int})

        metadata_pheno_all_sub.append(metadata_pheno)
    if len(metadata_pheno_all_sub)>1:
        metadata_pheno=metadata_pheno_all_sub[0]
        for i in metadata_pheno_all_sub[1:]:
            metadata_pheno = pd.merge(metadata_pheno, i, how="outer", on=["id"])# merge antibiotics within one species
    else:
        pass#no need for merge.

    metadata_pheno.to_csv(path_metadata_multi, sep="\t", index=True, header=True)
    metadata_pheno['id'].to_csv(path_ID_multi, sep="\t", index=False, header=False)



def extract_info(path_sequence,s,level,f_all,f_phylotree,f_kma,f_pre_meta, f_pre_cluster,f_cluster_folds,f_nn_score,f_res,f_merge_mution_gene,
                 f_matching_io,f_nn,cv, epochs, learning,f_scaler,f_fixed_threshold,f_nn_base,f_optimize_score,i_CV,temp_path):
    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    if f_all == False:
        data = data.loc[s, :]
    df_species = data.index.tolist()

    for species in df_species:
        antibiotics, _, _ =  load_data.extract_info(species, False, level)

        if f_pre_meta:
            prepare_meta(species,antibiotics,level,temp_path)
        if f_pre_cluster:
            _,_,path_ID_multi,path_metadata_multi,path_cluster,path_cluster_results,path_cluster_temp,_,_ ,_,_,_ ,_,_=\
                    name_utility.GETname_AAfeatureSSMA('AytanAktug',level,species, '',temp_path)
            file_utility.make_dir(os.path.dirname(path_cluster))
            merge_scaffolds.extract_info(path_sequence,path_ID_multi,path_cluster,16)
        if f_cluster_folds:
            _,_,path_ID_multi,path_metadata_multi,path_cluster,path_cluster_results,path_cluster_temp,_,_ ,_,_,_ ,_,_=\
                    name_utility.GETname_AAfeatureSSMA('AytanAktug',level,species, '',temp_path)
            _, _, folders_sampleName = cluster2folds.prepare_folders(cv, 42, path_ID_multi,path_cluster_results, 'new')

            folds_txt=name_utility.GETname_foldsSSMA(species,level, True, False)
            print(folds_txt)
            with open(folds_txt, 'w') as f:
                json.dump(folders_sampleName, f)

        # =================================
        # 2. Analysing PointFinder results
        # Analysing ResFinder results
        # =================================
        if f_res == True:
            _,_,path_ID_multi,path_metadata_multi,path_cluster,path_cluster_results,path_cluster_temp,path_res_result,path_point_repre_results,\
           path_res_repre_results,path_mutation_gene_results,path_x ,path_y,path_name = \
                 name_utility.GETname_AAfeatureSSMA('AytanAktug',level,species, '',temp_path)
            if species in ['Klebsiella pneumoniae','Escherichia coli','Staphylococcus aureus','Mycobacterium tuberculosis','Salmonella enterica',
                       'Neisseria gonorrhoeae','Enterococcus faecium','Campylobacter jejuni']:

                scored_representation_blast.extract_info(path_res_result,path_ID_multi,path_point_repre_results,True,True)#SNP, the last para not zip format.
            ResFinder_analyser_blast.extract_info(path_res_result,path_ID_multi,path_res_repre_results,True)#GPA,the last para means not zip format.

        if f_merge_mution_gene == True:

            _,_,path_ID_multi,path_metadata_multi,path_cluster,path_cluster_results,path_cluster_temp,path_res_result,path_point_repre_results,\
                path_res_repre_results,path_mutation_gene_results,path_x ,path_y,path_name = \
                 name_utility.GETname_AAfeatureSSMA('AytanAktug',level,species, '',temp_path)
            if species in ['Klebsiella pneumoniae', 'Escherichia coli', 'Staphylococcus aureus','Mycobacterium tuberculosis', 'Salmonella enterica',
                           'Neisseria gonorrhoeae', 'Enterococcus faecium', 'Campylobacter jejuni']:
                merge_resfinder_pointfinder.extract_info(path_point_repre_results,path_res_repre_results,path_mutation_gene_results)
            else:  # only AMR gene feature
                merge_resfinder.extract_info(path_ID_multi, path_res_repre_results,path_mutation_gene_results)

        if f_matching_io == True:
            match_feature(species, temp_path, antibiotics, level)

        # =================================
        #3.  model
        # =================================
        if f_nn == True:
            _,_,path_ID_multi,path_metadata_multi,path_cluster,path_cluster_results,path_cluster_temp,path_res_result,path_point_repre_results,\
                path_res_repre_results,path_mutation_gene_results,path_x ,path_y,path_name = \
                 name_utility.GETname_AAfeatureSSMA('AytanAktug',level,species, '',temp_path)

            save_name_score,save_name_weights, save_name_loss = name_utility.GETname_AAscoreSSMA('AytanAktug',species,learning, epochs,
                     f_fixed_threshold, f_nn_base, f_optimize_score,temp_path,f_kma,f_phylotree)#if learning=0.0, and epoch = 0, it means hyper parameter selection mode.

            file_utility.make_dir(os.path.dirname(save_name_score))
            file_utility.make_dir(os.path.dirname(save_name_weights))
            file_utility.make_dir(os.path.dirname(save_name_loss))



            if f_nn_score: #wrap scores after finish 10 outer loop of nested CV.

                nn_SSMA.multiAnti_score(cv,save_name_score)  # hyperparmeter selection in inner loop of nested CV

            else: #run nested CV
                sys.stdout = open(save_name_loss+str(i_CV), 'w')
                nn_SSMA.multiAnti(species,antibiotics, level, path_x, path_y, path_name, cv,[i_CV],f_scaler, f_fixed_threshold,
                                  f_nn_base,f_phylotree,f_kma, f_optimize_score,save_name_weights,save_name_score)  # hyperparmeter selection in inner loop of nested CV #todo [,2,3,4,5,6,7,8,9]

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    #parameters for feature building
    parser.add_argument('-path_sequence', '--path_sequence', type=str,required=False,
                        help='Path of the directory with PATRIC sequences.')
    parser.add_argument('-f_pre_meta', '--f_pre_meta', dest='f_pre_meta', action='store_true',
                        help=' prepare metadata for multi-species model.')
    parser.add_argument('-f_pre_cluster', '--f_pre_cluster', dest='f_pre_cluster', action='store_true',
                        help='Prepare files for Kma clustering')
    parser.add_argument('-f_cluster_folds', '--f_cluster_folds', dest='f_cluster_folds', action='store_true',
                        help=' Generate KMA folds.')
    parser.add_argument('-f_res', '--f_res', dest='f_res', action='store_true',
                        help='Analyse Point/ResFinder results')
    parser.add_argument('-f_merge_mution_gene', '--f_merge_mution_gene', dest='f_merge_mution_gene',
                        action='store_true',
                        help=' Merging ResFinder and PointFinder results')
    parser.add_argument('-f_matching_io', '--f_matching_io', dest='f_matching_io', action='store_true',
                        help='Matching input and output results')
    parser.add_argument('-l', '--level', default='loose', type=str,
                        help='Quality control: strict or loose')

    # parameters for nn nestedCV

    parser.add_argument("-cv", "--cv_number", default=10, type=int,
                        help='CV splits number')
    parser.add_argument("-i_CV", "--i_CV", type=int,
                        help=' the number of outer CV to run.')
    parser.add_argument("-e", "--epochs", default=0, type=int,
                        help='epochs (only for output names purpose).  0 indicate using the hyperparameter optimization.')
    parser.add_argument("-learning", "--learning", default=0.0, type=float,
                        help='learning rate (only for output names purpose).  0.0 indicate using the hyperparameter optimization.')
    parser.add_argument('-f_scaler', '--f_scaler', dest='f_scaler', action='store_true',
                        help='normalize the data')
    parser.add_argument('-f_fixed_threshold', '--f_fixed_threshold', dest='f_fixed_threshold', action='store_true',
                        help='set a fixed threshod:0.5 in NN learning.')
    parser.add_argument('-f_nn', '--f_nn', dest='f_nn', action='store_true',
                        help='Run the NN model')
    parser.add_argument('-f_nn_base', '--f_nn_base', dest='f_nn_base', action='store_true',
                        help=' .')
    parser.add_argument('-f_nn_score', '--f_nn_scorer', dest='f_nn_score', action='store_true',
                        help='Wrap NN model scores from outer loops.')
    parser.add_argument('-f_optimize_score', '--f_optimize_score', default='f1_macro', type=str,
                        help='the optimizing-score for choosing the best estimator in inner loop. Choose: auc or f1_macro.')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                        help='kma based cv folders.')

    #others parameters
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
            \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
            \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                help='Directory to store temporary files.')
    # parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    # parser.add_argument('-debug', '--debug', dest='debug', action='store_true',help='debug')
    parsedArgs = parser.parse_args()
    # parser.print_help()
    # print(parsedArgs)
    extract_info(parsedArgs.path_sequence, parsedArgs.species, parsedArgs.level, parsedArgs.f_all,parsedArgs.f_phylotree,parsedArgs.f_kma,
                 parsedArgs.f_pre_meta, parsedArgs.f_pre_cluster, parsedArgs.f_cluster_folds,parsedArgs.f_nn_score, parsedArgs.f_res,
                 parsedArgs.f_merge_mution_gene, parsedArgs.f_matching_io, parsedArgs.f_nn, parsedArgs.cv_number,  parsedArgs.epochs,
                 parsedArgs.learning, parsedArgs.f_scaler, parsedArgs.f_fixed_threshold, parsedArgs.f_nn_base,parsedArgs.f_optimize_score,parsedArgs.i_CV,parsedArgs.temp_path)
