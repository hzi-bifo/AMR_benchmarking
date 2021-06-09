import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import numpy as np
import copy
import ast
import time
import amr_utility.name_utility
import amr_utility.graph_utility
import amr_utility.file_utility
import argparse
import statistics
import amr_utility.load_data
import pandas as pd
import data_preparation.merge_scaffolds_khuModified
import data_preparation.scored_representation_blast_khuModified
import data_preparation.ResFinder_analyser_blast_khuModified
import data_preparation.merge_resfinder_pointfinder_khuModified
import data_preparation.merge_input_output_files_khuModified
import data_preparation.merge_resfinder_khuModified
import resfinder.ResFinder_PointFinder_concat_khuModified
import neural_networks.Neural_networks_khuModified_hyperpara as nn_module_hyper
import neural_networks.Neural_networks_khuModified_earlys as nn_module
# import data_preparation.discrete_merge
# import neural_networks.Neural_networks_khuModified as nn_module_original
import csv
import neural_networks.cluster_folders
import subprocess
import pickle

'''
## Notes to author self:
To change if anti choosable: means the codes need to refine if anti is choosable.
## Notes to all:
This concatenated scripts depends on relevant discete parts. So fisrt run the main_discrete_merge.py with exactly the same parameters.
'''


def extract_info(path_sequence,list_species,selected_anti,level,f_all,f_pre_meta,f_phylo_prokka,f_phylo_roary,
                 f_pre_cluster,f_cluster,f_cluster_folders,f_run_res,f_res,threshold_point,min_cov_point,f_merge_mution_gene,f_matching_io,f_divers_rank,f_nn,f_nn_nCV,f_nn_all,cv, random,
                 hidden, epochs, re_epochs, learning,f_scaler,f_fixed_threshold,f_nn_base,f_optimize_score,n_jobs):
    if path_sequence=='/net/projects/BIFO/patric_genome':
        path_large_temp='/net/sgi/metagenomics/data/khu/benchmarking/phylo'
    else:
        fileDir = os.path.dirname(os.path.realpath('__file__'))
        path_large_temp = os.path.join(fileDir, 'large_temp')


    merge_name = []

    data = pd.read_csv('metadata/' + str(level) + '_multi-species_summary.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    if f_all:
        list_species = data.index.tolist()[:-1]
        data = data.loc[list_species, :]
    else:
        # --------------------------------------------------------
        # drop columns(antibotics) all zero
        # data = data.loc[:, (data != 0).any(axis=0)]
        # drop columns(antibotics) less than 2 antibiotics
        data = data.loc[list_species, :]
        data = data.loc[:, (data.sum() > 1)]
        print(data)


    All_antibiotics = data.columns.tolist()  # all envolved antibiotics # todo
    print(All_antibiotics)
    for n in list_species:
        merge_name.append(n[0] + n.split(' ')[1][0])
    merge_name = '_'.join(merge_name)  # e.g.Se_Kp_Pa
    # multi_log = './log/temp/' + str(level) + '/multi_concat/' + merge_name



    # --------------------------------------------------------
    # high level names
    multi_log,path_metadata_multi,path_metadata_pheno_multi,path_res_concat,path_point_repre_results_concat,path_res_repre_results_concat,path_mutation_gene_results_concat \
        = amr_utility.name_utility.GETname_multi_bench_concat(level, path_large_temp, merge_name,threshold_point,min_cov_point)
    amr_utility.file_utility.make_dir(multi_log)
    #1. Metadata
    if f_pre_meta==True:
        'First run main_discrete_merge.py' \
        'This scripts only move existing metadata from discrete part'
        # prepare the ID list and phenotype list for all species.
        # the same as discrete merge.
        # data_preparation.discrete_merge.prepare_meta(path_large_temp,list_species,[],level,f_all)#[] means all possible will anti.
        for s in list_species:
            print(s,': \t moving metadata from discrete part...')

            _, _, _, _, \
            _, path_metadata_s_multi_discrete, path_metadata_pheno_s_multi_discrete, _, _, _, \
            _, _, _ = \
                amr_utility.name_utility.GETname_multi_bench_multi_species(level, path_large_temp, merge_name, s)


            cmd1= 'cp %s %s' % (path_metadata_s_multi_discrete, multi_log)
            subprocess.run(cmd1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

            cmd2 = 'cp %s %s' % (path_metadata_pheno_s_multi_discrete, multi_log)
            subprocess.run(cmd2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        multi_log_dicrete,path_metadata_multi_dicrete,path_metadata_pheno_multi_dicrete,_,_,_,_,_,_,_=amr_utility.name_utility.GETname_multi_bench_multi(level, path_large_temp, merge_name)
        cmd3 = 'cp %s %s' % (path_metadata_multi_dicrete, multi_log)
        subprocess.run(cmd3, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        cmd4 = 'cp %s %s' % (path_metadata_pheno_multi_dicrete, multi_log)
        subprocess.run(cmd4, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    #2.Prepare folders: clustering or phylo-tree split.
    # copy kma clustering files from discrte merge. All species.
    if f_cluster:
        for s in list_species:
            print(s,': \t  moving kma clusters from discrete part...')

            _, _, path_cluster_results_multi, _, \
            _, _, _, _, _, _, \
            _, _, _ = \
                amr_utility.name_utility.GETname_multi_bench_multi_species(level, path_large_temp, merge_name, s)

            cmd = 'cp %s %s' % (path_cluster_results_multi, multi_log)
            subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)



    # 3. run resfinder
    if f_run_res==True:

        # path_res_concat=path_large_temp+'/resfinder_results/merge_species_t_'+str(threshold_point)+'_cov_'+str(min_cov_point)+'/'
        amr_utility.file_utility.make_dir(path_res_concat)
        ID_strain=np.genfromtxt(path_metadata_multi, dtype="str")
        # 'merge_species' corresponds to the concatenated database under db_pointfinder.
        resfinder.ResFinder_PointFinder_concat_khuModified.determination('merge_species',path_sequence,path_res_concat,ID_strain,threshold_point,min_cov_point,n_jobs)


    if f_res==True or f_merge_mution_gene == True :
        #4 & 5
        #select one species for testing, train on the rest.

        # =================================
        # 4. Analysing PointFinder results
        # Analysing ResFinder results
        # =================================

        data_preparation.scored_representation_blast_khuModified.extract_info(path_res_concat,
                                                                              path_metadata_multi,
                                                                              path_point_repre_results_concat,
                                                                              True, True)  # SNP,no zip

        data_preparation.ResFinder_analyser_blast_khuModified.extract_info(path_res_concat,
                                                                           path_metadata_multi,
                                                                           path_res_repre_results_concat,
                                                                           True)  # GPA, no zip


        # merge
        print('finish analysing')
        data_preparation.merge_resfinder_pointfinder_khuModified.extract_info(path_point_repre_results_concat,
                                                                              path_res_repre_results_concat,
                                                                              path_mutation_gene_results_concat)
    if f_divers_rank:
        # rank spceis by diversity via FastANI
        # later use the species with lowest diversity as evaluation set.
        pass
        for s in  list_species:
            _, _, _, _, \
            _, path_metadata_s_multi_discrete, path_metadata_pheno_s_multi_discrete, _, _, _, \
            _, _, _ = \
                amr_utility.name_utility.GETname_multi_bench_multi_species(level, path_large_temp, merge_name, s)
            _, _, _, _, _, _, _, _,_ ,path_ani= \
                amr_utility.name_utility.GETname_multi_bench_concat_species(level, path_large_temp, merge_name,
                                                                            str(s.replace(" ", "_")), threshold_point,
                                                                            min_cov_point)
            #prepare QUERY_LIST and REFERENCE_LIST for fastANI
            data_sub_anti = pd.read_csv(path_metadata_pheno_s_multi_discrete, dtype={'id': object}, index_col=0, sep="\t")
            data_sub_anti['genome_id_location'] = '/vol/projects/BIFO/patric_genome/' + data_sub_anti[
                'id'].astype(str) + '.fna'
            data_sub_anti['genome_id_location'].to_csv(path_metadata_s_multi_discrete + '_path', sep="\t", index=False, header=False)

            file_sub = os.path.join('./cv_folders/' + str(level) + '/multi_concat/', merge_name, str(s.replace(" ", "_")) + '_ani.sh')
            amr_utility.file_utility.make_dir(os.path.dirname(file_sub))
            run_file = open(file_sub, "w")  # to change if anti choosable
            run_file.write("#!/bin/bash")
            run_file.write("\n")
            run_file = amr_utility.file_utility.hzi_cpu_header(run_file,
                                                                str(s.replace(" ", "_")) + '_ani.sh',
                                                                20)
            cmd = 'fastANI --threads 20 --ql %s  --rl %s --matrix -o %s' % (path_metadata_s_multi_discrete+'_path',path_metadata_s_multi_discrete+'_path',path_ani)
            run_file.write(cmd)
            run_file.write("\n")
            run_file.close()


    if f_matching_io == True:
        # feature_dimension_all = pd.DataFrame(index=list_species, columns=['feature dimension'])
        count=0
        for species_testing in list_species:
            print('species_testing',species_testing)
            list_species_training=list_species[:count] + list_species[count+1 :]
            count+=1
            # print(list_species_training)
            # do a nested CV on list_species, select the best estimator for testing on the standing out species
            merge_name_train=[]
            for n in list_species_training:
                merge_name_train.append(n[0] + n.split(' ')[1][0])
            merge_name_train = '_'.join(merge_name_train)  # e.g.Se_Kp_Pa
            merge_name_test = species_testing.replace(" ", "_")

            # prepare metadata for training, testing  no need(use the single species meta data)
            path_id_train, path_metadata_train, path_point_repre_results_train, path_res_repre_results_train,\
            path_mutation_gene_results_train,path_x_y_train,_, _, _,_=\
                amr_utility.name_utility.GETname_multi_bench_concat_species(level, path_large_temp, merge_name,merge_name_train,threshold_point,min_cov_point)

            path_id_test, path_metadata_test, path_point_repre_results_test,path_res_repre_results_test, \
            path_mutation_gene_results_test,path_x_y_test,_, _, _,_ = \
                amr_utility.name_utility.GETname_multi_bench_concat_species(level, path_large_temp,merge_name, merge_name_test,threshold_point,min_cov_point)

            # path_metadata_test = multi_log + merge_name_test + '_meta'
            # path_id_test = multi_log + merge_name_test + '_id'

            #1. train
            # ------
            train_meta=[]
            for s in list_species_training:
                sub=pd.read_csv(multi_log+str(s.replace(" ", "_"))+'_meta.txt', sep="\t", index_col=0, header=0,dtype={'id': object})
                train_meta.append(sub)
            df_path_meta_train=train_meta[0]
            for i in train_meta[1:]:
                df_path_meta_train = pd.concat([df_path_meta_train, i], ignore_index=True, sort=False)

            print('checking anti order \n',df_path_meta_train)
            df_path_meta_train['id'].to_csv(path_id_train, sep="\t", index=False, header=False)# Note!!! cluster folders will index the name acoording to this ID list

            df_path_meta_train=df_path_meta_train.loc[:, np.insert(np.array(All_antibiotics,dtype='object'), 0, 'id')]
            df_path_meta_train=df_path_meta_train.fillna(-1)
            print('checking anti order \n', df_path_meta_train)
            df_path_meta_train.to_csv(path_metadata_train,sep="\t", index=False, header=False)
            # df_path_meta_train.loc[:,All_antibiotics].to_csv(path_y_train, sep="\t", index=False, header=False)
            # df_path_meta_train['id'].to_csv(path_name_train, sep="\t", index=False, header=False)

            #2. test
            # -----
            id_test=np.genfromtxt(path_id_test, dtype="str")
            df_path_meta_test=pd.DataFrame(index=np.arange(len(id_test)),columns=np.insert(np.array(All_antibiotics, dtype='object'), 0, 'id'))#initialize with the right order of anti.
            df_path_meta_test_all=pd.read_csv(multi_log+str(species_testing.replace(" ", "_"))+'_meta.txt', sep="\t", index_col=0, header=0,dtype={'id': object})
            #  add all the antibiotic completely for testing dataset. And delete the antibiotis that are not included in this set of species combination.
            print('check anti order test')
            print(df_path_meta_test_all)
            df_path_meta_test.loc[:,'id']=df_path_meta_test_all.loc[:,'id']
            for anti in All_antibiotics:
                if anti in df_path_meta_test_all.columns:
                    df_path_meta_test.loc[:, anti] = df_path_meta_test_all.loc[:, anti]

            df_path_meta_test = df_path_meta_test.fillna(-1)
            print('check anti order test')
            print(df_path_meta_test)
            df_path_meta_test.to_csv(path_metadata_test,sep="\t", index=False, header=False)# multi_log + merge_name_train + '_metaresfinder'





            #get train from whole
            id_train=np.genfromtxt(path_id_train, dtype="str")

            feature=np.genfromtxt(path_mutation_gene_results_concat, dtype="str")
            n_feature = feature.shape[1] - 1  # number of features
            # df_feature=pd.DataFrame(feature, index=None, columns=np.insert(np.array(np.arange(n_feature),dtype='object'), 0, 'id'))
            # df_feature=df_feature.set_index('id')
            df_feature = pd.DataFrame(feature[:,1:], index=feature[:,0],
                                      columns=np.array(np.arange(n_feature), dtype='object'))
            print(df_feature)
            df_feature_train = df_feature.loc[id_train,:]
            df_feature_test = df_feature.loc[id_test, :]
            df_feature_train.to_csv(path_mutation_gene_results_train,sep="\t", index=True, header=False)
            print(df_feature_train)
            df_feature_test.to_csv(path_mutation_gene_results_test,sep="\t", index=True, header=False)
            #need to check. checked.
            print(df_feature_test)
            # preparing x y
            data_preparation.merge_input_output_files_khuModified.extract_info(path_id_train,path_mutation_gene_results_train,
                                                                               path_metadata_train, path_x_y_train)

            data_preparation.merge_input_output_files_khuModified.extract_info(path_id_test,path_mutation_gene_results_test,
                                                                               path_metadata_test, path_x_y_test)
            # exit()

    if f_nn == True:
        # =================================
        # 3.  model. June8th
        # =================================
        #analyze the results from fastANI
        RANK=[]
        for s in list_species:
            _, _, _, _, _, _, _, _, _, path_ani = \
                amr_utility.name_utility.GETname_multi_bench_concat_species(level, path_large_temp, merge_name,
                                                                            str(s.replace(" ", "_")), threshold_point,
                                                                            min_cov_point)
            pass
            # todo


        exit()
        count = 0
        score_test_all = []
        for species_testing in list_species:
            #select the first of the rest species in RANK as valuation set.
            #
            rank=copy.deepcopy(RANK)
            rank.remove(species_testing)
            merge_name_eval=rank[0]
            print('the anibiotics to predict:', All_antibiotics)
            print('the species to test: ', species_testing)
            list_species_training=list_species[:count] + list_species[count+1 :]
            count += 1
            # do a nested CV on list_species, select the best estimator for testing on the standing out species
            merge_name_train=[]
            for n in list_species_training:
                merge_name_train.append(n[0] + n.split(' ')[1][0])
            merge_name_train = '_'.join(merge_name_train)  # e.g.Se_Kp_Pa
            merge_name_test = species_testing.replace(" ", "_")

            # prepare metadata for training, testing  no need(use the single species meta data)

            _, _, _, _, _,_, path_x_train, path_y_train, path_name_train,_=\
                amr_utility.name_utility.GETname_multi_bench_concat_species(level, path_large_temp, merge_name,merge_name_train,threshold_point,min_cov_point)
            _, _, _, _, _,_, path_x_test, path_y_test, path_name_test,_ = \
                amr_utility.name_utility.GETname_multi_bench_concat_species(level, path_large_temp, merge_name,merge_name_test,threshold_point,min_cov_point)
            _, _, _, _, _, _, path_x_val, path_y_val, path_name_val, _ = \
                amr_utility.name_utility.GETname_multi_bench_concat_species(level, path_large_temp, merge_name,
                                                                            merge_name_eval, threshold_point,
                                                                            min_cov_point)

            name_weights_folder = amr_utility.name_utility.GETname_multi_bench_folder_concat(merge_name,merge_name_train,level, learning,
                                                                                        epochs,f_fixed_threshold,
                                                                                        f_nn_base, f_optimize_score,threshold_point,min_cov_point)

            amr_utility.file_utility.make_dir(name_weights_folder)  # for storage of weights.


            save_name_score_concat = amr_utility.name_utility.GETname_multi_bench_save_name_score_concat(merge_name,
                                                                                                         merge_name_train,
                                                                                                         level,
                                                                                                         learning,
                                                                                                         epochs,
                                                                                                         f_fixed_threshold,
                                                                                                         f_nn_base,
                                                                                                         f_optimize_score,threshold_point,min_cov_point)

            score=nn_module_hyper.concat_eval(merge_name_train, 'all_possible_anti_concat', level, path_x_train, path_y_train,
                                    path_name_train, path_x_val, path_y_val,
                                    path_name_val, path_x_test, path_y_test,
                                    path_name_test,
                                    random, f_scaler, f_fixed_threshold,f_nn_base,
                                    f_optimize_score, save_name_score_concat, merge_name, threshold_point,
                                    min_cov_point)
            score.append(merge_name_eval)#add valuation information.
            score_test_all.append(score)
            # save the score related to the standing out testing data.
        save_name_score_concat = amr_utility.name_utility.GETname_multi_bench_save_name_score_concat(merge_name,
                                                                                                     merge_name_test,
                                                                                                     level,
                                                                                                     learning,
                                                                                                     epochs,
                                                                                                     f_fixed_threshold,
                                                                                                     f_nn_base,
                                                                                                     f_optimize_score,threshold_point,min_cov_point)
        with open(save_name_score_concat+'_TEST.pickle', 'wb') as f:  # overwrite
            pickle.dump(score_test_all, f)

    if f_nn_nCV==True:
        '''Do a nested CV on above training data, in order to show the model robustness on predicting stand-out species.'''

        count = 0
        for species_testing in list_species:
            print('the anibiotics to predict:', All_antibiotics)
            print('the species to test: ', species_testing)
            list_species_training = list_species[:count] + list_species[count + 1:]
            count += 1
            # do a nested CV on list_species, select the best estimator for testing on the standing out species
            merge_name_train = []
            for n in list_species_training:
                merge_name_train.append(n[0] + n.split(' ')[1][0])
            merge_name_train = '_'.join(merge_name_train)  # e.g.Se_Kp_Pa
            merge_name_test = species_testing.replace(" ", "_")

            # prepare metadata for training, testing  no need(use the single species meta data)

            _, _, _, _, _, _, path_x_train, path_y_train, path_name_train = \
                amr_utility.name_utility.GETname_multi_bench_concat_species(level, path_large_temp, merge_name,
                                                                            merge_name_train, threshold_point,
                                                                            min_cov_point)
            _, _, _, _, _, _, path_x_test, path_y_test, path_name_test = \
                amr_utility.name_utility.GETname_multi_bench_concat_species(level, path_large_temp, merge_name,
                                                                            merge_name_test, threshold_point,
                                                                            min_cov_point)

            name_weights_folder = amr_utility.name_utility.GETname_multi_bench_folder_concat(merge_name,
                                                                                             merge_name_train, level,
                                                                                             learning,
                                                                                             epochs, f_fixed_threshold,
                                                                                             f_nn_base,
                                                                                             f_optimize_score,
                                                                                             threshold_point,
                                                                                             min_cov_point)

            amr_utility.file_utility.make_dir(name_weights_folder)  # for storage of weights.
            path_cluster_results = []
            for s in list_species_training:
                _, _, path_cluster_results_multi, _, \
                _, _, _, _, _, _, \
                _, _, _ = \
                    amr_utility.name_utility.GETname_multi_bench_multi_species(level, path_large_temp, merge_name, s)
                path_cluster_results.append(path_cluster_results_multi)  # note: this depend on discrte parts.

            # in the eval fundtion, 2nd parameter is only used in output names.
            # Nested CV.
            # scores related to nCV
            save_name_score_concat = amr_utility.name_utility.GETname_multi_bench_save_name_score_concat(merge_name,
                                                                                                         merge_name_train,
                                                                                                         level,
                                                                                                         learning,
                                                                                                         epochs,
                                                                                                         f_fixed_threshold,
                                                                                                         f_nn_base,
                                                                                                         f_optimize_score,
                                                                                                         threshold_point,
                                                                                                         min_cov_point)

            nn_module_hyper.eval(merge_name_train, 'all_possible_anti_concat', level, path_x_train,
                                          path_y_train,
                                          path_name_train, path_cluster_results, cv,
                                          random, re_epochs, f_scaler, f_fixed_threshold, f_nn_base,
                                          f_optimize_score, save_name_score_concat, merge_name, threshold_point,
                                          min_cov_point)



    if f_nn_all==True:
        '''Do a nested CV for all species, for a comparison with multi-discrete model. Finished on June 8th.'''
        path_cluster_results = []
        for s in list_species:
            _, _, path_cluster_results_multi, _, \
            _, _, _, _, _, _, \
            _, _, _ = \
                amr_utility.name_utility.GETname_multi_bench_multi_species(level, path_large_temp, merge_name, s)
            path_cluster_results.append(path_cluster_results_multi)  # note: this depend on discrte parts.

        #todo create relevant files. Seems finished. June 8th
        _, _, _, _, \
        path_mutation_gene_results_all, path_x_y_all, path_x_all, path_y_all, path_name_all,_= \
            amr_utility.name_utility.GETname_multi_bench_concat_species(level, path_large_temp, merge_name,
                                                                        merge_name, threshold_point,
                                                                        min_cov_point)
        path_id_all=path_metadata_multi #multi_log + 'ID'
        path_metadata_all=path_metadata_pheno_multi #multi_log + 'pheno.txt'

        data_preparation.merge_input_output_files_khuModified.extract_info(path_id_all,
                                                                           path_mutation_gene_results_all,
                                                                           path_metadata_all, path_x_y_all)
        # scores related to nCV
        save_name_score_concat = amr_utility.name_utility.GETname_multi_bench_save_name_score_concat(merge_name,
                                                                                                     merge_name,
                                                                                                     level,
                                                                                                     learning,
                                                                                                     epochs,
                                                                                                     f_fixed_threshold,
                                                                                                     f_nn_base,
                                                                                                     f_optimize_score,
                                                                                                     threshold_point,
                                                                                                     min_cov_point)

        nn_module_hyper.eval(merge_name, 'all_possible_anti_concat', level, path_x_all, path_y_all,
                                      path_name_all, path_cluster_results, cv,
                                      random, re_epochs, f_scaler, f_fixed_threshold, f_nn_base,
                                      f_optimize_score, save_name_score_concat, merge_name, threshold_point,
                                      min_cov_point)


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-path_sequence', '--path_sequence', default='/net/projects/BIFO/patric_genome', type=str,
    #                     required=False,
    #                     help='path of sequence,another option: \'/vol/projects/BIFO/patric_genome\'')
    # parser.add_argument('-path_large_temp', '--path_large_temp',
    #                     default='/net/sgi/metagenomics/data/khu/benchmarking/phylo', type=str,
    #                     required=False,
    #                     help='path for large temp files/folders, another option: \'/vol/projects/khu/amr/benchmarking/large_temp\'')
    parser.add_argument('-path_sequence', '--path_sequence', default='/vol/projects/BIFO/patric_genome', type=str,
                        required=False,
                        help='path of sequence,another option: \'/net/projects/BIFO/patric_genome\'')

    parser.add_argument('-f_pre_meta', '--f_pre_meta', dest='f_pre_meta', action='store_true',
                        help=' prepare metadata for multi-species model.')

    parser.add_argument('-f_phylo_prokka', '--f_phylo_prokka', dest='f_phylo_prokka', action='store_true',
                        help='phylo-tree based split bash generating, w.r.t. Prokka.')
    parser.add_argument('-f_phylo_roary', '--f_phylo_roary', dest='f_phylo_roary', action='store_true',
                        help='phylo-tree based split bash generating, w.r.t. Roary')
    parser.add_argument('-f_pre_cluster', '--f_pre_cluster', dest='f_pre_cluster', action='store_true',
                        help='Kma cluster')
    parser.add_argument('-f_cluster', '--f_cluster', dest='f_cluster', action='store_true',
                        help='Kma cluster')  # c program
    parser.add_argument('-f_run_res', '--f_run_res', dest='f_run_res', action='store_true',
                        help='Running Point/ResFinder tools.')

    parser.add_argument('-f_cluster_folders', '--f_cluster_folders', dest='f_cluster_folders', action='store_true',
                        help='Compare new split method with old(original) method.')  # c program
    parser.add_argument('-f_res', '--f_res', dest='f_res', action='store_true',
                        help='Analyse Point/ResFinder results')
    parser.add_argument('-t_p', '--threshold_point', default=0.8, type=float,
                        help='Threshold for identity of Pointfinder. ')
    parser.add_argument('-l_p', '--min_cov_point', default=0.6, type=float,
                        help=' Minimum (breadth-of) coverage of Pointfinder. ')
    parser.add_argument('-f_divers_rank', '--f_divers_rank', dest='f_divers_rank',
                        action='store_true',
                        help='Rank species by diversity.')

    parser.add_argument('-f_merge_mution_gene', '--f_merge_mution_gene', dest='f_merge_mution_gene',
                        action='store_true',
                        help=' Merging ResFinder and PointFinder results')
    parser.add_argument('-f_matching_io', '--f_matching_io', dest='f_matching_io', action='store_true',
                        help='Matching input and output results')
    # para for nn nestedCV
    parser.add_argument('-f_nn', '--f_nn', dest='f_nn', action='store_true',
                        help='Run the NN model')
    parser.add_argument('-f_nn_nCV', '--f_nn_nCV', dest='f_nn_nCV', action='store_true',
                        help='Do a nested on the species involved in the training for predicting a stand-out species.')
    parser.add_argument('-f_nn_all', '--f_nn_all', dest='f_nn_all', action='store_true',
                        help='Do a nested CV for all species, for comparison with multi-discrete model.')
    parser.add_argument("-cv", "--cv_number", default=10, type=int,
                        help='CV splits number')
    parser.add_argument("-r", "--random", default=42, type=int,
                        help='random state related to shuffle cluster order')
    parser.add_argument("-d", "--hidden", default=200, type=int,
                        help='dimension of hidden layer')
    parser.add_argument("-e", "--epochs", default=2000, type=int,
                        help='epochs')
    parser.add_argument("-re_e", "--re_epochs", default=500, type=int,
                        help='epochs')
    parser.add_argument("-learning", "--learning", default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('-l', '--level', default='loose', type=str,
                        help='Quality control: strict or loose')
    parser.add_argument('-f_scaler', '--f_scaler', dest='f_scaler', action='store_true',
                        help='normalize the data')
    parser.add_argument('-f_fixed_threshold', '--f_fixed_threshold', dest='f_fixed_threshold', action='store_true',
                        help='set a fixed threshod:0.5.')
    parser.add_argument('-f_nn_base', '--f_nn_base', dest='f_nn_base', action='store_true',
                        help='benchmarking baseline.')
    parser.add_argument('-f_optimize_score', '--f_optimize_score', default='f1_macro', type=str,
                        help='the optimizing-score for choosing the best estimator in inner loop. Choose: auc or f1_macro.')
    parser.add_argument('-anti', '--anti', default=[], type=str, nargs='+', help='one antibioticseach time to run: e.g.\'ciprofloxacin\' \
	            \'gentamicin\' \'ofloxacin\' \'tetracycline\' \'trimethoprim\' \'imipenem\' \
	            \'meropenem\' \'amikacin\'...')#todo
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
            \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
            \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument('-n_jobs','--n_jobs', default=1, type=int, help='Number of jobs to run in parallel for Resfinder tool.')
    # parser.add_argument('-debug', '--debug', dest='debug', action='store_true',help='debug')
    parsedArgs = parser.parse_args()
    # parser.print_help()
    # print(parsedArgs)
    extract_info(parsedArgs.path_sequence,parsedArgs.species,parsedArgs.anti, parsedArgs.level, parsedArgs.f_all,
                 parsedArgs.f_pre_meta,parsedArgs.f_phylo_prokka,
                 parsedArgs.f_phylo_roary, parsedArgs.f_pre_cluster, parsedArgs.f_cluster, parsedArgs.f_cluster_folders,parsedArgs.f_run_res,
                 parsedArgs.f_res,parsedArgs.threshold_point,parsedArgs.min_cov_point,
                 parsedArgs.f_merge_mution_gene, parsedArgs.f_matching_io,parsedArgs.f_divers_rank,
                 parsedArgs.f_nn, parsedArgs.f_nn_nCV,parsedArgs.f_nn_all,parsedArgs.cv_number, parsedArgs.random, parsedArgs.hidden, parsedArgs.epochs,
                 parsedArgs.re_epochs,
                 parsedArgs.learning, parsedArgs.f_scaler, parsedArgs.f_fixed_threshold, parsedArgs.f_nn_base,parsedArgs.f_optimize_score,parsedArgs.n_jobs)
