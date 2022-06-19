import amr_utility.name_utility
import amr_utility.file_utility
import argparse
import json,pickle
import statistics
import amr_utility.load_data
import pandas as pd
import numpy as np
import os
import data_preparation.merge_scaffolds_khuModified
import data_preparation.scored_representation_blast_khuModified
import data_preparation.ResFinder_analyser_blast_khuModified
import data_preparation.merge_resfinder_pointfinder_khuModified
import data_preparation.merge_input_output_files_khuModified
import data_preparation.merge_resfinder_khuModified
import neural_networks.Neural_networks_khuModified_hyperpara as nn_module_hyper
import neural_networks.nn_multiA as nn_module_multiA
import neural_networks.cluster_folders as pre_cluster_folders
import cv_folders.create_phylotree
import neural_networks.cluster_folders

'''single-s , multi-anti model'''

def prepare_meta(multi_log,path_large_temp,species,antibiotics,level):
    '''
    :param path_large_temp: path for storage large intermediate files
    :param species
    :param level: QC
    :return: each species' metadata of selected antibitocs. combined metadata of all selected speceis(all antibiotics).
    '''

    metadata_pheno_all_sub=[]
    for anti in antibiotics:
        path_feature, path_res_result, path_metadata, path_large_temp_kma, path_large_temp_prokka, path_large_temp_roary, \
        path_metadata_prokka, path_cluster_temp, path_metadata_pheno, path_roary_results, path_cluster_results, path_point_repre_results, \
        path_res_repre_results, path_mutation_gene_results, path_x_y, path_x, path_y, path_name = \
            amr_utility.name_utility.GETname_multi_bench_main_feature(level, species, anti, path_large_temp)#TODO:name

        metadata_pheno = pd.read_csv(path_metadata_pheno,  sep="\t",header=None,names=['id',anti],dtype={'id': object,'pheno':int})
        #
        # print(metadata_pheno)
        # print('----------------------')
        metadata_pheno_all_sub.append(metadata_pheno)
    if len(metadata_pheno_all_sub)>1:
        metadata_pheno=metadata_pheno_all_sub[0]
        for i in metadata_pheno_all_sub[1:]:
            metadata_pheno = pd.merge(metadata_pheno, i, how="outer", on=["id"])# merge antibiotics within one species
            # print(metadata_pheno)
            # print('************')
    else:
        pass#no need for merge.
    metadata_pheno.to_csv(multi_log+str(species.replace(" ", "_"))+'_meta.txt', sep="\t", index=True, header=True)
    metadata_pheno['id'].to_csv(multi_log+str(species.replace(" ", "_"))+'_id', sep="\t", index=False, header=False)


def match_feature(species, path_large_temp, antibiotics, level,cv):

    # for anti in antibiotics:
    multi_log,path_ID_multi,path_metadata_multi,_,_,_,_,_,_,path_x ,path_y,path_name\
        = amr_utility.name_utility.GETname_multiAnti(level, path_large_temp, species,'') #3rdMay2022.name updated.
    #todo
    # path_metadata_s_multi
    meta_s = pd.read_csv(path_metadata_multi, sep="\t", header=0, index_col=0, dtype={'id': object, 'pheno': int})

    path_large_temp_kma_multi, path_cluster_temp_multi, path_cluster_results_multi, path_large_temp_roary_multi, \
            path_roary_results_multi, path_metadata_s_multi, path_metadata_pheno_s_multi, path_large_temp_prokka, path_res_result, path_point_repre_results_multi, \
            path_res_repre_results_multi, path_mutation_gene_results_multi, path_feature_multi = \
                amr_utility.name_utility.GETname_multiAnti_feature(level, path_large_temp, species)
    feature_s=np.genfromtxt(path_mutation_gene_results_multi, dtype="str")
    # print(path_mutation_gene_results_multi)
    print(feature_s.shape)
    n_feature_s=feature_s.shape[1]-1#number of features for this species
    df_feature_s=pd.DataFrame(feature_s, index=None, columns=np.insert(np.array(np.arange(n_feature_s),dtype='object'), 0, 'id'))#,dtype={'id': object}
    print(df_feature_s)
    df_feature_s.set_index(df_feature_s.columns[0],inplace=True)
    print(df_feature_s)
    df_feature_s.index.to_series().to_csv(path_name,header=False, index=False,sep="\t")
    df_feature_s.to_csv(path_x,index=False,header=False, sep="\t")

    #Merge meta and pheno to make sure the use the same id list(order) as df_feature_s. note: not  meta_s.
    df_feature_s_f = pd.merge(df_feature_s, meta_s, how="outer", on=["id"])
    df_phenotype_final=df_feature_s_f.loc[:, antibiotics]
    df_phenotype_final=df_phenotype_final.fillna(-1)
    print(df_phenotype_final)
    df_phenotype_final.to_csv(path_y,index=False,header=False, sep="\t")




def prepare_folds(multi_log,species,path_sequence,path_large_temp_kma,path_metadata_s_multi,path_cluster_temp_multi,path_cluster_results_multi):
    '''
    comcatenate folds from single-s model. rm the duplicates.
    '''

    data_preparation.merge_scaffolds_khuModified.extract_info(path_sequence,path_metadata_s_multi,path_large_temp_kma,16)
    #prepare the qsub scripts
    path_to_kma_clust = ("kma_clustering")
    # if Neisseria gonorrhoeae, then use  -ht 1.0 -hq 1.0, otherwise, all genomes will be clustered into one group #todo check
    run_file = open(multi_log+ "kma.sh", "w")
    run_file.write("#!/bin/bash")
    run_file.write("\n")
    run_file = amr_utility.file_utility.hzi_cpu_header(run_file,str(species.replace(" ", "_")) + "_kma",1)
    run_file.write("\n")

    # run_file.write("(")
    if species in ['Neisseria gonorrhoeae','Mycobacterium tuberculosis','Klebsiella pneumoniae']:
        cmd="cat %s | %s -i -- -k 16 -Sparse - -ht 0.98 -hq 0.98 -NI -o %s &> %s"\
             %(path_large_temp_kma ,path_to_kma_clust,path_cluster_temp_multi,path_cluster_results_multi)

    else:
        cmd = "cat %s | %s -i -- -k 16 -Sparse - -ht 0.9 -hq 0.9 -NI -o %s &> %s" \
              % (path_large_temp_kma, path_to_kma_clust, path_cluster_temp_multi, path_cluster_results_multi)
    run_file.write(cmd)
    # run_file.write("\n")
    # run_file.write("wait")
    run_file.write("\n")
    # run_file.write('rm '+ path_large_temp_kma )
    run_file.write("\n")
    # run_file.write('rm ' + path_cluster_temp+'.seq.b '+path_cluster_temp+'.length.b '+path_cluster_temp+'.name '+path_cluster_temp+'.comp.b')
    run_file.write("echo \" running... \"")
    run_file.close()


def run(species,path_sequence,path_large_temp,list_species,level,f_all,f_pre_meta,f_phylo_prokka,f_phylo_roary,
        f_pre_cluster,f_cluster,f_cluster_folders,f_res,f_merge_mution_gene,f_matching_io,f_nn,cv, seed,
                 hidden, epochs, re_epochs, learning,f_scaler,f_fixed_threshold,f_nn_base,f_optimize_score,f_phylotree,f_random,f_qsub):
    print(species)
    antibiotics, _, _ = amr_utility.load_data.extract_info(species, False, level)
    multi_log,path_ID_multi,path_metadata_multi,_,_,_,_,_,_,path_x ,path_y,path_name\
            = amr_utility.name_utility.GETname_multiAnti(level, path_large_temp, species,'') #3rdMay2022.name updated.

    if f_pre_meta==True:
        prepare_meta(multi_log,path_large_temp,species,antibiotics,level)#[] means all possible will anti.


    # =================================
    # 1.folds preparing
    # =================================

    if f_pre_cluster == True:
        path_large_temp_kma_multi, path_cluster_temp_multi, path_cluster_results_multi, path_large_temp_roary_multi, \
        path_roary_results_multi, path_metadata_s_multi, path_metadata_pheno_s_multi, path_large_temp_prokka, path_res_result, path_point_repre_results_multi, \
        path_res_repre_results_multi, path_mutation_gene_results_multi, path_feature_multi = \
            amr_utility.name_utility.GETname_multiAnti_feature(level, path_large_temp,species)
        prepare_folds(multi_log,species, path_sequence,path_large_temp_kma_multi,path_metadata_s_multi,path_cluster_temp_multi,path_cluster_results_multi )

    # =================================
    # 2. Analysing PointFinder results
    # Analysing ResFinder results
    # =================================
    if f_res == True:
        # for s in list_species:
        #     print(s)

        path_large_temp_kma_multi, path_cluster_temp_multi, path_cluster_results_multi, path_large_temp_roary_multi, \
        path_roary_results_multi, path_metadata_s_multi, path_metadata_pheno_s_multi, path_large_temp_prokka, path_res_result, path_point_repre_results_multi, \
        path_res_repre_results_multi, path_mutation_gene_results_multi, path_feature_multi = \
            amr_utility.name_utility.GETname_multiAnti_feature(level, path_large_temp,species)
        if species in ['Klebsiella pneumoniae','Escherichia coli','Staphylococcus aureus','Mycobacterium tuberculosis','Salmonella enterica',
                   'Neisseria gonorrhoeae','Enterococcus faecium','Campylobacter jejuni']:

            data_preparation.scored_representation_blast_khuModified.extract_info(path_res_result, path_metadata_s_multi,
                                                                                  path_point_repre_results_multi, True,True)#SNP,no zip
        data_preparation.ResFinder_analyser_blast_khuModified.extract_info(path_res_result, path_metadata_s_multi,
                                                                           path_res_repre_results_multi,True)  # GPA,no zip


    if f_merge_mution_gene == True:
        # for s in list_species:
        path_large_temp_kma_multi, path_cluster_temp_multi, path_cluster_results_multi, path_large_temp_roary_multi, \
        path_roary_results_multi, path_metadata_s_multi, path_metadata_pheno_s_multi, path_large_temp_prokka, path_res_result, path_point_repre_results_multi, \
        path_res_repre_results_multi, path_mutation_gene_results_multi, path_feature_multi = \
            amr_utility.name_utility.GETname_multiAnti_feature(level, path_large_temp, species)
        if species in ['Klebsiella pneumoniae', 'Escherichia coli', 'Staphylococcus aureus',
                       'Mycobacterium tuberculosis', 'Salmonella enterica',
                       'Neisseria gonorrhoeae', 'Enterococcus faecium', 'Campylobacter jejuni']:
            data_preparation.merge_resfinder_pointfinder_khuModified.extract_info(path_point_repre_results_multi,
                                                                                  path_res_repre_results_multi,
                                                                                  path_mutation_gene_results_multi)
        else:  # only AMR gene feature
            data_preparation.merge_resfinder_khuModified.extract_info(path_metadata_s_multi, path_res_repre_results_multi,
                                                                      path_mutation_gene_results_multi)



    if f_matching_io == True:
        # print('Different from single-s model.')
        match_feature(species, path_large_temp, antibiotics, level,cv)
        # path_large_temp_kma_multi, path_cluster_temp_multi, path_cluster_results_multi, path_large_temp_roary_multi, \
        # path_roary_results_multi, path_metadata_s_multi, path_metadata_pheno_s_multi, path_large_temp_prokka, path_res_result, path_point_repre_results_multi, \
        # path_res_repre_results_multi, path_mutation_gene_results_multi, path_feature_multi = \
        #     amr_utility.name_utility.GETname_multiAnti_feature(level, path_large_temp, species)
        #
        # data_preparation.merge_input_output_files_khuModified.extract_info(path_ID_multi,path_mutation_gene_results_multi,path_metadata_multi,multi_log)


    # =================================
    #3.  model
    # =================================
    if f_nn == True:
        path_large_temp_kma_multi, path_cluster_temp_multi, path_cluster_results_multi, path_large_temp_roary_multi, \
        path_roary_results_multi, path_metadata_s_multi, path_metadata_pheno_s_multi, path_large_temp_prokka, path_res_result, path_point_repre_results_multi, \
        path_res_repre_results_multi, path_mutation_gene_results_multi, path_feature_multi = \
            amr_utility.name_utility.GETname_multiAnti_feature(level, path_large_temp, species)
        name_weights_folder = amr_utility.name_utility.GETname_multiAnti_folder_multi(species,level, learning, epochs,
                                                                                   f_fixed_threshold,f_nn_base,f_phylotree,f_random,f_optimize_score)
        print(name_weights_folder)
        amr_utility.file_utility.make_dir(name_weights_folder)#for storage of weights.


        save_name_score = amr_utility.name_utility.GETname_multi_bench_save_name_score(species,antibiotics,
                                                                                       level,
                                                                                      0.0,0,
                                                                                       f_fixed_threshold,
                                                                                       f_nn_base,
                                                                                       f_optimize_score) #todo check



        # nn_module_multiA.multiAnti(species,antibiotics, level, path_x, path_y, path_name, path_mutation_gene_results_multi, cv, seed,
        #                      re_epochs, f_scaler, f_fixed_threshold, f_nn_base,f_phylotree,f_random, f_optimize_score, save_name_score,0.0,0, None,
        #                      None, None,'res')  # hyperparmeter selection in inner loop of nested CV #todo check


        score=nn_module_hyper.multiAnti(species,antibiotics, level, path_x, path_y, path_name, path_mutation_gene_results_multi, cv, seed,
                             re_epochs, f_scaler, f_fixed_threshold, f_nn_base,f_phylotree,f_random, f_optimize_score, save_name_score,0.0,0, None,
                             None, None,'res')  # hyperparmeter selection in inner loop of nested CV #todo check

        if f_phylotree:
            with open(save_name_score + '_all_score_Tree.pickle', 'wb') as f:  # overwrite
                pickle.dump(score, f)
        elif f_random:
            with open(save_name_score + '_all_score_Random.pickle', 'wb') as f:
                pickle.dump(score, f)
        else:
            with open(save_name_score + '_all_score.pickle', 'wb') as f:
                pickle.dump(score, f)

    if f_cluster_folders == True:
        #analysis the number of each species' samples in each folder.

        # for species in list_species:
        path_large_temp_kma_multi, path_cluster_temp_multi, path_cluster_results_multi, path_large_temp_roary_multi, \
            path_roary_results_multi, path_metadata_s_multi, path_metadata_pheno_s_multi, path_large_temp_prokka, path_res_result, path_point_repre_results_multi, \
            path_res_repre_results_multi, path_mutation_gene_results_multi, path_feature_multi = amr_utility.name_utility.GETname_multiAnti_feature(level, path_large_temp, species)


        folders_sampleIndex,_,folders_sampleName= neural_networks.cluster_folders.prepare_folders(cv, seed, path_name, path_cluster_results_multi,'new')

        amr_utility.file_utility.make_dir('./cv_folders/' + str(level) + '/multi_anti/')
        folds_txt='./cv_folders/' + str(level) + '/multi_anti/'+str(species.replace(" ", "_"))+'_KMA_cv.json'

        with open(folds_txt, 'w') as f:
                json.dump(folders_sampleName, f) #only for supplemental 4
        folds_p='./cv_folders/' + str(level) + '/multi_anti/'+str(species.replace(" ", "_"))+'_kma.pickle'

        with open(folds_p, 'wb') as f:
            pickle.dump(folders_sampleIndex, f)

    if f_qsub:
        amr_utility.file_utility.make_dir('log/qsub/multi_anti')
        for i_outerCV in range(cv):
            run_file_name='log/qsub/multi_anti/'+str(species.replace(" ", "_"))+str(i_outerCV)+'.sh'

            run_file = open(run_file_name, "w")
            run_file.write("#!/bin/bash")
            run_file.write("\n")
            #
            run_file = amr_utility.file_utility.hzi_cpu_header5(run_file,
                                                                    str(species.replace(" ", "_"))+str(i_outerCV),
                                                                    'multi_bench_torch','t4')
            # run_file = amr_utility.file_utility.hzi_cpu_header4(run_file,
            #                                                         str(species.replace(" ", "_"))+str(i_outerCV),1,
            #                                                         'multi_bench','all.q')
            cmd = 'python main_multi_anti_parallel.py -f_nn -f_fixed_threshold -s "%s" -i_CV %s' % (species,i_outerCV)
            run_file.write(cmd)
            run_file.write("\n")




def extract_info(path_sequence,list_species,level,f_all,f_phylotree,f_random,f_pre_meta,f_phylo_prokka,f_phylo_roary,
                 f_pre_cluster,f_cluster,f_cluster_folders,f_res,f_merge_mution_gene,f_matching_io,f_nn,cv, seed,
                 hidden, epochs, re_epochs, learning,f_scaler,f_fixed_threshold,f_nn_base,f_optimize_score,f_qsub):
    data = pd.read_csv('metadata/' + str(level) + '_Species_antibiotic_FineQuality.csv', index_col=0,
                       dtype={'genome_id': object},sep="\t")

    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    # for training model on part of the dataset:-------------
    # data=data.loc[['Escherichia coli'],:]
    if f_all ==False:
        data = data.loc[list_species, :]
    # --------------------------------------------------------
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()

    if path_sequence=='/net/projects/BIFO/patric_genome':
        path_large_temp='/net/sgi/metagenomics/data/khu/benchmarking/phylo'
    else:
        fileDir = os.path.dirname(os.path.realpath('__file__'))
        path_large_temp = os.path.join(fileDir, 'large_temp')
        amr_utility.file_utility.make_dir(path_large_temp)

    for species in df_species:
        multi_log = './log/temp/' + str(level) + '/multi_anti/' + str(species.replace(" ", "_"))
        amr_utility.file_utility.make_dir(multi_log)
        run(species,path_sequence,path_large_temp,list_species,level,f_all,f_pre_meta,f_phylo_prokka,f_phylo_roary,f_pre_cluster,f_cluster,f_cluster_folders,f_res,f_merge_mution_gene,f_matching_io,f_nn,cv, seed,
                 hidden, epochs, re_epochs, learning,f_scaler,f_fixed_threshold,f_nn_base,f_optimize_score,f_phylotree,f_random,f_qsub)



if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_sequence', '--path_sequence', default='/vol/projects/BIFO/patric_genome', type=str,
                        required=False,
                        help='path of sequence,another option: \'/net/projects/BIFO/patric_genome\'')

    # parser.add_argument('-path_large_temp', '--path_large_temp',
    #                     default='/net/sgi/metagenomics/data/khu/benchmarking/phylo', type=str,
    #                     required=False,
    #                     help='path for large temp files/folders, another option: \'/vol/projects/khu/amr/benchmarking/large_temp\'')
    # parser.add_argument('-path_large_temp', '--path_large_temp',
    #                     default='./large_temp', type=str,
    #                     required=False,
    #                     help='path for large temp files/folders, another option: \'/net/sgi/metagenomics/data/khu/benchmarking/phylo\'')
    parser.add_argument('-f_pre_meta', '--f_pre_meta', dest='f_pre_meta', action='store_true',
                        help=' prepare metadata for multi-species model.')
    parser.add_argument('-f_qsub', '--f_qsub', dest='f_qsub',
                        help='Prepare scriptd for qsub.', action='store_true', )
    parser.add_argument('-f_phylo_prokka', '--f_phylo_prokka', dest='f_phylo_prokka', action='store_true',
                        help='phylo-tree based split bash generating, w.r.t. Prokka.')
    parser.add_argument('-f_phylo_roary', '--f_phylo_roary', dest='f_phylo_roary', action='store_true',
                        help='phylo-tree based split bash generating, w.r.t. Roary')
    parser.add_argument('-f_pre_cluster', '--f_pre_cluster', dest='f_pre_cluster', action='store_true',
                        help='Kma cluster')
    parser.add_argument('-f_cluster', '--f_cluster', dest='f_cluster', action='store_true',
                        help='Kma cluster')  # c program
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    # parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
    #                     help='kma based cv folders.')

    parser.add_argument('-f_random', '--f_random', dest='f_random', action='store_true',
                        help='random cv folders.')

    parser.add_argument('-f_cluster_folders', '--f_cluster_folders', dest='f_cluster_folders', action='store_true',
                        help='Compare new split method with old(original) method.')  # c program
    parser.add_argument('-f_res', '--f_res', dest='f_res', action='store_true',
                        help='Analyse Point/ResFinder results')
    parser.add_argument('-f_merge_mution_gene', '--f_merge_mution_gene', dest='f_merge_mution_gene',
                        action='store_true',
                        help=' Merging ResFinder and PointFinder results')
    parser.add_argument('-f_matching_io', '--f_matching_io', dest='f_matching_io', action='store_true',
                        help='Matching input and output results')
    # para for nn nestedCV
    parser.add_argument('-f_nn', '--f_nn', dest='f_nn', action='store_true',
                        help='Run the NN model')
    parser.add_argument("-cv", "--cv_number", default=10, type=int,
                        help='CV splits number')
    parser.add_argument( "--seed", default=42, type=int,
                        help='random state related to shuffle cluster order')
    parser.add_argument("-d", "--hidden", default=200, type=int,
                        help='dimension of hidden layer')
    parser.add_argument("-e", "--epochs", default=0, type=int,
                        help='epochs')
    parser.add_argument("-re_e", "--re_epochs", default=500, type=int,
                        help='epochs')
    parser.add_argument("-learning", "--learning", default=0.0, type=float,
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
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
            \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
            \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    # parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    # parser.add_argument('-debug', '--debug', dest='debug', action='store_true',help='debug')
    parsedArgs = parser.parse_args()
    # parser.print_help()
    # print(parsedArgs)
    extract_info(parsedArgs.path_sequence, parsedArgs.species, parsedArgs.level, parsedArgs.f_all,parsedArgs.f_phylotree,parsedArgs.f_random,
                 parsedArgs.f_pre_meta,parsedArgs.f_phylo_prokka,
                 parsedArgs.f_phylo_roary, parsedArgs.f_pre_cluster, parsedArgs.f_cluster, parsedArgs.f_cluster_folders,
                 parsedArgs.f_res,
                 parsedArgs.f_merge_mution_gene, parsedArgs.f_matching_io,
                 parsedArgs.f_nn, parsedArgs.cv_number, parsedArgs.seed, parsedArgs.hidden, parsedArgs.epochs,
                 parsedArgs.re_epochs,
                 parsedArgs.learning, parsedArgs.f_scaler, parsedArgs.f_fixed_threshold, parsedArgs.f_nn_base,parsedArgs.f_optimize_score,parsedArgs.f_qsub)
