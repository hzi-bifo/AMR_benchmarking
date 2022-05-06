import amr_utility.name_utility
import amr_utility.file_utility
import argparse,pickle
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
import neural_networks.cluster_folders as pre_cluster_folders
import cv_folders.create_phylotree

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
    #Merge meta and pheno to make sure the use the same id list(order).
    df_feature_s_f = pd.merge(df_feature_s, meta_s, how="outer", on=["id"])

    df_phenotype_final=df_feature_s_f.loc[:, antibiotics]
    df_phenotype_final=df_phenotype_final.fillna(-1)
    print(df_phenotype_final)
    df_phenotype_final.to_csv(path_y,index=False,header=False, sep="\t")




def prepare_folds(antibiotics,cv,level, path_large_temp, species,random):
    # 1. kma
    for anti in antibiotics:
        multi_log,_,_,path_folds_kma_raw,path_folds_kma,path_folds_tree_raw,path_folds_tree,path_folds_random_raw,path_folds_random,_,_,_\
        = amr_utility.name_utility.GETname_multiAnti(level, path_large_temp, species,anti) #3rdMay2022.name updated.
        path_feature, path_res_result, path_metadata, path_large_temp_kma, path_large_temp_prokka, path_large_temp_roary, \
                    path_metadata_prokka, path_cluster_temp, path_metadata_pheno, path_roary_results, path_cluster_results, path_point_repre_results, \
                    path_res_repre_results, path_mutation_gene_results, path_x_y, path_x, path_y, path_name = \
                        amr_utility.name_utility.GETname_multi_bench_main_feature(level, species, anti,path_large_temp)
        folders_sample_new,split_new_k,folder_sampleName_new = pre_cluster_folders.prepare_folders(cv, random, path_metadata, path_cluster_results,'new')
        # folds_txt_raw=multi_log+str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_kma_raw.pickle"
        with open(path_folds_kma_raw, 'wb') as f:  # overwrite
            pickle.dump(folder_sampleName_new, f)
    folds_kma_multiAnti=[]
    for iter in range(cv):
        folds_kma_multiAnti_sub=[]
        for anti in antibiotics:#merge folds of all anti of the same species
            ulti_log,_,_,path_folds_kma_raw,path_folds_kma,path_folds_tree_raw,path_folds_tree,path_folds_random_raw,path_folds_random,_,_,_\
                = amr_utility.name_utility.GETname_multiAnti(level, path_large_temp, species,anti) #3rdMay2022.name updated.
            # folds_txt_raw=multi_log+str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_kma_raw.pickle"
            folds_each=pickle.load(open(path_folds_kma_raw, "rb"))
            folds_kma_multiAnti_sub=folds_kma_multiAnti_sub+folds_each[iter]
            folds_kma_multiAnti_sub = list( dict.fromkeys(folds_kma_multiAnti_sub) )#drop duplicates

        folds_kma_multiAnti_sub=[iso_name.replace("iso_", "").replace("\n", "") for iso_name in folds_kma_multiAnti_sub]
        folds_kma_multiAnti.append(folds_kma_multiAnti_sub)
    # folds_txt=multi_log+str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_kma.pickle"
    with open(path_folds_kma, 'wb') as f:  # overwrite
        pickle.dump(folds_kma_multiAnti, f)

    #2.phylo-tree
    if species != "Mycobacterium tuberculosis":
        for anti in antibiotics:

            ulti_log,_,_,path_folds_kma_raw,path_folds_kma,path_folds_tree_raw,path_folds_tree,path_folds_random_raw,path_folds_random,_,_,_\
                = amr_utility.name_utility.GETname_multiAnti(level, path_large_temp, species,anti) #3rdMay2022.name updated.
            Tree='cv_folders/'+'loose'+'/'+str(species.replace(" ", "_"))+'/cv_tree_'+ str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'.txt'
            mapping_file='cv_folders/'+'loose'+'/'+str(species.replace(" ", "_"))+'/mapping_2.npy'
            tree_names=[]
            with open(Tree) as f:
                lines = f.readlines()
                for i in lines:
                    # print(i.split('\t'))
                    tree_names_sub=[]
                    for each in i.split('\t'):
                        each=each.replace("\n", "")
                        # decode the md5 name to iso names
                        mapping_dic = np.load(mapping_file, allow_pickle='TRUE').item()
                        decoder_name = mapping_dic[each]
                        iso_name=decoder_name[0]
                        #------------------------------------
                        tree_names_sub.append(iso_name.replace("iso_", "").replace("\n", ""))
                        # print(tree_names_sub)
                    tree_names.append(tree_names_sub)
            # print(tree_names)
            with open(path_folds_tree_raw, 'wb') as f:  # overwrite
                pickle.dump(tree_names, f)
        folds_tree_multiAnti=[]
        for iter in range(cv):
            folds_tree_multiAnti_sub=[]
            for anti in antibiotics:#merge folds of all anti of the same species
                ulti_log,_,_,path_folds_kma_raw,path_folds_kma,path_folds_tree_raw,path_folds_tree,path_folds_random_raw,path_folds_random,_,_,_\
                    = amr_utility.name_utility.GETname_multiAnti(level, path_large_temp, species,anti) #3rdMay2022.name updated.
                # folds_txt_raw=multi_log+str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_kma_raw.pickle"
                folds_each=pickle.load(open(path_folds_tree_raw, "rb"))
                folds_tree_multiAnti_sub=folds_tree_multiAnti_sub+folds_each[iter]
                folds_tree_multiAnti_sub = list( dict.fromkeys(folds_tree_multiAnti_sub) )#drop duplicates
            folds_tree_multiAnti.append(folds_tree_multiAnti_sub)
        # folds_txt=multi_log+str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_kma.pickle"
        with open(path_folds_tree, 'wb') as f:  # overwrite
            pickle.dump(folds_tree_multiAnti, f)
    #3. random_folds
    for anti in antibiotics:
        ulti_log,_,_,path_folds_kma_raw,path_folds_kma,path_folds_tree_raw,path_folds_tree,path_folds_random_raw,path_folds_random,_,_,_\
                = amr_utility.name_utility.GETname_multiAnti(level, path_large_temp, species,anti) #3rdMay2022.name updated.
        if species != "Mycobacterium tuberculosis":
            Random_path='cv_folders/'+'loose'+'/'+str(species.replace(" ", "_"))+'/cv_random_'+ str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'.txt'
            mapping_file='cv_folders/'+'loose'+'/'+str(species.replace(" ", "_"))+'/mapping_2.npy'
            Random_names=[]
            with open(Random_path) as f:
                lines = f.readlines()
                for i in lines:
                    # print(i.split('\t'))
                    Random_names_sub=[]
                    for each in i.split('\t'):
                        each=each.replace("\n", "")
                        # decode the md5 name to iso names
                        mapping_dic = np.load(mapping_file, allow_pickle='TRUE').item()
                        decoder_name = mapping_dic[each]
                        iso_name=decoder_name[0]
                        #------------------------------------
                        Random_names_sub.append(iso_name.replace("iso_", "").replace("\n", ""))
                        # print(tree_names_sub)
                    Random_names.append(Random_names_sub)


        else:#only for the case of "Mycobacterium tuberculosis"
            main_path='cv_folders/'+'loose'+'/'+str(species.replace(" ", "_"))+"/"+ \
                  str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_random_cv.pickle"
            Random_names = pickle.load(open(main_path, "rb"))
        # print(Random_names)
        with open(path_folds_random_raw, 'wb') as f:  # overwrite
            pickle.dump(Random_names, f)

    folds_random_multiAnti=[]
    for iter in range(cv):

        folds_random_multiAnti_sub=[]
        for anti in antibiotics:#merge folds of all anti of the same species
            ulti_log,_,_,path_folds_kma_raw,path_folds_kma,path_folds_tree_raw,path_folds_tree,path_folds_random_raw,path_folds_random,_,_,_\
                = amr_utility.name_utility.GETname_multiAnti(level, path_large_temp, species,anti) #3rdMay2022.name updated.
            # folds_txt_raw=multi_log+str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_kma_raw.pickle"
            folds_each=pickle.load(open(path_folds_random_raw, "rb"))

            folds_eachcv=[str(element) for element in folds_each[iter]]
            folds_random_multiAnti_sub=folds_random_multiAnti_sub+folds_eachcv
            folds_random_multiAnti_sub = list( dict.fromkeys(folds_random_multiAnti_sub) )#drop duplicates
        folds_random_multiAnti.append(folds_random_multiAnti_sub)
    # folds_txt=multi_log+str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_kma.pickle"
    with open(path_folds_random, 'wb') as f:  # overwrite
        pickle.dump(folds_random_multiAnti, f)




def run(species,path_sequence,path_large_temp,list_species,level,f_all,f_pre_meta,f_phylo_prokka,f_phylo_roary,
        f_pre_cluster,f_cluster,f_cluster_folders,f_res,f_merge_mution_gene,f_matching_io,f_nn,cv, random,
                 hidden, epochs, re_epochs, learning,f_scaler,f_fixed_threshold,f_nn_base,f_optimize_score,f_phylotree,f_random):
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
        prepare_folds(antibiotics,cv,level, path_large_temp, species,random)

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



        score=nn_module_hyper.multiAnti(species,antibiotics, level, path_x, path_y, path_name, path_mutation_gene_results_multi, cv, random,
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






def extract_info(path_sequence,list_species,level,f_all,f_phylotree,f_random,f_pre_meta,f_phylo_prokka,f_phylo_roary,
                 f_pre_cluster,f_cluster,f_cluster_folders,f_res,f_merge_mution_gene,f_matching_io,f_nn,cv, random,
                 hidden, epochs, re_epochs, learning,f_scaler,f_fixed_threshold,f_nn_base,f_optimize_score):
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
    for species in df_species:
        multi_log = './log/temp/' + str(level) + '/multi_anti/' + str(species.replace(" ", "_"))
        amr_utility.file_utility.make_dir(multi_log)
        run(species,path_sequence,path_large_temp,list_species,level,f_all,f_pre_meta,f_phylo_prokka,f_phylo_roary,f_pre_cluster,f_cluster,f_cluster_folders,f_res,f_merge_mution_gene,f_matching_io,f_nn,cv, random,
                 hidden, epochs, re_epochs, learning,f_scaler,f_fixed_threshold,f_nn_base,f_optimize_score,f_phylotree,f_random,)



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
    parser.add_argument("-r", "--random", default=42, type=int,
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
                 parsedArgs.f_nn, parsedArgs.cv_number, parsedArgs.random, parsedArgs.hidden, parsedArgs.epochs,
                 parsedArgs.re_epochs,
                 parsedArgs.learning, parsedArgs.f_scaler, parsedArgs.f_fixed_threshold, parsedArgs.f_nn_base,parsedArgs.f_optimize_score)
