import amr_utility.name_utility
import amr_utility.graph_utility
import amr_utility.file_utility
import amr_utility.load_data
import cv_folders.cluster_folders
import argparse,itertools,os
from pathlib import Path
import numpy as np
import pickle,json
import pandas as pd

def extract_info(level, f_phylotree, f_kma ):
    cv=10
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    data = pd.read_csv('metadata/' + str(level) + '_Species_antibiotic_FineQuality.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    # for training model on part of the dataset:-------------
    # data=data.loc[['Escherichia coli'],:]
    # if f_all == False:
    #     data = data.loc[s, :]
    if -f_phylotree:
        s=['Escherichia coli','Staphylococcus aureus','Salmonella enterica','Klebsiella pneumoniae',\
           'Pseudomonas aeruginosa','Acinetobacter baumannii','Streptococcus pneumoniae', 'Campylobacter jejuni','Enterococcus faecium','Neisseria gonorrhoeae']
        data = data.loc[s, :]
    # --------------------------------------------------------
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    # print(data)


    for species, antibiotics in zip(df_species, antibiotics):
        antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
        i_anti = 0

        for anti in antibiotics:

            # 1. exrtact CV folders----------------------------------------------------------------
            _, p_names = amr_utility.name_utility.GETsave_name_modelID(level, species, anti)
            Random_State = 42
            p_clusters = amr_utility.name_utility.GETname_folder(species, anti, level)
            if f_phylotree:  # phylo-tree based cv folders
                folders_index = cv_folders.cluster_folders.prepare_folders_tree(cv, species, anti, p_names,
                                                                                      False)
            elif f_kma:  # kma cluster based cv folders
                folders_index, _, _ = cv_folders.cluster_folders.prepare_folders(cv, Random_State, p_names,
                                                                                       p_clusters,
                                                                                       'new')
            else:#random
                folders_index = cv_folders.cluster_folders.prepare_folders_random(cv, species, anti, p_names,
                                                                                      False)

            id=ID[i_anti]
            i_anti+=1
            idname=[]
            for each_folds in folders_index:
                id_sub=[]
                for each_s in each_folds:
                    id_sub.append(id[each_s])
                idname.append(id_sub)
            amr_utility.file_utility.make_dir("./cv_folders/supplement/"+str(species.replace(" ", "_")))
            if f_phylotree:
                folds_txt = "./cv_folders/supplement/"+str(species.replace(" ", "_"))+"/"+\
                                       str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_plyotree_cv.json"
            elif f_kma:
                folds_txt = "./cv_folders/supplement/"+str(species.replace(" ", "_"))+"/"+\
                                       str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_KMA_cv.json"
            else:
                folds_txt = "./cv_folders/supplement/"+str(species.replace(" ", "_"))+"/"+\
                                       str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_random_cv.json"


            # with open(folds_txt, 'wb') as f:  # overwrite
            #     pickle.dump(idname, f)
            with open(folds_txt, 'w') as f:
                json.dump(idname, f)



# def extract_multi(level):
#     cv=11
#     learning=0.0
#     epochs=0
#     merge_name = []
#     f_fixed_threshold=True
#     f_nn_base=False
#     f_optimize_score='f1_macro'
#     random=42
#     data = pd.read_csv('metadata/' + str(level) + '_multi-species_summary.csv', index_col=0,
#                        dtype={'genome_id': object}, sep="\t")
#     list_species = data.index.tolist()[:-1]
#     data = data.loc[list_species, :]
#     # drop columns(antibotics) all zero
#     data = data.loc[:, (data != 0).any(axis=0)]
#     All_antibiotics = data.columns.tolist()  # all envolved antibiotics # todo
#     for n in list_species:
#         merge_name.append(n[0] + n.split(' ')[1][0])
#
#
#     name_weights_folder = amr_utility.name_utility.GETname_multi_bench_folder_multi(merge_name,level, learning, epochs,
#                                                                                    f_fixed_threshold,f_nn_base,f_optimize_score)
#
#
#
#     path_cluster_results=[]
#     for s in list_species:
#         path_large_temp_kma_multi, path_cluster_temp_multi, path_cluster_results_multi, path_large_temp_roary_multi, \
#         path_roary_results_multi, path_metadata_s_multi, path_metadata_pheno_s_multi, path_large_temp_prokka, path_res_result, path_point_repre_results_multi, \
#         path_res_repre_results_multi, path_mutation_gene_results_multi, path_feature_multi = \
#             amr_utility.name_utility.GETname_multi_bench_multi_species(level, path_large_temp, merge_name, s)
#         path_cluster_results.append(path_cluster_results_multi)
#     folders_sample_new,split_new_k = cv_folders.cluster_folders.prepare_folders(cv, random, path_ID_multi, path_cluster_results,'new')
