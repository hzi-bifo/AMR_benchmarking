import amr_utility.name_utility
import amr_utility.graph_utility
import amr_utility.file_utility
import amr_utility.load_data
import cv_folders.cluster_folders
import argparse,itertools,os
from pathlib import Path
import numpy as np
import pickle
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
                                       str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_plyotree_cv.pickle"
            elif f_kma:
                folds_txt = "./cv_folders/supplement/"+str(species.replace(" ", "_"))+"/"+\
                                       str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_KMA_cv.pickle"
            else:
                folds_txt = "./cv_folders/supplement/"+str(species.replace(" ", "_"))+"/"+\
                                       str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_random_cv.pickle"
            with open(folds_txt, 'wb') as f:  # overwrite
                pickle.dump(idname, f)

