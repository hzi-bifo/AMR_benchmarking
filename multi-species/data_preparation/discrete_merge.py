import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import sys
sys.path.append('../')
sys.path.insert(0, os.getcwd())
import numpy as np
import amr_utility.name_utility
import amr_utility.graph_utility
import argparse
import amr_utility.load_data
import pandas as pd


def extract_info(path_sequence,path_large_temp,selected_anti,list_species,level,f_all):
    # data storage: one combination one file!
    #e.g. : ./temp/loose/Sa_Kp_Pa/meta/ & ./temp/loose/Sa_Kp_Pa/

    merge_name=[]

    data = pd.read_csv('metadata/' + str(level) + '_multi-species_summary.csv', index_col=0,
                           dtype={'genome_id': object}, sep="\t")
    if f_all:
        list_species=data.index.tolist()[:-1]

    data = data.loc[list_species, :]

    # --------------------------------------------------------
    # drop columns(antibotics) all zero
    data=data.loc[:, (data != 0).any(axis=0)]
    All_antibiotics = data.columns.tolist()#all envolved antibiotics # todo
    for n in list_species:
        merge_name.append(n[0] + n.split(' ')[1][0])
    merge_name = '_'.join(merge_name)#e.g.Se_Kp_Pa
    multi_log = './log/temp/' + str(level) + '/multi_species/'+merge_name+'/'
    logDir = os.path.join(multi_log)
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
            print("Can't create logging directory:", logDir)


    # todo: if anti can be selected, then this name should be reconsidered.
    #path_metadata_multi = multi_log + '/'+ save_name_anti+'meta.txt'# both ID and pheno.#so far, no this option.
    path_metadata_multi = multi_log + '/meta.txt'# both ID and pheno.

    cols = data.columns
    bt = data.apply(lambda x: x > 0)
    data_species_anti = bt.apply(lambda x: list(cols[x.values]), axis=1)
    print(data_species_anti)# dataframe of each species and coresponding selected antibiotics.

    # 1.
    ID_all=[] #D: n_species* (sample number for each species)
    metadata_pheno_all=[]
    for species in list_species:

        metadata_pheno_all_sub=[]
        for anti in data_species_anti[species]:
            path_feature, path_res_result, path_metadata, path_large_temp_kma, path_large_temp_prokka, path_large_temp_roary, \
            path_metadata_prokka, path_cluster_temp, path_metadata_pheno, path_roary_results, path_cluster_results, path_point_repre_results, \
            path_res_repre_results, path_mutation_gene_results, path_x_y, path_x, path_y, path_name = \
                amr_utility.name_utility.GETname_multi_bench_main_feature(level, species, anti, path_large_temp)

            metadata_pheno = pd.read_csv(path_metadata_pheno,  sep="\t",header=None,names=['id',anti],dtype={'id': object,'pheno':int})
            #
            # print(metadata_pheno)
            # print('----------------------')
            metadata_pheno_all_sub.append(metadata_pheno)
        if len(metadata_pheno_all_sub)>1:
            metadata_pheno=metadata_pheno_all_sub[0]
            for i in metadata_pheno_all_sub[1:]:
                metadata_pheno = pd.merge(metadata_pheno, i, how="outer", on=["id"])
                # print(metadata_pheno)
                # print('************')
        else:
            pass

        metadata_pheno.to_csv(multi_log+str(species.replace(" ", "_"))+'_meta', sep="\t", index=False, header=True)
        metadata_pheno['id'].to_csv(multi_log+str(species.replace(" ", "_"))+'_id', sep="\t", index=False, header=False)
        metadata_pheno_all.append(metadata_pheno)
        # print(metadata_pheno)

    metadata_pheno_f=metadata_pheno_all[0]
    for i in metadata_pheno_all[1:]:
        metadata_pheno_f =  metadata_pheno_f.append(i)

    # print(metadata_pheno_f)

    metadata_pheno.to_csv(multi_log + 'pheno', sep="\t",index=False, header=True)
    metadata_pheno['id'].to_csv(multi_log + 'ID', sep="\t", index=False,header=False)

