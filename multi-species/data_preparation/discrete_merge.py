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
import amr_utility.file_utility
import argparse
import amr_utility.load_data
import pandas as pd




def merge_feature(merge_name,path_large_temp,list_species,All_antibiotics,level):
    '''
    :return: merged feature matrix , data_x, data_y, data_name
    '''
    count=0
    id_feature_all = []  # feature dataframe of each species
    id_pheno_all = []
    feature_dimension_all=pd.DataFrame( index=list_species,columns=['feature dimension'])
    if len(list_species)<2:
        print('pleas feed in at lest 2 species.')
        exit()
    _, path_ID_multi, _, _, _, _, _, path_x, path_y, path_name \
        = amr_utility.name_utility.GETname_multi_bench_multi(level, path_large_temp, merge_name)
    for s in list_species:
        path_large_temp_kma_multi, path_cluster_temp_multi, path_cluster_results_multi, path_large_temp_roary_multi, \
        path_roary_results_multi, path_metadata_s_multi, path_metadata_pheno_s_multi, path_large_temp_prokka, path_res_result, path_point_repre_results_multi, \
        path_res_repre_results_multi, path_mutation_gene_results_multi, path_feature_multi=\
            amr_utility.name_utility.GETname_multi_bench_multi_species(level, path_large_temp, merge_name, s)#todo add to previous place using this function

        #meta
        # path_metadata_s_multi
        meta_s = pd.read_csv(path_metadata_pheno_s_multi, sep="\t", header=0, index_col=0, dtype={'id': object, 'pheno': int})

        #feature matrix:
        feature_s=np.genfromtxt(path_mutation_gene_results_multi, dtype="str")
        n_feature_s=feature_s.shape[1]-1#number of features for this species
        feature_dimension_all.loc[s,'feature dimension']=n_feature_s
        df_feature_s=pd.DataFrame(feature_s, index=None, columns=np.insert(np.array(np.arange(n_feature_s)+count,dtype='object'), 0, 'id'))#,dtype={'id': object}
        # print(df_feature_s.dtypes)#objecte
        #combine feature and pheno matrix


        # print(df_feature_s)
        id_pheno_all.append(meta_s)
        id_feature_all.append(df_feature_s)
        # print(df_feature_s)
        count += n_feature_s

    feature_dimension_all.to_csv(path_feature_multi+'feature_Dimension.txt', sep="\t")
    df_feature_s_f=id_feature_all[0]
    for i in id_feature_all[1:]:
        df_feature_s_f= pd.concat([df_feature_s_f, i], ignore_index=True, sort=False)

    df_pheno_s_f = id_pheno_all[0]
    for i in id_pheno_all[1:]:
        df_pheno_s_f = pd.concat([df_pheno_s_f, i], ignore_index=True, sort=False)

    #Merge meta and pheno to make sure the use the same id list(order).
    df_feature_s_f = pd.merge(df_feature_s_f, df_pheno_s_f, how="outer", on=["id"])

    df_feature_s_f = df_feature_s_f.set_index('id')
    print('======================')
    print(df_feature_s_f)
    print(feature_dimension_all)


    #Note!!! force the data_x 's order in according with id_list. Also done in merge_input_output_files_khuModified.py file..
    id_list = np.genfromtxt(path_ID_multi, dtype="str")
    df_feature_s_f = df_feature_s_f.reindex(id_list)
    # df_feature_s_f = df_feature_s_f.reset_index()


    # print(df_feature_s_f.columns)
    # exit()
    # Pad nan with 0 in feature matrix, with -1 in phen matrix
    df_feature_final=df_feature_s_f.loc[:,np.array(np.arange(sum(feature_dimension_all['feature dimension'].to_list())),dtype='object')]#exclude pheno part
    df_feature_final=df_feature_final.fillna(0)
    df_feature_final.to_csv(path_x,index=False,header=False, sep="\t")

    df_phenotype_final=df_feature_s_f.loc[:, All_antibiotics]
    df_phenotype_final=df_phenotype_final.fillna(-1)
    print(df_phenotype_final)
    df_phenotype_final.to_csv(path_y,index=False,header=False, sep="\t")
    df_feature_s_f.index.to_series().to_csv(path_name,header=False, index=False,sep="\t")

    print('Feature part of discerte multi-s model finished. Can procede to NN model now.')
    # done!
    # Note, discrete version of multi-s model,no need to use merge_input_output_files_khuModified.py file.







def prepare_meta(path_large_temp,list_species,selected_anti,level,f_all):
    '''
    :param path_large_temp: path for storage large intermediate files
    :param list_species: species in multi-s model
    :param selected_anti: currently use all possibe antis w.r.t. selected species
    :param level: QC
    :param f_all:Flag for selecting all possible species in our dataset
    :return: each species' metadata of selected antibitocs. combined metadata of all selected speceis(all antibiotics).
    '''

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
    # multi_log = './log/temp/' + str(level) + '/multi_species/'+merge_name+'/'
    # # amr_utility.file_utility.make_dir(multi_log)
    #
    #
    # # todo: if anti can be selected, then this name should be reconsidered.
    # #path_metadata_multi = multi_log + '/'+ save_name_anti+'meta.txt'# both ID and pheno.#so far, no this option.
    # path_metadata_multi = multi_log + '/meta.txt'# both ID and pheno.
    multi_log, path_ID_multi,path_metadata_multi, _, _, _, _, _, _, _ \
        = amr_utility.name_utility.GETname_multi_bench_multi(level, path_large_temp, merge_name)

    if selected_anti==[]:
        cols = data.columns
        bt = data.apply(lambda x: x > 0)#all possible antibiotics
        data_species_anti = bt.apply(lambda x: list(cols[x.values]), axis=1)
        print(data_species_anti)# dataframe of each species and coresponding selected antibiotics.
    else:
        print('Not possible to choose anti by user yet.')
        exit()
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
                metadata_pheno = pd.merge(metadata_pheno, i, how="outer", on=["id"])# merge antibiotics within one species
                # print(metadata_pheno)
                # print('************')
        else:
            pass#no need for merge.

        metadata_pheno.to_csv(multi_log+str(species.replace(" ", "_"))+'_meta.txt', sep="\t", index=True, header=True)
        metadata_pheno['id'].to_csv(multi_log+str(species.replace(" ", "_"))+'_id', sep="\t", index=False, header=False)
        metadata_pheno_all.append(metadata_pheno)
        # print(metadata_pheno)

    metadata_pheno_f=metadata_pheno_all[0]
    for i in metadata_pheno_all[1:]:
        metadata_pheno_f =  metadata_pheno_f.append(i) # append all the species

    # print(metadata_pheno_f)

    metadata_pheno_f.to_csv(path_metadata_multi, sep="\t",index=True, header=True)
    metadata_pheno_f['id'].to_csv(path_ID_multi, sep="\t", index=False,header=False)

