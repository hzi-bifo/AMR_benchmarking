import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import numpy as np
from pathlib import Path
import ast
import time
import amr_utility.name_utility
import amr_utility.graph_utility
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
# import data_preparation.merge_species_khuModified
import neural_networks.Neural_networks_khuModified_hyperpara as nn_module_hyper
import neural_networks.Neural_networks_khuModified_earlys as nn_module
# import data_preparation.discrete_merge
# import neural_networks.Neural_networks_khuModified as nn_module_original
import csv
import neural_networks.cluster_folders

##Notes to author self:
# to change if anti choosable: means the codes need to refine if anti is choosable.
def merge_feature(merge_name,path_large_temp,list_species,All_antibiotics,level):
    '''
    :return: merged feature matrix , data_x, data_y, data_name
    '''
    count=0
    id_feature_all = []  # feature dataframe of each species
    id_pheno_all = []
    feature_dimension_all=pd.DataFrame( index=list_species,columns=['feature dimension'])
    print(All_antibiotics)
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
    # --------------------------------------------------------
    # drop columns(antibotics) all zero
    # data=data.loc[:, (data != 0).any(axis=0)]
    All_antibiotics = data.columns.tolist()#all envolved antibiotics # todo
    print(All_antibiotics)
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



def make_dir(name):
    logDir = os.path.join(name)
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
            print('Make directory: ',logDir)
        except OSError:
            print("Can't create logging directory:", logDir)

def run(merge_name,path_sequence,path_large_temp,list_species,All_antibiotics,level,f_all,f_pre_meta,f_phylo_prokka,f_phylo_roary,
        f_pre_cluster,f_cluster,f_cluster_folders,f_res,f_merge_mution_gene,f_matching_io,f_nn,cv, random,
                 hidden, epochs, re_epochs, learning,f_scaler,f_fixed_threshold,f_nn_base,f_optimize_score):
    if f_pre_meta==True:
        prepare_meta(path_large_temp,list_species,[],level,f_all)#[] means all possible will anti.

    #high level names
    multi_log,path_ID_multi,_,run_file_kma,run_file_roary1,run_file_roary2,run_file_roary3,path_x,path_y,path_name\
        =amr_utility.name_utility.GETname_multi_bench_multi(level,path_large_temp,merge_name)
    # =================================
    # 1. clusteromg or make phylo-tree
    # =================================
    # '/cv_folders/loose'
    if f_pre_cluster == True:

        for s in list_species:
            path_large_temp_kma_multi, path_cluster_temp_multi, path_cluster_results_multi, path_large_temp_roary_multi, \
            path_roary_results_multi, path_metadata_s_multi, path_metadata_pheno_s_multi, path_large_temp_prokka, path_res_result, path_point_repre_results_multi, \
            path_res_repre_results_multi, path_mutation_gene_results_multi, path_feature_multi = \
                amr_utility.name_utility.GETname_multi_bench_multi_species(level, path_large_temp, merge_name, s)
            data_preparation.merge_scaffolds_khuModified.extract_info(path_sequence, path_metadata_s_multi, path_large_temp_kma_multi, 16)

        print('finish merge_scaffold!')
    if f_cluster == True:

        if path_sequence == '/vol/projects/BIFO/patric_genome':
            for s in list_species:
                path_large_temp_kma_multi, path_cluster_temp_multi, path_cluster_results_multi, path_large_temp_roary_multi, \
                path_roary_results_multi, path_metadata_s_multi, path_metadata_pheno_s_multi, path_large_temp_prokka, path_res_result, path_point_repre_results_multi, \
                path_res_repre_results_multi, path_mutation_gene_results_multi, path_feature_multi= \
                    amr_utility.name_utility.GETname_multi_bench_multi_species(level, path_large_temp, merge_name, s)
                path_to_kma_clust = ("kma_clustering")
                # if Neisseria gonorrhoeae, then use  -ht 1.0 -hq 1.0, otherwise, all genomes will be clustered into one group #todo check....

                file_sub=os.path.join(os.path.dirname(run_file_kma), merge_name,str(s.replace(" ", "_"))+'_kma.sh')
                make_dir(os.path.dirname(file_sub))
                run_file = open(file_sub, "w")  # to change if anti choosable
                run_file.write("#!/bin/bash")
                run_file.write("\n")


                run_file = amr_utility.file_utility.hzi_cpu_header3(run_file,
                                                                    str(s.replace(" ", "_"))+'_kma.sh',
                                                                    2)
                # run_file.write("(")
                # if s == 'Neisseria gonorrhoeae' or s=='Mycobacterium tuberculosis':
                if s in ['Mycobacterium tuberculosis']:
                    cmd = "cat %s | %s -i -- -k 16 -Sparse - -ht 0.99 -hq 0.99 -NI -o %s &> %s" \
                          % (path_large_temp_kma_multi, path_to_kma_clust, path_cluster_temp_multi, path_cluster_results_multi)
                elif s in ['Klebsiella pneumoniae']:
                    cmd = "cat %s | %s -i -- -k 16 -Sparse - -ht 0.98 -hq 0.98 -NI -o %s &> %s" \
                          % (path_large_temp_kma_multi, path_to_kma_clust, path_cluster_temp_multi,
                             path_cluster_results_multi)
                else:
                    cmd = "cat %s | %s -i -- -k 16 -Sparse - -ht 0.9 -hq 0.9 -NI -o %s &> %s" \
                          % (path_large_temp_kma_multi, path_to_kma_clust, path_cluster_temp_multi, path_cluster_results_multi)
                run_file.write(cmd)
                run_file.write("\n")
                run_file.write("wait")
                run_file.write("\n")
                # run_file.write('rm '+ path_large_temp_kma_multi )
                run_file.write("\n")
                run_file.write(
                    'rm ' + path_cluster_temp_multi + '.seq.b ' + path_cluster_temp_multi + '.length.b ' + path_cluster_temp_multi + '.name ' + path_cluster_temp_multi + '.comp.b')
                run_file.write("\n")
                run_file.write("\n")
                # run_file.write("echo \" running... \"")
                run_file.close()
        else:
            make_dir(os.path.dirname(run_file_kma))
            run_file = open(run_file_kma, "w")  # to change if anti choosable
            run_file.write("#!/bin/bash")
            run_file.write("\n")
            for s in list_species:
                path_large_temp_kma_multi, path_cluster_temp_multi, path_cluster_results_multi, path_large_temp_roary_multi, \
                path_roary_results_multi, path_metadata_s_multi, path_metadata_pheno_s_multi, path_large_temp_prokka, path_res_result, path_point_repre_results_multi, \
                path_res_repre_results_multi, path_mutation_gene_results_multi, path_feature_multi = \
                    amr_utility.name_utility.GETname_multi_bench_multi_species(level, path_large_temp, merge_name, s)
                path_to_kma_clust = ("kma_clustering")
                # if Neisseria gonorrhoeae, then use  -ht 1.0 -hq 1.0, otherwise, all genomes will be clustered into one group #todo check....

                run_file.write("(")
                if s in ['Mycobacterium tuberculosis','Klebsiella pneumoniae']:
                    cmd = "cat %s | %s -i -- -k 16 -Sparse - -ht 0.98 -hq 0.98 -NI -o %s &> %s" \
                          % (path_large_temp_kma_multi, path_to_kma_clust, path_cluster_temp_multi,
                             path_cluster_results_multi)
                else:
                    cmd = "cat %s | %s -i -- -k 16 -Sparse - -ht 0.9 -hq 0.9 -NI -o %s &> %s" \
                          % (path_large_temp_kma_multi, path_to_kma_clust, path_cluster_temp_multi,
                             path_cluster_results_multi)
                run_file.write(cmd)
                run_file.write("\n")
                # run_file.write("wait")
                run_file.write("\n")
                # run_file.write('rm '+ path_large_temp_kma_multi )
                run_file.write("\n")
                run_file.write(
                    'rm ' + path_cluster_temp_multi + '.seq.b ' + path_cluster_temp_multi + '.length.b ' + path_cluster_temp_multi + '.name ' + path_cluster_temp_multi + '.comp.b')
                run_file.write("\n")
                # run_file.write("echo \" one thread finised! \"")

                run_file.write(")&")

                run_file.write("\n")
            run_file.write("echo \" running... \"")
            run_file.close()

    if f_phylo_prokka == True:
        print('No need to redo, if single-species was run.')
        pass


    if f_phylo_roary == True:

        make_dir(os.path.dirname(run_file_roary1))#the same parent folder
        run_file = open(run_file_roary1, "w")
        run_file.write("#!/bin/bash")
        run_file.write("\n")
        for s in list_species:
            path_large_temp_kma_multi, path_cluster_temp_multi, path_cluster_results_multi, path_large_temp_roary_multi, \
            path_roary_results_multi, path_metadata_s_multi, path_metadata_pheno_s_multi, path_large_temp_prokka, path_res_result, path_point_repre_results_multi, \
            path_res_repre_results_multi, path_mutation_gene_results_multi, path_feature_multi = \
                amr_utility.name_utility.GETname_multi_bench_multi_species(level, path_large_temp, merge_name, s)
            make_dir(path_large_temp_roary_multi)

            run_file.write("\n")
            cmd = 'cat %s|' % path_metadata_s_multi
            run_file.write(cmd)
            run_file.write("\n")
            run_file.write('while read i; do')
            run_file.write("\n")
            cmd = 'cp %s/${i}/*.gff %s/${i}.gff' % (path_large_temp_prokka, path_large_temp_roary_multi)
            run_file.write(cmd)
            run_file.write("\n")
            run_file.write('done')
            run_file.write("\n")
            run_file.write("wait")
            run_file.write("\n")

        run_file = open(run_file_roary2, "w")
        run_file.write("#!/bin/bash")
        run_file.write("\n")
        for s in list_species:
            path_large_temp_kma_multi, path_cluster_temp_multi, path_cluster_results_multi, path_large_temp_roary_multi, \
            path_roary_results_multi, path_metadata_s_multi, path_metadata_pheno_s_multi, path_large_temp_prokka, path_res_result, path_point_repre_results_multi, \
            path_res_repre_results_multi, path_mutation_gene_results_multi, path_feature_multi = \
                amr_utility.name_utility.GETname_multi_bench_multi_species(level, path_large_temp, merge_name, s)
            # make_dir(path_large_temp_roary_multi)
            run_file.write("\n")
            cmd = 'roary -p 20 -f %s -e --mafft -v %s/*.gff' % (path_roary_results_multi, path_large_temp_roary_multi)
            run_file.write(cmd)
            run_file.write("\n")
            run_file.write("wait")
            run_file.write("\n")
            run_file.write("echo \" one finished. \"")
            run_file.write("\n")

        run_file = open(run_file_roary3, "w")
        run_file.write("#!/bin/bash")
        run_file.write("\n")
        for s in list_species:
            path_large_temp_kma_multi, path_cluster_temp_multi, path_cluster_results_multi, path_large_temp_roary_multi, \
            path_roary_results_multi, path_metadata_s_multi, path_metadata_pheno_s_multi, path_large_temp_prokka, path_res_result, path_point_repre_results_multi, \
            path_res_repre_results_multi, path_mutation_gene_results_multi, path_feature_multi = \
                amr_utility.name_utility.GETname_multi_bench_multi_species(level, path_large_temp, merge_name, s)
            run_file.write("\n")
            cmd = 'fasttreeMP -nt -gtr %s/core_gene_alignment.aln > %s/my_tree.newick' % (
            path_roary_results_multi, path_roary_results_multi)
            run_file.write(cmd)
            run_file.write("\n")
            run_file.write("wait")
            run_file.write("\n")
            run_file.write("echo \" one finished. \"")
            run_file.write("\n")
            run_file.write("wait")
            run_file.write("\n")

    # =================================
    # 2. Analysing PointFinder results
    # Analysing ResFinder results
    # =================================
    if f_res == True:
        for s in list_species:
            print(s)

            path_large_temp_kma_multi, path_cluster_temp_multi, path_cluster_results_multi, path_large_temp_roary_multi, \
            path_roary_results_multi, path_metadata_s_multi, path_metadata_pheno_s_multi, path_large_temp_prokka, path_res_result, path_point_repre_results_multi, \
            path_res_repre_results_multi, path_mutation_gene_results_multi, path_feature_multi = \
                amr_utility.name_utility.GETname_multi_bench_multi_species(level, path_large_temp, merge_name, s)
            if s in ['Klebsiella pneumoniae','Escherichia coli','Staphylococcus aureus','Mycobacterium tuberculosis','Salmonella enterica',
                       'Neisseria gonorrhoeae','Enterococcus faecium','Campylobacter jejuni']:

                data_preparation.scored_representation_blast_khuModified.extract_info(path_res_result, path_metadata_s_multi,
                                                                                      path_point_repre_results_multi, True,True)#SNP,no zip
            data_preparation.ResFinder_analyser_blast_khuModified.extract_info(path_res_result, path_metadata_s_multi,
                                                                               path_res_repre_results_multi,True)  # GPA,no zip


    if f_merge_mution_gene == True:
        for s in list_species:
            path_large_temp_kma_multi, path_cluster_temp_multi, path_cluster_results_multi, path_large_temp_roary_multi, \
            path_roary_results_multi, path_metadata_s_multi, path_metadata_pheno_s_multi, path_large_temp_prokka, path_res_result, path_point_repre_results_multi, \
            path_res_repre_results_multi, path_mutation_gene_results_multi, path_feature_multi = \
                amr_utility.name_utility.GETname_multi_bench_multi_species(level, path_large_temp, merge_name, s)
            if s in ['Klebsiella pneumoniae', 'Escherichia coli', 'Staphylococcus aureus',
                           'Mycobacterium tuberculosis', 'Salmonella enterica',
                           'Neisseria gonorrhoeae', 'Enterococcus faecium', 'Campylobacter jejuni']:
                data_preparation.merge_resfinder_pointfinder_khuModified.extract_info(path_point_repre_results_multi,
                                                                                      path_res_repre_results_multi,
                                                                                      path_mutation_gene_results_multi)
            else:  # only AMR gene feature
                data_preparation.merge_resfinder_khuModified.extract_info(path_metadata_s_multi, path_res_repre_results_multi,
                                                                          path_mutation_gene_results_multi)

    if f_matching_io == True:
        print('Different from single-s model.')
        merge_feature(merge_name, path_large_temp, list_species, All_antibiotics, level)


    # =================================
    #3.  model
    # =================================
    if f_nn == True:
        name_weights_folder = amr_utility.name_utility.GETname_multi_bench_folder_multi(merge_name,level, learning, epochs,
                                                                                   f_fixed_threshold,f_nn_base,f_optimize_score)


        make_dir(name_weights_folder)#for storage of weights.
        path_cluster_results=[]
        for s in list_species:
            path_large_temp_kma_multi, path_cluster_temp_multi, path_cluster_results_multi, path_large_temp_roary_multi, \
            path_roary_results_multi, path_metadata_s_multi, path_metadata_pheno_s_multi, path_large_temp_prokka, path_res_result, path_point_repre_results_multi, \
            path_res_repre_results_multi, path_mutation_gene_results_multi, path_feature_multi = \
                amr_utility.name_utility.GETname_multi_bench_multi_species(level, path_large_temp, merge_name, s)
            path_cluster_results.append(path_cluster_results_multi)

        # in the eval fundtion, 2nd parameter is only used in names.
        # save_name_score = amr_utility.name_utility.GETname_multi_bench_save_name_score(merge_name,'all_possible_anti', level,
        #                                                                                learning, epochs,
        #                                                                                f_fixed_threshold,
        #                                                                                f_nn_base,
        #                                                                                f_optimize_score)
        # nn_module.eval(merge_name, 'all_possible_anti', level, path_x, path_y, path_name, path_cluster_results, cv, random, hidden,
        #                epochs,re_epochs, learning, f_scaler, f_fixed_threshold, f_nn_base,f_optimize_score,save_name_score,None,None,None) # the last 3 Nones mean not concat multi-s model.
        #todo save_name_score
        save_name_score = amr_utility.name_utility.GETname_multi_bench_save_name_score(merge_name, 'all_possible_anti',
                                                                                       level,
                                                                                      0.0,0,
                                                                                       f_fixed_threshold,
                                                                                       f_nn_base,
                                                                                       f_optimize_score)

        nn_module_hyper.eval(merge_name,'all_possible_anti', level, path_x, path_y, path_name, path_cluster_results, cv, random,
                             re_epochs, f_scaler, f_fixed_threshold, f_nn_base, f_optimize_score, save_name_score, None,
                             None, None)  # hyperparmeter selection in inner loop of nested CV

    if f_cluster_folders == True:
        #analysis the number of each species' samples in each folder.
        # split_original_all=[]
        # split_new_k_all=[]
        path_cluster_results=[]
        for s in list_species:
            path_large_temp_kma_multi, path_cluster_temp_multi, path_cluster_results_multi, path_large_temp_roary_multi, \
            path_roary_results_multi, path_metadata_s_multi, path_metadata_pheno_s_multi, path_large_temp_prokka, path_res_result, path_point_repre_results_multi, \
            path_res_repre_results_multi, path_mutation_gene_results_multi, path_feature_multi = \
                amr_utility.name_utility.GETname_multi_bench_multi_species(level, path_large_temp, merge_name, s)
            path_cluster_results.append(path_cluster_results_multi)
        folders_sample,split_original = neural_networks.cluster_folders.prepare_folders(cv, random,path_ID_multi, path_cluster_results,'original')
        folders_sample_new,split_new_k = neural_networks.cluster_folders.prepare_folders(cv, random, path_ID_multi, path_cluster_results,'new')
        ### split_original_all.append(split_original)
        ### split_new_k_all.append(split_new_k)

        print(split_original)
        split_original=np.array(split_original)#the number of samples of each species in each folder
        split_new_k=np.array(split_new_k)
        std_o = np.std(split_original, axis=0, ddof=1)
        std_n = np.std(split_new_k, axis=0, ddof=1)

        ### split_original_all.append(std_o)
        ### split_new_k_all.append(std_n)

        print(split_original)
        #todo check: make a bar plot for the number of samples of each species, the x-axis is the folder 1-10
        amr_utility.file_utility.plot_kma_split(split_original, split_new_k, level, list_species, merge_name)


def extract_info(path_sequence,list_species,selected_anti,level,f_all,f_pre_meta,f_phylo_prokka,f_phylo_roary,
                 f_pre_cluster,f_cluster,f_cluster_folders,f_res,f_merge_mution_gene,f_matching_io,f_nn,cv, random,
                 hidden, epochs, re_epochs, learning,f_scaler,f_fixed_threshold,f_nn_base,f_optimize_score):
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
    # --------------------------------------------------------
    # drop columns(antibotics) all zero
    data = data.loc[:, (data != 0).any(axis=0)]
    All_antibiotics = data.columns.tolist()  # all envolved antibiotics # todo
    for n in list_species:
        merge_name.append(n[0] + n.split(' ')[1][0])
    merge_name = '_'.join(merge_name)  # e.g.Se_Kp_Pa
    multi_log = './log/temp/' + str(level) + '/multi_species/' + merge_name

    make_dir(multi_log)

    if path_sequence=='/net/projects/BIFO/patric_genome':
        path_large_temp='/net/sgi/metagenomics/data/khu/benchmarking/phylo'
    else:
        fileDir = os.path.dirname(os.path.realpath('__file__'))
        path_large_temp = os.path.join(fileDir, 'large_temp')

    run(merge_name,path_sequence,path_large_temp,list_species,All_antibiotics,level,f_all,f_pre_meta,f_phylo_prokka,f_phylo_roary,f_pre_cluster,f_cluster,f_cluster_folders,f_res,f_merge_mution_gene,f_matching_io,f_nn,cv, random,
             hidden, epochs, re_epochs, learning,f_scaler,f_fixed_threshold,f_nn_base,f_optimize_score)







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
    # parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    # parser.add_argument('-debug', '--debug', dest='debug', action='store_true',help='debug')
    parsedArgs = parser.parse_args()
    # parser.print_help()
    # print(parsedArgs)
    extract_info(parsedArgs.path_sequence, parsedArgs.species,parsedArgs.anti, parsedArgs.level, parsedArgs.f_all,
                 parsedArgs.f_pre_meta,parsedArgs.f_phylo_prokka,
                 parsedArgs.f_phylo_roary, parsedArgs.f_pre_cluster, parsedArgs.f_cluster, parsedArgs.f_cluster_folders,
                 parsedArgs.f_res,
                 parsedArgs.f_merge_mution_gene, parsedArgs.f_matching_io,
                 parsedArgs.f_nn, parsedArgs.cv_number, parsedArgs.random, parsedArgs.hidden, parsedArgs.epochs,
                 parsedArgs.re_epochs,
                 parsedArgs.learning, parsedArgs.f_scaler, parsedArgs.f_fixed_threshold, parsedArgs.f_nn_base,parsedArgs.f_optimize_score)
