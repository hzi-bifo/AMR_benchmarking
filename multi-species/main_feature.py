
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import numpy as np
from pathlib import Path
import shutil
import ast
import statistics
import operator
import time
import amr_utility.name_utility
import amr_utility.graph_utility
import amr_utility.file_utility
import argparse
import amr_utility.load_data
import pandas as pd
from itertools import repeat
import multiprocessing as mp
import subprocess
import data_preparation.merge_scaffolds_khuModified
import data_preparation.scored_representation_blast_khuModified
import data_preparation.ResFinder_analyser_blast_khuModified
import data_preparation.merge_resfinder_pointfinder_khuModified
import data_preparation.merge_input_output_files_khuModified
import data_preparation.merge_resfinder_khuModified
# import neural_networks.Neural_networks_khuModified_earlys as nn_module
import neural_networks.Neural_networks_khuModified_hyperpara as nn_module_hyper
# import data_preparation.discrete_merge
# import neural_networks.Neural_networks_khuModified as nn_module_original #no use now.
import csv
import neural_networks.cluster_folders as pre_cluster_folders
import cv_folders.create_phylotree
import pickle


def run(path_sequence,path_large_temp,species,anti,level,f_phylo_prokka,f_phylo_roary,f_pre_cluster,f_cluster,f_cluster_folders,
        run_file,f_res,f_merge_mution_gene,f_matching_io,f_merge_species,f_nn,cv,
        random, hidden, epochs, re_epochs, learning,f_scaler,f_fixed_threshold,f_nn_base,f_phylotree,f_random,f_optimize_score):

    print(species, anti, '-------------------------------------------------')
    # path_sequence = '/net/projects/BIFO/patric_genome'# in the future, change it to custom setting variable.

    path_feature, path_res_result, path_metadata, path_large_temp_kma, path_large_temp_prokka, path_large_temp_roary, \
    path_metadata_prokka, path_cluster_temp, path_metadata_pheno, path_roary_results, path_cluster_results, path_point_repre_results, \
    path_res_repre_results, path_mutation_gene_results, path_x_y, path_x, path_y, path_name = \
        amr_utility.name_utility.GETname_multi_bench_main_feature(level, species, anti,path_large_temp)


    if f_phylo_prokka==True:
        # run_file.write("(")
        run_file.write("\n")
        cmd='cat %s|' % path_metadata_prokka
        run_file.write(cmd)
        run_file.write("\n")
        run_file.write('while read i; do')
        run_file.write("\n")
        cmd='prokka --quiet --cpus 20 --genus %s --outdir %s/${i}  %s/${i}.fna' %(species.split(' ')[0],path_large_temp_prokka,path_sequence)
        run_file.write(cmd)
        run_file.write("\n")
        run_file.write('done')
        run_file.write("\n")
        run_file.write("wait \n")
        run_file.write("echo \" one finished.\" \n")
        run_file.write("wait")
        run_file.write("\n")

    if f_phylo_roary == True:
        make_dir(path_large_temp_roary)


        run_file.write("\n")
        cmd = 'cat %s|' % path_metadata
        run_file.write(cmd)
        run_file.write("\n")
        run_file.write('while read i; do')
        run_file.write("\n")
        cmd='cp %s/${i}/*.gff %s/${i}.gff' % (path_large_temp_prokka,path_large_temp_roary)
        run_file.write(cmd)
        run_file.write("\n")
        run_file.write('done')
        run_file.write("\n")
        run_file.write("wait")
        run_file.write("\n")

    if f_phylo_roary=='step2':#seond roary bash file
        run_file.write("\n")
        cmd='roary -p 20 -f %s -e --mafft -v %s/*.gff' % (path_roary_results,path_large_temp_roary)
        run_file.write(cmd)
        run_file.write("\n")
        run_file.write("wait")
        run_file.write("\n")
        run_file.write("echo \" one finished. \"")
        run_file.write("\n")


    if f_phylo_roary=='step3':#3rd roary bash file
        run_file.write("\n")
        # run_file.write("(")
        cmd = 'fasttree -nt -gtr %s/core_gene_alignment.aln > %s/my_tree.newick' % (path_roary_results,path_roary_results)
        run_file.write(cmd)
        # run_file.write(")&")
        run_file.write("\n")
        run_file.write("wait")
        run_file.write("\n")
        run_file.write("echo \" one finished. \"")
        # run_file.write("\n")
        # run_file.write("wait")
        # run_file.write("\n")
    if f_phylo_roary=='step4':#3rd roary bash file
        run_file.write("\n")
        # run_file.write("(")
        cmd = 'Rscript --vanilla phylo_tree.r -f \'%s/core_gene_alignment.aln\' -o \'%s/nj_tree.newick\'' % (
        path_roary_results, path_roary_results)
        run_file.write(cmd)
        # run_file.write(")&")
        run_file.write("\n")
        run_file.write("wait")
        run_file.write("\n")
        run_file.write("echo \" one finished. \"")
        # run_file.write("\n")
        # run_file.write("wait")
        # run_file.write("\n")


    if f_pre_cluster==True:

        data_preparation.merge_scaffolds_khuModified.extract_info(path_sequence,path_metadata,path_large_temp_kma ,16)

        print(species,anti,': finished merge_scaffold!')
    if f_cluster==True:
        path_to_kma_clust = ("kma_clustering")
        # if Neisseria gonorrhoeae, then use  -ht 1.0 -hq 1.0, otherwise, all genomes will be clustered into one group #todo check

        run_file.write("(")
        if species in ['Neisseria gonorrhoeae','Mycobacterium tuberculosis','Klebsiella pneumoniae']:
            if species=='Mycobacterium tuberculosis' and (anti in ['amikacin','ethambutol','ethiomide','ethionamide','kanamycin','ofloxacin']):# ethambutol
                cmd = "cat %s | %s -i -- -k 16 -Sparse - -ht 0.99 -hq 0.99 -NI -o %s &> %s" \
                      % (path_large_temp_kma, path_to_kma_clust, path_cluster_temp, path_cluster_results)
            else:
                cmd="cat %s | %s -i -- -k 16 -Sparse - -ht 0.98 -hq 0.98 -NI -o %s &> %s"\
                     %(path_large_temp_kma ,path_to_kma_clust,path_cluster_temp,path_cluster_results)

        else:
            cmd = "cat %s | %s -i -- -k 16 -Sparse - -ht 0.9 -hq 0.9 -NI -o %s &> %s" \
                  % (path_large_temp_kma, path_to_kma_clust, path_cluster_temp, path_cluster_results)
        run_file.write(cmd)
        run_file.write("\n")
        run_file.write("wait")
        run_file.write("\n")
        # run_file.write('rm '+ path_large_temp_kma )
        run_file.write("\n")
        run_file.write('rm ' + path_cluster_temp+'.seq.b '+path_cluster_temp+'.length.b '+path_cluster_temp+'.name '+path_cluster_temp+'.comp.b')
        run_file.write("\n")
        run_file.write("echo \" one thread finised! \"")
        run_file.write(")&")
        run_file.write("\n")

    if f_cluster_folders == True:
        try:
            folders_sample,split_original,_ = pre_cluster_folders.prepare_folders(cv, random,path_metadata, path_cluster_results,'original')
        except:
            folders_sample, split_original=[],[]
        folders_sample_new,split_new_k,folder_sampleName_new = pre_cluster_folders.prepare_folders(cv, random, path_metadata, path_cluster_results,'new')
        return split_original,split_new_k,folder_sampleName_new#for Ehsan


    if f_res==True:
        #2. Analysing PointFinder results
        # Analysing ResFinder results
        if species in ['Klebsiella pneumoniae','Escherichia coli','Staphylococcus aureus','Mycobacterium tuberculosis','Salmonella enterica',
                       'Neisseria gonorrhoeae','Enterococcus faecium','Campylobacter jejuni']:
            data_preparation.scored_representation_blast_khuModified.extract_info(path_res_result,path_metadata,
                                                                                         path_point_repre_results,True,True)#SNP, the last para not zip format.

        data_preparation.ResFinder_analyser_blast_khuModified.extract_info(path_res_result,path_metadata,
                                                                           path_res_repre_results,True)#GPA,the last para means not zip format.

    if f_merge_mution_gene==True:
        # 3. Merging ResFinder and PointFinder results
        if species in ['Klebsiella pneumoniae','Escherichia coli','Staphylococcus aureus','Mycobacterium tuberculosis','Salmonella enterica',
                       'Neisseria gonorrhoeae','Enterococcus faecium','Campylobacter jejuni']:
            data_preparation.merge_resfinder_pointfinder_khuModified.extract_info(path_point_repre_results,
                                                                              path_res_repre_results,path_mutation_gene_results)
        else:#only AMR gene feature
            data_preparation.merge_resfinder_khuModified.extract_info(path_metadata,path_res_repre_results,path_mutation_gene_results)


    if f_matching_io==True:
        #4. Matching input and output results
        data_preparation.merge_input_output_files_khuModified.extract_info(path_metadata,path_mutation_gene_results,path_metadata_pheno,path_x_y)

    if f_merge_species==True:
        #5. Multi-species merging

        print('Please use main_discrete_merge.py or main_concatenated_merge.py')




    if f_nn==True:
        #6. nn
        # save_name_score = amr_utility.name_utility.GETname_multi_bench_save_name_score(species, anti, level,
        #                                                                                learning, epochs,
        #                                                                                f_fixed_threshold,
        #                                                                                f_nn_base,
        #                                                                                f_optimize_score)#none-hyperparmeter mode.
        # none-hyperparmeter mode.
        # nn_module.eval(species, anti, level, path_x,path_y, path_name, path_cluster_results, cv, random, hidden, epochs,
        #                re_epochs, learning,f_scaler, f_fixed_threshold,f_nn_base,f_optimize_score,save_name_score,None,None,None) # the last 3 Nones mean not concat multi-s model.
        #f_nn_base corresponds to no early stoppping
        save_name_score = amr_utility.name_utility.GETname_multi_bench_save_name_score(species, anti, level,
                                                                                       learning, epochs,
                                                                                       f_fixed_threshold,
                                                                                       f_nn_base,
                                                                                       f_optimize_score)#if learning=0.0, and epoch = 0, it means hyper parameter selection mode.

        score=nn_module_hyper.eval(species, anti, level, path_x,path_y, path_name, path_cluster_results, cv, random,
                             re_epochs, f_scaler, f_fixed_threshold, f_nn_base,f_phylotree,f_random, f_optimize_score, save_name_score,
                             learning, epochs,None,None,None,'res')# hyperparmeter selection in inner loop of nested CV
        if f_phylotree:
            with open(save_name_score + '_all_score_Tree.pickle', 'wb') as f:  # overwrite
                pickle.dump(score, f)
        elif f_random:
            with open(save_name_score + '_all_score_Random.pickle', 'wb') as f:
                pickle.dump(score, f)
        else:
            with open(save_name_score + '_all_score.pickle', 'wb') as f:
                pickle.dump(score, f)

def make_dir(name):
    logDir = os.path.join(name)
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
            print("Can't create logging directory:", logDir)

def extract_info(path_sequence,s,level,f_phylo_prokka,f_phylo_roary,f_pre_cluster,f_cluster,f_cluster_folders,f_res,f_merge_mution_gene,f_matching_io,f_merge_species,f_nn,cv, random,
                 hidden, epochs, re_epochs, learning,f_scaler,f_fixed_threshold,f_nn_base,f_phylotree,f_kma,f_random,f_optimize_score,n_jobs,f_all,debug,f_qsub):
    if path_sequence=='/net/projects/BIFO/patric_genome':
        path_large_temp='/net/sgi/metagenomics/data/khu/benchmarking/phylo'
    else:
        fileDir = os.path.dirname(os.path.realpath('__file__'))
        path_large_temp = os.path.join(fileDir, 'large_temp')
        print(path_large_temp)


    data = pd.read_csv('metadata/' + str(level) + '_Species_antibiotic_FineQuality.csv', index_col=0,
                       dtype={'genome_id': object},sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    # for training model on part of the dataset:-------------
    # data=data.loc[['Escherichia coli'],:]
    if f_all ==False:
        data = data.loc[s, :]
    # --------------------------------------------------------
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    # print(data)
    make_dir(path_large_temp + '/clustering')
    make_dir(path_large_temp + '/prokka')
    make_dir(path_large_temp + '/roary')


    for species in df_species:
        make_dir('log/temp/' + str(level) + '/' + str(species.replace(" ", "_")))
        make_dir('log/results/' + str(level) + '/' + str(species.replace(" ", "_")))
        make_dir('log/qsub/' + str(level) + '/' )
    if f_qsub:
        for species, antibiotics in zip(df_species, antibiotics):
            amr_utility.file_utility.make_dir('log/qsub')
            run_file_name='log/qsub/'+str(species.replace(" ", "_"))+'.sh'
            amr_utility.file_utility.make_dir('log/qsub')
            run_file = open(run_file_name, "w")
            run_file.write("#!/bin/bash")
            run_file.write("\n")
            # if path_sequence == '/vol/projects/BIFO/patric_genome':
            if Path(fileDir).parts[1] == 'vol':
                run_file = amr_utility.file_utility.hzi_cpu_header5(run_file,
                                                                    str(species.replace(" ", "_")),
                                                                    'multi_bench_torch','gpu.q')

            cmd = 'python main_feature.py -f_nn -f_phylotree -f_fixed_threshold -f_optimize_score \'f1_macro\' -learning 0.0 -e 0 -s "%s"' % (species)
            run_file.write(cmd)
            run_file.write("\n")
            # ==========================================================================
            run_file_name='log/qsub/'+str(species.replace(" ", "_"))+'_kma.sh'
            amr_utility.file_utility.make_dir('log/qsub')
            run_file = open(run_file_name, "w")
            run_file.write("#!/bin/bash")
            run_file.write("\n")
            # if path_sequence == '/vol/projects/BIFO/patric_genome':
            if Path(fileDir).parts[1] == 'vol':
                run_file = amr_utility.file_utility.hzi_cpu_header5(run_file,
                                                                    str(species.replace(" ", "_")),
                                                                    'multi_bench_torch','gpu.q')

            cmd = 'python main_feature.py -f_nn -f_kma -f_fixed_threshold -f_optimize_score \'f1_macro\' -learning 0.0 -e 0 -s "%s"' % (species)
            run_file.write(cmd)
            run_file.write("\n")
            #==================================================================================
            run_file_name='log/qsub/'+str(species.replace(" ", "_"))+'_random.sh'
            amr_utility.file_utility.make_dir('log/qsub')
            run_file = open(run_file_name, "w")
            run_file.write("#!/bin/bash")
            run_file.write("\n")
            # if path_sequence == '/vol/projects/BIFO/patric_genome':
            if Path(fileDir).parts[1] == 'vol':
                run_file = amr_utility.file_utility.hzi_cpu_header5(run_file,
                                                                    str(species.replace(" ", "_")),
                                                                    'multi_bench_torch','gpu.q')

            cmd = 'python main_feature.py -f_nn -f_random -f_fixed_threshold -f_optimize_score \'f1_macro\' -learning 0.0 -e 0 -s "%s"' % (species)
            run_file.write(cmd)
            run_file.write("\n")


    if debug==True:#no n_job
        for species, antibiotics in zip(df_species, antibiotics):
            print(species)
            run_file = None

            antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
            # for anti in ['mupirocin', 'penicillin', 'rifampin', 'tetracycline', 'vancomycin']:
            for anti in antibiotics:
                run(path_sequence,path_large_temp,species, anti, level, f_phylo_prokka,f_phylo_roary,f_pre_cluster, f_cluster,f_cluster_folders, run_file, f_res, f_merge_mution_gene, f_matching_io,
                    f_merge_species, f_nn, cv, random, hidden, epochs, re_epochs, learning, f_scaler, f_fixed_threshold,f_nn_base,f_phylotree,f_random,f_optimize_score)

    else:
        #should be starting here when everything finished.
        if f_cluster==True:
            # C program from bash
            # kma clustering of samples
            # no n_job version:
            make_dir('cv_folders/' + str(level))

            for species, antibiotics in zip(df_species, antibiotics):
                # produce a bash file
                run_file = open('./cv_folders/' + str(level) + '/' + str(species.replace(" ", "_")) + "_kma.sh", "w")
                run_file.write("#!/bin/bash")
                run_file.write("\n")


                antibiotics_selected = ast.literal_eval(antibiotics)
                print(species)
                print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)
                antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
                if path_sequence == '/vol/projects/BIFO/patric_genome':
                    run_file = amr_utility.file_utility.hzi_cpu_header(run_file,
                                                                    str(species.replace(" ", "_")) + "_kma",len(antibiotics_selected))
                for anti in antibiotics:
                    run(path_sequence,path_large_temp,species, anti, level, f_phylo_prokka,f_phylo_roary,f_pre_cluster, f_cluster,f_cluster_folders, run_file, f_res, f_merge_mution_gene,
                        f_matching_io,
                        f_merge_species, f_nn, cv, random, hidden, epochs, re_epochs, learning, f_scaler,
                        f_fixed_threshold,f_nn_base,f_phylotree,f_random,f_optimize_score)
                run_file.write("echo \" running... \"")
                run_file.close()

        elif f_cluster_folders == True:#check if clusters are reasonably split into folders
            split_original_all, split_new_k_all = [], []
            combination=[]
            for species, antibiotics in zip(df_species, antibiotics):
                run_file=None

                antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)

                for anti in antibiotics:
                    split_original,split_new_k, splits_new_name=run(path_sequence, path_large_temp, species, anti, level, f_phylo_prokka, f_phylo_roary,
                        f_pre_cluster, f_cluster, f_cluster_folders,run_file, f_res, f_merge_mution_gene,
                        f_matching_io,f_merge_species, f_nn, cv, random, hidden, epochs, re_epochs, learning, f_scaler,
                        f_fixed_threshold, f_nn_base,f_phylotree,f_random,f_optimize_score)
                    combination.append(species+' '+anti)
                    try:
                        std_o= statistics.stdev(split_original)
                    except:
                        pass
                    std_n = statistics.stdev(split_new_k)
                    split_original_all.append(std_o)
                    split_new_k_all.append(std_n)

                    #for Ehsan. Have generated, so comment in order to avoid unnecessary modification.
                    '''
                    print(splits_new_name)
                    make_dir('cv_folders/' + str(level) + '/G2P/')
                    with open('cv_folders/' + str(level) + '/G2P/'+str(species.replace(" ", "_"))+'_'+str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
                              +'_CVfolders.txt', 'w') as f:  # overwrite
                        f.writelines('\n'.join([' '.join(i) for i in splits_new_name]))

                    '''
            #plot difference of std
            amr_utility.file_utility.plot_kma_split_dif(split_original_all,split_new_k_all,level,combination)

            #todo plot phenotype distributions in each folder
            # print('folders with only one phenotype:')





        elif f_phylo_prokka==True:
            make_dir('cv_folders/' + str(level))
            run_file = open('./cv_folders/run_prokka.sh', "w")
            run_file.write("#!/bin/bash")
            run_file.write("\n")
            if path_sequence=='/vol/projects/BIFO/patric_genome':
                run_file=amr_utility.file_utility.hzi_cpu_header(run_file,'run_prokka', 20)

            for species, antibiotics in zip(df_species, antibiotics):

                #for storage large prokka and roary temp files
                make_dir(path_large_temp+'/prokka/' + str(species.replace(" ", "_")))


                run(path_sequence,path_large_temp,species, antibiotics, level, f_phylo_prokka,f_phylo_roary,f_pre_cluster, f_cluster, f_cluster_folders,run_file, f_res, f_merge_mution_gene,
                    f_matching_io,
                    f_merge_species, f_nn, cv, random, hidden, epochs, re_epochs, learning, f_scaler,
                    f_fixed_threshold, f_nn_base,f_phylotree,f_random,f_optimize_score)
            run_file.write("echo \" All finished. \"")
            run_file.close()


        elif f_phylo_roary==True:
            # -------------------------------------------------------------------------------------------------------
            #  all isolates are involved in at least one model. Only need to run once.
            for species, antibiotics in zip(df_species, antibiotics):

                antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
                ALL = []
                for anti in antibiotics:
                    name = 'metadata/model/' + str(level) + '/Data_' + str(species.replace(" ", "_")) + '_' + str(
                        anti.translate(str.maketrans({'/': '_', ' ': '_'}))) + '.txt'
                    name_list = pd.read_csv(name, index_col=0, dtype={'genome_id': object}, sep="\t")

                    ALL.append(name_list)

                # combine the list for all antis
                species_dna = ALL[0]
                for i in ALL[1:]:
                    species_dna = pd.merge(species_dna, i, how="outer", on=["genome_id"])  # merge antibiotics within one species
                print(species_dna)
                species_dna_final = species_dna.loc[:, ['genome_id']]
                amr_utility.file_utility.make_dir('cv_folders/'+str(level) + '/'+str(species.replace(" ", "_")))
                species_dna_final.to_csv('cv_folders/'+str(level) + '/'+str(species.replace(" ", "_"))+'/id_list', sep="\t", index=False, header=False)


            cv_folders.create_phylotree.pre_bash_roary(df_species,antibiotics,path_large_temp,level,path_sequence)
            # for species, antibiotics in zip(df_species, antibiotics):
            #     # for storage large prokka and roary temp files
            #
            #     amr_utility.file_utility.make_dir(path_large_temp + '/results_roary/' + str(level))
            #
            #     # # ----------------by each anti, may no need any more. Aug 10th.
            #     #
            #
            #     #1.
            #     run_file = open('./cv_folders/' + str(level) + '/' + str(species.replace(" ", "_")) + "_roary1.sh", "w")
            #     run_file.write("#!/bin/bash")
            #     run_file.write("\n")
            #     if path_sequence == '/vol/projects/BIFO/patric_genome':
            #         run_file = amr_utility.file_utility.hzi_cpu_header(run_file, str(species.replace(" ", "_")) +'run_roary1', 2)
            #     antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
            #     for anti in antibiotics:
            #
            #         run(path_sequence,path_large_temp,species, anti, level, f_phylo_prokka, f_phylo_roary, f_pre_cluster, f_cluster,f_cluster_folders, run_file, f_res,
            #             f_merge_mution_gene,
            #             f_matching_io, f_merge_species, f_nn, cv, random, hidden, epochs, re_epochs, learning, f_scaler,
            #             f_fixed_threshold, f_nn_base,f_phylotree,f_optimize_score)
            #     run_file.write("echo \" one species finished. \"")
            #     run_file.close()
            #     print('------------------------------')
            #     #2.
            #     f_phylo_roary = 'step2'
            #     run_file = open('./cv_folders/' + str(level) + '/' + str(species.replace(" ", "_")) + "_roary2.sh", "w")
            #     run_file.write("#!/bin/bash")
            #     run_file.write("\n")
            #     if path_sequence == '/vol/projects/BIFO/patric_genome':
            #         run_file = amr_utility.file_utility.hzi_cpu_header(run_file,str(species.replace(" ", "_")) + 'run_roary2', 20)
            #     # antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
            #     for anti in antibiotics:
            #
            #         run(path_sequence,path_large_temp,species, anti, level, f_phylo_prokka, f_phylo_roary, f_pre_cluster, f_cluster,f_cluster_folders, run_file, f_res,
            #             f_merge_mution_gene,
            #             f_matching_io, f_merge_species, f_nn, cv, random, hidden, epochs, re_epochs, learning, f_scaler,
            #             f_fixed_threshold, f_nn_base,f_phylotree,f_optimize_score)
            #     run_file.write("echo \" All finished. \"")
            #     run_file.close()# ----------------by each anti, may no need any more. Aug 10th.
            #     #4.
            #     f_phylo_roary = 'step4'  # R package phylo-trees
            #     run_file = open('./cv_folders/' + str(level) + '/' + str(species.replace(" ", "_")) + "_roary4.sh", "w")
            #     run_file.write("#!/bin/bash")
            #     run_file.write("\n")
            #     if path_sequence == '/vol/projects/BIFO/patric_genome':
            #         run_file = amr_utility.file_utility.hzi_cpu_header2(run_file,
            #                                                             str(species.replace(" ", "_")) + 'run_roary4',
            #                                                             1)
            #         run_file.write("\n")
            #     # run_file.write("export OMP_NUM_THREADS=20")
            #     run_file.write("\n")
            #     # antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
            #     for anti in antibiotics:
            #         run(path_sequence, path_large_temp, species, anti, level, f_phylo_prokka, f_phylo_roary,
            #             f_pre_cluster, f_cluster, f_cluster_folders, run_file, f_res,
            #             f_merge_mution_gene,
            #             f_matching_io, f_merge_species, f_nn, cv, random, hidden, epochs, re_epochs, learning, f_scaler,
            #             f_fixed_threshold, f_nn_base, f_phylotree,f_optimize_score)

        elif f_merge_species == True:
            print('Please use main_discrete_merge.')
            exit()


        elif f_nn==True or f_nn_base==True:
            for species, antibiotics in zip(df_species, antibiotics):
                # only for preparing qsub files, comment later.


                # produce a bash file
                run_file=None
                antibiotics_selected = ast.literal_eval(antibiotics)
                print(species)
                print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)
                antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)

                name_weights_folder=amr_utility.name_utility.GETname_multi_bench_folder(species,level,learning,epochs,f_fixed_threshold,f_nn_base,f_phylotree,f_random,f_optimize_score)
                # temp folder for storing weights
                make_dir(name_weights_folder)
                print(name_weights_folder)
                # for anti in ['ciprofloxacin']:
                # for anti in antibiotics:
                if species=='Mycobacterium tuberculosis':
                    # for anti in ['pyrazinamide','rifampicin','rifampin','streptomycin']:
                    for anti in ['streptomycin']:
                        run(path_sequence,path_large_temp,species, anti, level,f_phylo_prokka, f_phylo_roary,f_pre_cluster, f_cluster,f_cluster_folders, run_file, f_res, f_merge_mution_gene,
                            f_matching_io,f_merge_species, f_nn, cv, random, hidden, epochs, re_epochs, learning, f_scaler,
                            f_fixed_threshold,f_nn_base,f_phylotree,f_random,f_optimize_score)
                elif species=='Escherichia coli':
                    antibiotics=antibiotics[5:]
                    for anti in antibiotics:

                        run(path_sequence,path_large_temp,species, anti, level,f_phylo_prokka, f_phylo_roary,f_pre_cluster, f_cluster,f_cluster_folders, run_file, f_res, f_merge_mution_gene,
                            f_matching_io,f_merge_species, f_nn, cv, random, hidden, epochs, re_epochs, learning, f_scaler,
                            f_fixed_threshold,f_nn_base,f_phylotree,f_random,f_optimize_score)
                elif species=='Acinetobacter baumannii':
                    antibiotics=antibiotics[2:]
                    for anti in antibiotics:
                        run(path_sequence,path_large_temp,species, anti, level,f_phylo_prokka, f_phylo_roary,f_pre_cluster, f_cluster,f_cluster_folders, run_file, f_res, f_merge_mution_gene,
                            f_matching_io,f_merge_species, f_nn, cv, random, hidden, epochs, re_epochs, learning, f_scaler,
                            f_fixed_threshold,f_nn_base,f_phylotree,f_random,f_optimize_score)
                elif species=='Staphylococcus aureus':
                    antibiotics=[antibiotics[0]]
                    for anti in antibiotics:
                        run(path_sequence,path_large_temp,species, anti, level,f_phylo_prokka, f_phylo_roary,f_pre_cluster, f_cluster,f_cluster_folders, run_file, f_res, f_merge_mution_gene,
                            f_matching_io,f_merge_species, f_nn, cv, random, hidden, epochs, re_epochs, learning, f_scaler,
                            f_fixed_threshold,f_nn_base,f_phylotree,f_random,f_optimize_score)
                elif species=='Klebsiella pneumoniae':
                    antibiotics=antibiotics[3:]
                    for anti in antibiotics:
                    # for anti in ['trimethoprim/sulfamethoxazole']:
                        run(path_sequence,path_large_temp,species, anti, level,f_phylo_prokka, f_phylo_roary,f_pre_cluster, f_cluster,f_cluster_folders, run_file, f_res, f_merge_mution_gene,
                            f_matching_io,f_merge_species, f_nn, cv, random, hidden, epochs, re_epochs, learning, f_scaler,
                            f_fixed_threshold,f_nn_base,f_phylotree,f_random,f_optimize_score)
                elif species=='Salmonella enterica':
                    antibiotics=antibiotics[1:]
                    for anti in antibiotics:
                    # for anti in ['trimethoprim/sulfamethoxazole']:
                        run(path_sequence,path_large_temp,species, anti, level,f_phylo_prokka, f_phylo_roary,f_pre_cluster, f_cluster,f_cluster_folders, run_file, f_res, f_merge_mution_gene,
                            f_matching_io,f_merge_species, f_nn, cv, random, hidden, epochs, re_epochs, learning, f_scaler,
                            f_fixed_threshold,f_nn_base,f_phylotree,f_random,f_optimize_score)
                else:

                    for anti in antibiotics:
                        run(path_sequence,path_large_temp,species, anti, level,f_phylo_prokka, f_phylo_roary,f_pre_cluster, f_cluster,f_cluster_folders, run_file, f_res, f_merge_mution_gene,
                            f_matching_io,f_merge_species, f_nn, cv, random, hidden, epochs, re_epochs, learning, f_scaler,
                            f_fixed_threshold,f_nn_base,f_phylotree,f_random,f_optimize_score)
                # main_nn.make_visualization(species, antibiotics,level,f_fixed_threshold,epochs,learning)



        elif f_nn == True and n_jobs>1:# on cpu

            for species, antibiotics in zip(df_species, antibiotics):
                # only for preparing qsub files, comment later.


                # produce a bash file
                run_file=None
                antibiotics_selected = ast.literal_eval(antibiotics)
                print(species)
                print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)
                antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)

                name_weights_folder=amr_utility.name_utility.GETname_multi_bench_folder(species,level,learning,epochs,f_fixed_threshold,f_nn_base,f_optimize_score)
                # temp folder for storing weights
                make_dir(name_weights_folder)
                print(name_weights_folder)

                pool = mp.Pool(processes=n_jobs)
                pool.starmap(run,
                             zip(repeat(path_sequence), repeat(path_large_temp), repeat(species), antibiotics,
                                 repeat(level), repeat(f_phylo_prokka),
                                 repeat(f_phylo_roary), repeat(f_pre_cluster), repeat(f_cluster),
                                 repeat(f_cluster_folders), repeat(run_file), repeat(f_res),
                                 repeat(f_merge_mution_gene), repeat(f_matching_io), repeat(f_merge_species),
                                 repeat(f_nn), repeat(cv),
                                 repeat(random), repeat(hidden), repeat(epochs), repeat(re_epochs), repeat(learning),
                                 repeat(f_scaler), repeat(f_fixed_threshold),
                                 repeat(f_nn_base),repeat(f_phylotree), repeat(f_random),repeat(f_optimize_score)))

                pool.close()
                pool.join()

            # no use nay more . Aug 10.
            # for species, antibiotics in zip(df_species, antibiotics):
            #     antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
            #
            #
            #     # if path_sequence == '/vol/projects/BIFO/patric_genome':#make a transfer file, for extract gff
            #     with open('./log/temp/' + str(level) + '/' + str(species.replace(" ", "_")) + '/auc.sh', 'w') as outfile:
            #         outfile.write("#!/bin/bash")
            #
            #         outfile.write("\n")
            #         if path_sequence == '/vol/projects/BIFO/patric_genome':
            #             outfile = amr_utility.file_utility.hzi_cpu_header4(outfile,
            #                                                                str(species.replace(" ", "_")) + '_auc', len(antibiotics),
            #                                                                'multi_bench_torch', 'all.q')
            #         outfile.write("\n")
            #         cmd=(
            #             "python main_feature.py --n_jobs %s -f_nn -f_optimize_score \'auc\' -learning 0.0 -e 0 -s \'%s\'") % (str(len(antibiotics)),species)
            #         outfile.write(cmd)
            #
            #     with open('./log/temp/' + str(level) + '/' + str(species.replace(" ", "_")) + '/f1_select.sh',
            #               'w') as outfile:
            #         outfile.write("#!/bin/bash")
            #
            #         outfile.write("\n")
            #         if path_sequence == '/vol/projects/BIFO/patric_genome':
            #             outfile = amr_utility.file_utility.hzi_cpu_header4(outfile,
            #                                                                str(species.replace(" ", "_")) + '_f1select', len(antibiotics),
            #                                                                'multi_bench_torch', 'all.q')
            #         outfile.write("\n")
            #         cmd=(
            #             "python main_feature.py --n_jobs %s -f_nn -f_optimize_score \'f1_macro\' -learning 0.0 -e 0 -s \'%s\'") % (str(len(antibiotics)),species)
            #         outfile.write(cmd)
            #
            #     with open('./log/temp/' + str(level) + '/' + str(species.replace(" ", "_")) + '/f1.sh', 'w') as outfile:
            #         outfile.write("#!/bin/bash")
            #
            #         outfile.write("\n")
            #         if path_sequence == '/vol/projects/BIFO/patric_genome':
            #             outfile = amr_utility.file_utility.hzi_cpu_header4(outfile, str(species.replace(" ", "_")) + '_f1',
            #                                                                len(antibiotics), 'multi_bench_torch', 'all.q')
            #         outfile.write("\n")
            #         cmd=(
            #             "python main_feature.py --n_jobs %s -f_nn -f_fixed_threshold -f_optimize_score \'f1_macro\' -learning 0.0 -e 0 -s \'%s\'") % (str(len(antibiotics)),species)
            #
            #         outfile.write(cmd)
        else:#other process, should be very light and fast.
            for species in df_species:

                print(species)
                antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)

                run_file=None
                pool = mp.Pool(processes=n_jobs)
                pool.starmap(run,
                             zip(repeat(path_sequence),repeat(path_large_temp),repeat(species), antibiotics, repeat(level),repeat(f_phylo_prokka),
                                 repeat(f_phylo_roary),repeat( f_pre_cluster),repeat(f_cluster),repeat(f_cluster_folders),repeat(run_file),repeat(f_res),
                                 repeat(f_merge_mution_gene),repeat(f_matching_io),repeat(f_merge_species),repeat(f_nn),repeat(cv),
                                 repeat(random),repeat(hidden),repeat(epochs),repeat(re_epochs),repeat(learning),repeat(f_scaler),repeat(f_fixed_threshold),
                                 repeat(f_nn_base),repeat(f_phylotree),repeat(f_random),repeat(f_optimize_score)))

                pool.close()
                pool.join()






if __name__== '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-path_sequence', '--path_sequence', default= '/net/projects/BIFO/patric_genome', type=str, required=False,
    #                     help='path of sequence,another option: \'/vol/projects/BIFO/patric_genome\'')
    # parser.add_argument('-path_large_temp', '--path_large_temp', default='/net/sgi/metagenomics/data/khu/benchmarking/phylo', type=str,
    #                     required=False, help='path for large temp files/folders, another option: \'/vol/projects/khu/amr/benchmarking/large_temp\'')
    parser.add_argument('-path_sequence', '--path_sequence', default='/vol/projects/BIFO/patric_genome', type=str,
                        required=False,
                        help='path of sequence,another option: \'/net/projects/BIFO/patric_genome\'')
    # parser.add_argument('-path_large_temp', '--path_large_temp',
    #                     default='./large_temp', type=str,
    #                     required=False,
    #                     help='path for large temp files/folders, another option: \'/net/sgi/metagenomics/data/khu/benchmarking/phylo\'')
    parser.add_argument('-f_phylo_prokka', '--f_phylo_prokka', dest='f_phylo_prokka', action='store_true',
                        help='phylo-tree based split bash generating, w.r.t. Prokka.')
    parser.add_argument('-f_phylo_roary', '--f_phylo_roary', dest='f_phylo_roary', action='store_true',
                        help='phylo-tree based split bash generating, w.r.t. Roary')

    parser.add_argument('-f_pre_cluster', '--f_pre_cluster', dest='f_pre_cluster', action='store_true',
                        help='Kma cluster bash generating')
    parser.add_argument('-f_cluster', '--f_cluster', dest='f_cluster', action='store_true',
                                            help='Kma cluster')#c program
    parser.add_argument('-f_cluster_folders', '--f_cluster_folders', dest='f_cluster_folders', action='store_true',
                        help='Compare new split method with old(original) method.')

    parser.add_argument('-f_res', '--f_res', dest='f_res', action='store_true',
                        help='Analyse Point/ResFinder results')

    parser.add_argument('-f_merge_mution_gene', '--f_merge_mution_gene', dest='f_merge_mution_gene', action='store_true',
                        help=' Merging ResFinder and PointFinder results')
    parser.add_argument('-f_matching_io', '--f_matching_io', dest='f_matching_io', action='store_true',
                        help='Matching input and output results')
    parser.add_argument('-f_merge_species', '--f_merge_species', dest='f_merge_species', action='store_true',
                        help='Multi-species merging')

    #para for nn nestedCV
    parser.add_argument('-f_nn', '--f_nn', dest='f_nn', action='store_true',
                        help='Run the NN model')
    parser.add_argument("-cv", "--cv_number", default=10, type=int,
                        help='CV splits number')
    parser.add_argument("-r", "--random", default=42, type=int,
                        help='random state related to shuffle cluster order')
    parser.add_argument("-d", "--hidden", default=200, type=int,
                        help='dimension of hidden layer')
    parser.add_argument("-e", "--epochs", default=1000, type=int,
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
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                        help='kma based cv folders.')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    parser.add_argument('-f_random', '--f_random', dest='f_random', action='store_true',
                        help='random cv folders.')
    parser.add_argument('-f_optimize_score', '--f_optimize_score', default='f1_macro', type=str,
                        help='optimize score for choosing the best estimator in inner loop.')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument('-f_qsub', '--f_qsub', dest='f_qsub',
                        help='Prepare scriptd for qsub.', action='store_true', )
    #----------------------------------------------------------------------------------------------------------------
    # parser.add_argument("-o","--output", default=None, type=str, required=True,
	# 					help='Output file names')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
            \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
            \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parser.add_argument('-debug', '--debug', dest='debug', action='store_true',
                        help='debug')

    parsedArgs = parser.parse_args()
    # parser.print_help()
    # print(parsedArgs)
    # extract_info(parsedArgs.species,parsedArgs.xdata,parsedArgs.ydata,parsedArgs.p_names,parsedArgs.p_clusters,parsedArgs.cv_number,
                 # parsedArgs.random,parsedArgs.hidden,parsedArgs.epochs,parsedArgs.re_epochs,parsedArgs.learning,parsedArgs.f_scaler,
                #parsedArgs.f_fixed_threshold,parsedArgs.level,parsedArgs.n_jobs)
    extract_info(parsedArgs.path_sequence,parsedArgs.species,parsedArgs.level,parsedArgs.f_phylo_prokka,
                 parsedArgs.f_phylo_roary,parsedArgs.f_pre_cluster,parsedArgs.f_cluster,parsedArgs.f_cluster_folders,parsedArgs.f_res,
                 parsedArgs.f_merge_mution_gene,parsedArgs.f_matching_io,parsedArgs.f_merge_species,
                 parsedArgs.f_nn,parsedArgs.cv_number,parsedArgs.random,parsedArgs.hidden,parsedArgs.epochs,parsedArgs.re_epochs,
                 parsedArgs.learning,parsedArgs.f_scaler,parsedArgs.f_fixed_threshold,parsedArgs.f_nn_base,parsedArgs.f_phylotree,parsedArgs.f_kma,parsedArgs.f_random,
                 parsedArgs.f_optimize_score,parsedArgs.n_jobs,parsedArgs.f_all,parsedArgs.debug,parsedArgs.f_qsub)
