
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import numpy as np
import ast
import statistics
import operator
import seaborn as sns
from matplotlib import pyplot as plt
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
import main_nn
import neural_networks.Neural_networks_khuModified_earlys as nn_module
import neural_networks.Neural_networks_khuModified as nn_module_original
import csv
import neural_networks.cluster_folders


def run(path_sequence,path_large_temp,species,anti,level,f_phylo_prokka,f_phylo_roary,f_pre_cluster,f_cluster,f_cluster_folders,run_file,f_res,f_merge_mution_gene,f_matching_io,f_merge_species,f_nn,cv,
        random, hidden, epochs, re_epochs, learning,f_scaler,f_fixed_threshold,f_nn_base):
    logDir = os.path.join('log/temp/' + str(level)+'/'+str(species.replace(" ", "_")))
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
            print("Can't create logging directory:", logDir)
    logDir = os.path.join('log/results/'+str(level))
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
            print("Can't create logging directory:", logDir)
    print(species, anti, '-------------------------------------------------')
    # path_sequence = '/net/projects/BIFO/patric_genome'# in the future, change it to custom setting variable.
    # path_to_nn = ("./Neural_Networks/Neural_networks.py")
    path_to_kma_clust = ("kma_clustering")

    #names of temp files
    save_name_meta,save_name_modelID=amr_utility.name_utility.save_name_modelID(level, species, anti, f=False)
    # save_name_species=str(level)+'/'+str(species.replace(" ", "_"))
    save_name_anti=str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
    name_species_anti = str(species.replace(" ", "_")) + '_' + str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
    '''
    save_name_modelID='metadata/model/'+str(level)+'/Data_' + str(species.replace(" ", "_")) + '_' + str(
            anti.translate(str.maketrans({'/': '_', ' ': '_'})))
    '''
    path_feature = './log/temp/' + str(level)+'/'+str(species.replace(" ", "_")) # all feature temp data(except large files)
    # path_res_result='/net/flashtest/scratch/khu/benchmarking/Results/'+str(species.replace(" ", "_"))#old
    path_res_result = '/net/sgi/metagenomics/data/khu/benchmarking/Results/' + str(species.replace(" ", "_"))
    path_metadata='./'+save_name_modelID #anti,species
    # path_large_temp='/net/flashtest/scratch/khu/benchmarking/Results/clustering/'+name_species_anti+'all_strains_assembly.txt'#
    path_large_temp_kma = '/net/sgi/metagenomics/data/khu/benchmarking/Results/clustering/' + name_species_anti + 'all_strains_assembly.txt'
    path_large_temp_prokka =path_large_temp+'/prokka/'+str(species.replace(" ", "_"))
    path_large_temp_roary=path_large_temp+'/roary/' + str(level) +'/'+ name_species_anti
    path_metadata_prokka='metadata/model/id_' + str(species.replace(" ", "_"))
    path_cluster_temp=path_feature+'/clustered_90_'+save_name_anti# todo check!!! checked
    path_metadata_pheno=path_metadata+'resfinder'
    #temp results
    path_roary_results=path_large_temp+'/results/'+str(level) +'/'+ name_species_anti
    path_cluster_results=path_feature+'/'+save_name_anti+'_clustered_90.txt'
    path_point_repre_results=path_feature+'/'+save_name_anti+'_mutations.txt'
    path_res_repre_results = path_feature + '/' + save_name_anti + '_acquired_genes.txt'
    path_mutation_gene_results=path_feature + '/' + save_name_anti + '_res_point.txt'
    path_x_y = path_feature + '/' + save_name_anti + '_final_'

    path_x = path_x_y+'data_x.txt'
    path_y = path_x_y+'data_y.txt'
    path_name = path_x_y + 'data_names.txt'

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
        logDir = os.path.join(path_large_temp_roary)
        if not os.path.exists(logDir):
            try:
                os.makedirs(logDir)
            except OSError:
                print("Can't create logging directory:", logDir)


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
        cmd = 'FastTree -nt -gtr %s/core_gene_alignment.aln > my_tree.newick' % path_roary_results
        run_file.write(cmd)
        run_file.write("\n")
        run_file.write("wait")
        run_file.write("\n")
        run_file.write("echo \" one finished. \"")
        run_file.write("\n")
        run_file.write("wait")
        run_file.write("\n")



    if f_pre_cluster==True:
        data_preparation.merge_scaffolds_khuModified.extract_info(path_sequence,path_metadata,path_large_temp_kma ,16)

        print(species,anti,': finished merge_scaffold!')
    if f_cluster==True:
        # if Neisseria gonorrhoeae, then use  -ht 1.0 -hq 1.0, otherwise, all genomes will be clustered into one group #todo check

        run_file.write("(")
        cmd="cat %s | %s -i -- -k 16 -Sparse - -ht 0.9 -hq 0.9 -NI -o %s &> %s"\
             %(path_large_temp_kma ,path_to_kma_clust,path_cluster_temp,path_cluster_results)
        run_file.write(cmd)
        run_file.write("\n")
        run_file.write("wait")
        run_file.write("\n")
        run_file.write('rm '+ path_large_temp_kma )
        run_file.write("\n")
        run_file.write('rm ' + path_cluster_temp+'.seq.b '+path_cluster_temp+'.length.b '+path_cluster_temp+'.name'+path_cluster_temp+'.comp.b')
        run_file.write("\n")
        run_file.write("echo \" one thread finised! \"")
        run_file.write(")&")
        run_file.write("\n")

    if f_cluster_folders == True:

        split_original = neural_networks.cluster_folders.prepare_folders(cv, random,path_name, path_cluster_results,'original')
        split_new_k = neural_networks.cluster_folders.prepare_folders(cv, random, path_name, path_cluster_results,'new')
        return split_original,split_new_k

    if f_res==True:
        #2. Analysing PointFinder results
        # Analysing ResFinder results
        if species in ['Klebsiella pneumoniae','Escherichia coli','Staphylococcus aureus','Mycobacterium tuberculosis','Salmonella enterica',
                       'Neisseria gonorrhoeae','Enterococcus faecium','Campylobacter jejuni']:
            data_preparation.scored_representation_blast_khuModified.extract_info(path_res_result,path_metadata,
                                                                                         path_point_repre_results,True)

        data_preparation.ResFinder_analyser_blast_khuModified.extract_info(path_res_result,path_metadata,
                                                                           path_res_repre_results)

    if f_merge_mution_gene==True:
        # 3. Merging ResFinder and PointFinder results
        if species in ['Klebsiella pneumoniae','Escherichia coli','Staphylococcus aureus','Mycobacterium tuberculosis','Salmonella enterica',
                       'Neisseria gonorrhoeae','Enterococcus faecium','Campylobacter jejuni']:
            data_preparation.merge_resfinder_pointfinder_khuModified.extract_info(path_point_repre_results,
                                                                              path_res_repre_results,path_mutation_gene_results)
        else:#only AMR gene feature
            data_preparation.merge_resfinder_khuModified.extract_info(path_metadata,path_res_repre_results,path_mutation_gene_results)
            pass

    if f_matching_io==True:
        #4. Matching input and output results
        data_preparation.merge_input_output_files_khuModified.extract_info(path_mutation_gene_results,path_metadata_pheno,path_x_y)

    if f_merge_species==True:
        #5. Multi-species merging
        #todo
        pass



    if f_nn==True:
        #6. nn
        # main_nn.extract_info(species,path_x,path_y,path_name,path_cluster_results,cv, random, hidden, epochs, re_epochs,
        #                      learning,f_scaler,f_fixed_threshold,level)

        nn_module.eval(species, anti, level, path_x,path_y, path_name, path_cluster_results, cv, random, hidden, epochs,
                       re_epochs, learning,f_scaler, f_fixed_threshold)



    if f_nn_base == True:#benchmarking baseline.
        nn_module_original.eval(species, anti, level, path_x,path_y, path_name, path_cluster_results, cv, random, hidden, epochs,
                       re_epochs, learning,f_scaler, f_fixed_threshold)


# def extract_info(s,xdata,ydata,p_names,p_clusters,cv_number, random, hidden, epochs, re_epochs, learning,f_scaler,
#                  f_fixed_threshold, level,n_jobs):
def extract_info(path_sequence,path_large_temp,s,level,f_phylo_prokka,f_phylo_roary,f_pre_cluster,f_cluster,f_cluster_folders,f_res,f_merge_mution_gene,f_matching_io,f_merge_species,f_nn,cv, random,
                 hidden, epochs, re_epochs, learning,f_scaler,f_fixed_threshold,f_nn_base,n_jobs,debug):



    data = pd.read_csv('metadata/' + str(level) + '_Species_antibiotic_FineQuality.csv', index_col=0,
                       dtype={'genome_id': object},sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    # for training model on part of the dataset:-------------
    # data=data.loc[['Escherichia coli'],:]
    data = data.loc[s, :]
    # --------------------------------------------------------
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    # print(data)
    if debug==True:#no n_job
        for species, antibiotics in zip(df_species, antibiotics):
            print(species)
            antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)

            run_file = None

            antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
            # for anti in ['mupirocin', 'penicillin', 'rifampin', 'tetracycline', 'vancomycin']:
            for anti in antibiotics:
                run(path_sequence,path_large_temp,species, anti, level, f_phylo_prokka,f_phylo_roary,f_pre_cluster, f_cluster,f_cluster_folders, run_file, f_res, f_merge_mution_gene, f_matching_io,
                    f_merge_species, f_nn, cv, random, hidden, epochs, re_epochs, learning, f_scaler, f_fixed_threshold,f_nn_base)

    else:
        #should be starting here when everything finished.
        if f_cluster==True:
            # C program from bash
            # kma clustering of samples
            # no n_job version:
            logDir = os.path.join('cv_folders/' + str(level))
            if not os.path.exists(logDir):
                try:
                    os.makedirs(logDir)
                except OSError:
                    print("Can't create logging directory:", logDir)

            for species, antibiotics in zip(df_species, antibiotics):
                # produce a bash file
                run_file = open('./cv_folders/' + str(level) + '/' + str(species.replace(" ", "_")) + "_kma.sh", "w")
                run_file.write("#!/bin/bash")
                run_file.write("\n")
                antibiotics_selected = ast.literal_eval(antibiotics)
                print(species)
                print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)
                antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
                # for anti in ['mupirocin', 'penicillin', 'rifampin', 'tetracycline', 'vancomycin']:
                for anti in antibiotics:
                    run(path_sequence,path_large_temp,species, anti, level, f_phylo_prokka,f_phylo_roary,f_pre_cluster, f_cluster,f_cluster_folders, run_file, f_res, f_merge_mution_gene,
                        f_matching_io,
                        f_merge_species, f_nn, cv, random, hidden, epochs, re_epochs, learning, f_scaler,
                        f_fixed_threshold,f_nn_base)
                run_file.write("echo \" running... \"")
                run_file.close()

        elif f_cluster_folders == True:
            split_original_all, split_new_k_all = [], []
            for species, antibiotics in zip(df_species, antibiotics):
                run_file=None

                antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
                # for anti in ['mupirocin', 'penicillin', 'rifampin', 'tetracycline', 'vancomycin']:
                for anti in antibiotics:
                    split_original,split_new_k=run(path_sequence, path_large_temp, species, anti, level, f_phylo_prokka, f_phylo_roary,
                        f_pre_cluster, f_cluster, f_cluster_folders,run_file, f_res, f_merge_mution_gene,
                        f_matching_io,f_merge_species, f_nn, cv, random, hidden, epochs, re_epochs, learning, f_scaler,
                        f_fixed_threshold, f_nn_base)
                    std_o= statistics.stdev(split_original)
                    std_n = statistics.stdev(split_new_k)
                    split_original_all.append(std_o)
                    split_new_k_all.append(std_n)

            #plot difference of std
            amr_utility.file_utility.plot_kma_split_dif(split_original_all,split_new_k_all,level)


        elif f_phylo_prokka==True:
            logDir = os.path.join('cv_folders/' + str(level))
            if not os.path.exists(logDir):
                try:
                    os.makedirs(logDir)
                except OSError:
                    print("Can't create logging directory:", logDir)
            run_file = open('./cv_folders/run_prokka.sh', "w")
            run_file.write("#!/bin/bash")
            run_file.write("\n")
            if path_sequence=='/vol/projects/BIFO/patric_genome':
                run_file=amr_utility.file_utility.hzi_cpu_header(run_file,'run_prokka', 40)

            for species, antibiotics in zip(df_species, antibiotics):

                #for storage large prokka and roary temp files
                logDir = os.path.join(path_large_temp+'/prokka/' + str(
                        species.replace(" ", "_")))
                if not os.path.exists(logDir):
                    try:
                        os.makedirs(logDir)
                    except OSError:
                        print("Can't create logging directory:", logDir)

                run(path_sequence,path_large_temp,species, antibiotics, level, f_phylo_prokka,f_phylo_roary,f_pre_cluster, f_cluster, f_cluster_folders,run_file, f_res, f_merge_mution_gene,
                    f_matching_io,
                    f_merge_species, f_nn, cv, random, hidden, epochs, re_epochs, learning, f_scaler,
                    f_fixed_threshold, f_nn_base)
            run_file.write("echo \" All finished. \"")
            run_file.close()

        elif f_phylo_roary==True:
            # -------------------------------------------------------------------------------------------------------
            for species, antibiotics in zip(df_species, antibiotics):
                # for storage large prokka and roary temp files
                # logDir = os.path.join( path_large_temp+'/roary/' +str(level)+'/'+ str(
                #     species.replace(" ", "_")))
                # if not os.path.exists(logDir):
                #     try:
                #         os.makedirs(logDir)
                #     except OSError:
                #         print("Can't create logging directory:", logDir)
                # for storage roary results
                logDir = os.path.join(path_large_temp+'/results/' + str(level))
                if not os.path.exists(logDir):
                    try:
                        os.makedirs(logDir)
                    except OSError:
                        print("Can't create logging directory:", logDir)

                #for storage large prokka and roary temp files
                #1.
                f_phylo_roary = True
                run_file = open('./cv_folders/' + str(level) + '/' + str(species.replace(" ", "_")) + "_roary1.sh", "w")
                # run_file.write("#!/bin/bash")
                run_file.write("\n")
                antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
                for anti in antibiotics:
                    run(path_sequence,path_large_temp,species, anti, level, f_phylo_prokka, f_phylo_roary, f_pre_cluster, f_cluster,f_cluster_folders, run_file, f_res,
                        f_merge_mution_gene,
                        f_matching_io, f_merge_species, f_nn, cv, random, hidden, epochs, re_epochs, learning, f_scaler,
                        f_fixed_threshold, f_nn_base)
                run_file.write("echo \" All finished. \"")
                run_file.close()
                #2.
                f_phylo_roary = 'step2'
                run_file = open('./cv_folders/' + str(level) + '/' + str(species.replace(" ", "_")) + "_roary2.sh", "w")
                # run_file.write("#!/bin/bash")
                run_file.write("\n")
                antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
                for anti in antibiotics:

                    run(path_sequence,path_large_temp,species, anti, level, f_phylo_prokka, f_phylo_roary, f_pre_cluster, f_cluster,f_cluster_folders, run_file, f_res,
                        f_merge_mution_gene,
                        f_matching_io, f_merge_species, f_nn, cv, random, hidden, epochs, re_epochs, learning, f_scaler,
                        f_fixed_threshold, f_nn_base)
                run_file.write("echo \" All finished. \"")
                run_file.close()
            #3. join files that belong to the same species
            # -------------------------------------------------------------------------------------------------------
            ID_files=[]
            for species, antibiotics in zip(df_species, antibiotics):
                ID_files.append('./cv_folders/' + str(level) + '/' + str(species.replace(" ", "_")) + "_roary1.sh")

            with open('./cv_folders/run_roary1.sh', 'w') as outfile:
                outfile.write("#!/bin/bash")
                outfile.write("\n")
                if path_sequence == '/vol/projects/BIFO/patric_genome':
                    outfile = amr_utility.file_utility.hzi_cpu_header(outfile, 'run_roary1', 2)
                for names in ID_files:
                    # Open each file in read mode
                    with open(names) as infile:
                        outfile.write(infile.read())
                    outfile.write("\n")
            #-------------------------------------------------------------------------------------------------------
            ID_files=[]
            for species, antibiotics in zip(df_species, antibiotics):
                ID_files.append('./cv_folders/' + str(level) + '/' + str(species.replace(" ", "_")) + "_roary2.sh")

            with open('./cv_folders/run_roary2.sh', 'w') as outfile:
                outfile.write("#!/bin/bash")
                outfile.write("\n")
                if path_sequence == '/vol/projects/BIFO/patric_genome':
                    outfile = amr_utility.file_utility.hzi_cpu_header(outfile, 'run_roary2', 22)
                for names in ID_files:
                    # Open each file in read mode
                    with open(names) as infile:
                        outfile.write(infile.read())
                    outfile.write("\n")


        elif f_nn==True or f_nn_base==True or f_merge_species==True:
            #f_nn:Gpu, on luna


            for species, antibiotics in zip(df_species, antibiotics):


                # produce a bash file
                run_file=None
                antibiotics_selected = ast.literal_eval(antibiotics)
                print(species)
                print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)
                antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)

                name_weights_folder=amr_utility.name_utility.name_multi_bench_folder(species,level,learning,epochs,f_fixed_threshold)
                # temp folder for storing weights
                logDir = os.path.join(name_weights_folder)
                if not os.path.exists(logDir):
                    try:
                        os.makedirs(logDir)
                    except OSError:
                        print("Can't create logging directory:", logDir)
                # for anti in ['ciprofloxacin']:
                for anti in antibiotics:
                    run(path_sequence,path_large_temp,species, anti, level,f_phylo_prokka, f_phylo_roary,f_pre_cluster, f_cluster,f_cluster_folders, run_file, f_res, f_merge_mution_gene,
                        f_matching_io,f_merge_species, f_nn, cv, random, hidden, epochs, re_epochs, learning, f_scaler,
                        f_fixed_threshold,f_nn_base)
                # main_nn.make_visualization(species, antibiotics,level,f_fixed_threshold,epochs,learning)



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
                                 repeat(random),repeat(hidden),repeat(epochs),repeat(re_epochs),repeat(learning),repeat(f_scaler),repeat(f_fixed_threshold),repeat(f_nn_base)))

                pool.close()
                pool.join()






if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_sequence', '--path_sequence', default= '/net/projects/BIFO/patric_genome', type=str, required=False,
                        help='path of sequence,another option: \'/vol/projects/BIFO/patric_genome\'')
    parser.add_argument('-path_large_temp', '--path_large_temp', default='/net/sgi/metagenomics/data/khu/benchmarking/phylo', type=str,
                        required=False, help='path for large temp files/folders, another option: \'/vol/projects/khu/amr/benchmarking/large_temp\'')

    parser.add_argument('-f_phylo_prokka', '--f_phylo_prokka', dest='f_phylo_prokka', action='store_true',
                        help='phylo-tree based split bash generating, w.r.t. Prokka.')
    parser.add_argument('-f_phylo_roary', '--f_phylo_roary', dest='f_phylo_roary', action='store_true',
                        help='phylo-tree based split bash generating, w.r.t. Roary')

    parser.add_argument('-f_pre_cluster', '--f_pre_cluster', dest='f_pre_cluster', action='store_true',
                        help='Kma cluster bash generating')
    parser.add_argument('-f_cluster', '--f_cluster', dest='f_cluster', action='store_true',
                                            help='Kma cluster')#c program
    parser.add_argument('-f_cluster_folders', '--f_cluster_folders', dest='f_cluster_folders', action='store_true',
                        help='Compare new split method with old(original) method.')  # c program

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
    extract_info(parsedArgs.path_sequence,parsedArgs.path_large_temp,parsedArgs.species,parsedArgs.level,parsedArgs.f_phylo_prokka,
                 parsedArgs.f_phylo_roary,parsedArgs.f_pre_cluster,parsedArgs.f_cluster,parsedArgs.f_cluster_folders,parsedArgs.f_res,
                 parsedArgs.f_merge_mution_gene,parsedArgs.f_matching_io,parsedArgs.f_merge_species,
                 parsedArgs.f_nn,parsedArgs.cv_number,parsedArgs.random,parsedArgs.hidden,parsedArgs.epochs,parsedArgs.re_epochs,
                 parsedArgs.learning,parsedArgs.f_scaler,parsedArgs.f_fixed_threshold,parsedArgs.f_nn_base,parsedArgs.n_jobs,parsedArgs.debug)