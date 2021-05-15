import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import numpy as np
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
import data_preparation.discrete_merge
import neural_networks.Neural_networks_khuModified_earlys as nn_module
import data_preparation.discrete_merge
# import neural_networks.Neural_networks_khuModified as nn_module_original
import csv
import neural_networks.cluster_folders

##Notes to author self:
# to change if anti choosable: means the codes need to refine if anti is choosable.

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
        data_preparation.discrete_merge.prepare_meta(path_large_temp,list_species,[],level,f_all)#[] means all possible will anti.
    #high level names
    multi_log,path_metadata_multi,path_metadata_pheno_multi,run_file_kma,run_file_roary1,run_file_roary2,run_file_roary3,path_x,path_y,path_name\
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

        make_dir(os.path.dirname(run_file_kma))
        run_file = open(run_file_kma, "w")#to change if anti choosable
        run_file.write("#!/bin/bash")
        run_file.write("\n")
        for s in list_species:
            path_large_temp_kma_multi, path_cluster_temp_multi, path_cluster_results_multi, path_large_temp_roary_multi, \
            path_roary_results_multi, path_metadata_s_multi, path_metadata_pheno_s_multi, path_large_temp_prokka, path_res_result, path_point_repre_results_multi, \
            path_res_repre_results_multi, path_mutation_gene_results_multi, path_feature_multi= \
                amr_utility.name_utility.GETname_multi_bench_multi_species(level, path_large_temp, merge_name, s)
            path_to_kma_clust = ("kma_clustering")
            # if Neisseria gonorrhoeae, then use  -ht 1.0 -hq 1.0, otherwise, all genomes will be clustered into one group #todo check....

            run_file.write("(")
            if s == 'Neisseria gonorrhoeae':
                cmd = "cat %s | %s -i -- -k 16 -Sparse - -ht 0.98 -hq 0.98 -NI -o %s &> %s" \
                      % (path_large_temp_kma_multi, path_to_kma_clust, path_cluster_temp_multi, path_cluster_results_multi)
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
            run_file.write("echo \" one thread finised! \"")
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
    if f_cluster_folders == True:
        pass
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
                                                                                      path_point_repre_results_multi, True)#SNP
            data_preparation.ResFinder_analyser_blast_khuModified.extract_info(path_res_result, path_metadata_s_multi,
                                                                               path_res_repre_results_multi)  # GPA


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
        data_preparation.discrete_merge.merge_feature(merge_name, path_sequence, path_large_temp, list_species, All_antibiotics, level, f_all,
                      f_phylo_prokka, f_phylo_roary,
                      f_pre_cluster, f_cluster, f_cluster_folders, f_res, f_merge_mution_gene, f_matching_io)


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
        nn_module.eval(merge_name, 'all_possible_anti', level, path_x, path_y, path_name, path_cluster_results, cv, random, hidden,
                       epochs,re_epochs, learning, f_scaler, f_fixed_threshold, f_nn_base,f_optimize_score)

    if f_cluster_folders == True:
        #analysis the number of each species' samples in each folder.
        folders_sample,split_original = neural_networks.cluster_folders.prepare_folders(cv, random,path_name, path_cluster_results,'original')
        folders_sample_new,split_new_k = neural_networks.cluster_folders.prepare_folders(cv, random, path_name, path_cluster_results,'new')

        split_original=np.array(split_original)#the number of samples of each species in each folder
        split_new_k=np.array(split_new_k)
        std_o = np.std(split_original, axis=0, ddof=1)
        std_n = np.std(split_new_k, axis=0, ddof=1)

        # split_original_all.append(std_o)
        # split_new_k_all.append(std_n)

        #todo check: make a bar plot for the number of samples of each species, the x-axis is the folder 1-10
        amr_utility.file_utility.plot_kma_split(split_original, split_new_k, level, list_species, merge_name)


def extract_info(path_sequence,path_large_temp,list_species,selected_anti,level,f_all,f_pre_meta,f_phylo_prokka,f_phylo_roary,
                 f_pre_cluster,f_cluster,f_cluster_folders,f_res,f_merge_mution_gene,f_matching_io,f_nn,cv, random,
                 hidden, epochs, re_epochs, learning,f_scaler,f_fixed_threshold,f_nn_base,f_optimize_score):
    merge_name = []

    data = pd.read_csv('metadata/' + str(level) + '_multi-species_summary.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    if f_all:
        list_species = data.index.tolist()[:-1]

    data = data.loc[list_species, :]

    # --------------------------------------------------------
    # drop columns(antibotics) all zero
    data = data.loc[:, (data != 0).any(axis=0)]
    All_antibiotics = data.columns.tolist()  # all envolved antibiotics # todo
    for n in list_species:
        merge_name.append(n[0] + n.split(' ')[1][0])
    merge_name = '_'.join(merge_name)  # e.g.Se_Kp_Pa
    multi_log = './log/temp/' + str(level) + '/multi_species/' + merge_name

    make_dir(multi_log)


    run(merge_name,path_sequence,path_large_temp,list_species,All_antibiotics,level,f_all,f_pre_meta,f_phylo_prokka,f_phylo_roary,f_pre_cluster,f_cluster,f_cluster_folders,f_res,f_merge_mution_gene,f_matching_io,f_nn,cv, random,
             hidden, epochs, re_epochs, learning,f_scaler,f_fixed_threshold,f_nn_base,f_optimize_score)




if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_sequence', '--path_sequence', default='/net/projects/BIFO/patric_genome', type=str,
                        required=False,
                        help='path of sequence,another option: \'/vol/projects/BIFO/patric_genome\'')
    parser.add_argument('-path_large_temp', '--path_large_temp',
                        default='/net/sgi/metagenomics/data/khu/benchmarking/phylo', type=str,
                        required=False,
                        help='path for large temp files/folders, another option: \'/vol/projects/khu/amr/benchmarking/large_temp\'')
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
    extract_info(parsedArgs.path_sequence, parsedArgs.path_large_temp, parsedArgs.species,parsedArgs.anti, parsedArgs.level, parsedArgs.f_all,
                 parsedArgs.f_pre_meta,parsedArgs.f_phylo_prokka,
                 parsedArgs.f_phylo_roary, parsedArgs.f_pre_cluster, parsedArgs.f_cluster, parsedArgs.f_cluster_folders,
                 parsedArgs.f_res,
                 parsedArgs.f_merge_mution_gene, parsedArgs.f_matching_io,
                 parsedArgs.f_nn, parsedArgs.cv_number, parsedArgs.random, parsedArgs.hidden, parsedArgs.epochs,
                 parsedArgs.re_epochs,
                 parsedArgs.learning, parsedArgs.f_scaler, parsedArgs.f_fixed_threshold, parsedArgs.f_nn_base,parsedArgs.f_optimize_score)
