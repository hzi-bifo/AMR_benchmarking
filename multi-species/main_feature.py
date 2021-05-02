
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
import main_nn

def run(species,anti,level,f_pre_cluster,f_cluster,f_res,f_merge_mution_gene,f_matching_io,f_merge_species,f_nn,cv,
        random, hidden, epochs, re_epochs, learning,f_scaler,f_fixed_threshold):
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
    path_sequence = '/net/projects/BIFO/patric_genome'# in the future, change it to custom setting variable.
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
    path_res_result='/net/flashtest/scratch/khu/benchmarking/Results/'+str(species.replace(" ", "_"))
    path_metadata='./'+save_name_modelID #anti,species
    path_large_temp='/net/flashtest/scratch/khu/benchmarking/Results/clustering/'+name_species_anti+'all_strains_assembly.txt'
    path_cluster_temp=path_feature+'/clustered_90_'+save_name_anti# todo check!!!
    path_metadata_pheno=path_metadata+'resfinder'
    #temp results
    path_cluster_results=path_feature+'/'+save_name_anti+'_clustered_90.txt'
    path_point_repre_results=path_feature+'/'+save_name_anti+'_mutations.txt'
    path_res_repre_results = path_feature + '/' + save_name_anti + '_acquired_genes.txt'
    path_mutation_gene_results=path_feature + '/' + save_name_anti + '_res_point.txt'
    path_x_y = path_feature + '/' + save_name_anti + '_final_'

    path_x = path_x_y+'data_x.txt'
    path_y = path_x_y+'data_y.txt'
    path_name = path_x_y + 'data_names.txt'



    if f_pre_cluster==True:
        data_preparation.merge_scaffolds_khuModified.extract_info(path_sequence,path_metadata,path_large_temp,16)

        print(species,anti,': finished merge_scaffold!')
    if f_cluster==True:
        #will spaw other threads
        subprocess.run("cat %s | %s -i -- -k 16 -Sparse - -ht 0.9 -hq 0.9 -NI -o %s &> %s;wait;echo \' finished. \';"
                       "wait; rm %s %s.seq.b %s.length.b %s.name"
                       % (path_large_temp, path_to_kma_clust, path_cluster_temp, path_cluster_results,path_large_temp,
                          path_cluster_temp, path_cluster_temp,path_cluster_temp),
                       shell=True)
        print(species,anti,': finished kma!')
    if f_res==True:
        #2. Analysing PointFinder results
        # Analysing ResFinder results
        data_preparation.scored_representation_blast_khuModified.extract_info(path_res_result,path_metadata,
                                                                                     path_point_repre_results,True)

        data_preparation.ResFinder_analyser_blast_khuModified.extract_info(path_res_result,path_metadata,
                                                                           path_res_repre_results)

    if f_merge_mution_gene==True:
        # 3. Merging ResFinder and PointFinder results
        data_preparation.merge_resfinder_pointfinder_khuModified.extract_info(path_point_repre_results,
                                                                              path_res_repre_results,path_mutation_gene_results)


    if f_matching_io==True:
        #4. Matching input and output results
        data_preparation.merge_input_output_files_khuModified.extract_info(path_mutation_gene_results,path_metadata_pheno,path_x_y)

    if f_merge_species==True:
        #5. Multi-species merging
        #todo
        pass
    if f_nn==True:
        #6. nn
        main_nn.extract_info(species,path_x,path_y,path_name,path_cluster_results,cv, random, hidden, epochs, re_epochs,
                             learning,f_scaler,f_fixed_threshold)



# def extract_info(s,xdata,ydata,p_names,p_clusters,cv_number, random, hidden, epochs, re_epochs, learning,f_scaler,
#                  f_fixed_threshold, level,n_jobs):
def extract_info(s,level,f_pre_cluster,f_cluster,f_res,f_merge_mution_gene,f_matching_io,f_merge_species,f_nn,cv, random,
                 hidden, epochs, re_epochs, learning,f_scaler,f_fixed_threshold,n_jobs):

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

    # if f_cluster==False:
    for species in df_species:
        print(species)
        antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)

        pool = mp.Pool(processes=n_jobs)
        pool.starmap(run,
                     zip(repeat(species), antibiotics, repeat(level), repeat( f_pre_cluster),repeat(f_cluster), repeat(f_res),
                         repeat(f_merge_mution_gene),repeat(f_matching_io),repeat(f_merge_species),repeat(f_nn),repeat(cv)))
    pool.close()
    pool.join()

    # else:#C program from bash
    #     # no n_job version:
    #     for species, antibiotics in zip(df_species, antibiotics):
    #         antibiotics_selected = ast.literal_eval(antibiotics)
    #         print(species)
    #         print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)
    #         antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
    #         # for anti in ['mupirocin', 'penicillin', 'rifampin', 'tetracycline', 'vancomycin']:
    #         for anti in antibiotics:
    #
    #             run(species, anti, level, f_pre_cluster, f_cluster, f_res, f_nn)



if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f_pre_cluster', '--f_pre_cluster', dest='f_pre_cluster', action='store_true',
                        help='Kma cluster')
    parser.add_argument('-f_cluster', '--f_cluster', dest='f_cluster', action='store_true',
                                            help='Kma cluster')#c program
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
    parser.add_argument("-learning", "--learning", default=0.001, type=int,
                        help='learning rate')
    parser.add_argument('-l', '--level', default=None, type=str, required=True,
                        help='Quality control: strict or loose')
    parser.add_argument('-f_scaler', '--f_scaler', dest='f_scaler', action='store_true',
                        help='normalize the data')
    parser.add_argument('-f_fixed_threshold', '--f_fixed_threshold', dest='f_fixed_threshold', action='store_true',
                        help='set a fixed threshod:0.5.')
    #----------------------------------------------------------------------------------------------------------------
    # parser.add_argument("-o","--output", default=None, type=str, required=True,
	# 					help='Output file names')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
            \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
            \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parsedArgs = parser.parse_args()
    # parser.print_help()
    # print(parsedArgs)
    # extract_info(parsedArgs.species,parsedArgs.xdata,parsedArgs.ydata,parsedArgs.p_names,parsedArgs.p_clusters,parsedArgs.cv_number,
                 # parsedArgs.random,parsedArgs.hidden,parsedArgs.epochs,parsedArgs.re_epochs,parsedArgs.learning,parsedArgs.f_scaler,
                #parsedArgs.f_fixed_threshold,parsedArgs.level,parsedArgs.n_jobs)
    extract_info(parsedArgs.species,parsedArgs.level,parsedArgs.f_pre_cluster,parsedArgs.f_cluster,parsedArgs.f_res,
                 parsedArgs.f_merge_mution_gene,parsedArgs.f_matching_io,parsedArgs.f_merge_species,
                 parsedArgs.f_nn,parsedArgs.cv_number,parsedArgs.random,parsedArgs.hidden,parsedArgs.epochs,parsedArgs.re_epochs,
                 parsedArgs.learning,parsedArgs.f_scaler,parsedArgs.f_fixed_threshold,parsedArgs.n_jobs)