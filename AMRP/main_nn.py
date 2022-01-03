
import os
import ast,pickle
import amr_utility.name_utility
import amr_utility.graph_utility
import amr_utility.file_utility
import argparse
import amr_utility.load_data
import pandas as pd
import neural_networks.Neural_networks as nn_module_hyper
from sklearn.model_selection import ParameterGrid


'''running kmer based NN models'''




def extract_info(path_sequence,s,level,feature,cv, random, epochs, re_epochs, learning,f_scaler,f_fixed_threshold,
                 f_nn_base,f_phylotree,f_kma,f_optimize_score,n_jobs,f_all):
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


    for species, antibiotics in zip(df_species, antibiotics):
        # only for preparing qsub files, comment later.

        antibiotics_selected = ast.literal_eval(antibiotics)
        print(species)
        print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)
        antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)

        name_weights_folder=amr_utility.name_utility.g2pManu_weight_folder(species,level,learning,epochs,f_fixed_threshold,
                                                                    f_nn_base,f_phylotree,f_optimize_score,feature)
        # temp folder for storing weights
        amr_utility.file_utility.make_dir(name_weights_folder)
        print(name_weights_folder)
        for anti in antibiotics:


            for out_cv in range(cv):
                print('Starting outer: ', str(out_cv))


                path_x, path_y, path_name = amr_utility.name_utility.g2pManu_GETname(species,anti,level,feature)#share the same relative path with g2pmanubench.


                save_name_score = amr_utility.name_utility.GETsave_name_score(species, anti, 'nn_'+feature)
                p_clusters= amr_utility.name_utility.GETname_folder(species,anti,level)
                score,score2=nn_module_hyper.eval(species, anti, level, path_x,path_y, path_name, p_clusters, cv, random,
                                     re_epochs, f_scaler, f_fixed_threshold, f_nn_base,f_phylotree, f_optimize_score, save_name_score,learning, epochs,None,None,None,feature)# hyperparmeter selection in inner loop of nested CV
                with open(save_name_score+'_kma_'+str(f_kma)+'_tree_'+str(f_phylotree)+'.pickle', 'wb') as f:  # overwrite
                    pickle.dump(score2, f)
                with open(save_name_score + '_all_score.pickle', 'wb') as f:  # overwrite
                    pickle.dump(score, f)






if __name__== '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-path_sequence', '--path_sequence', default= '/net/projects/BIFO/patric_genome', type=str, required=False,
    #                     help='path of sequence,another option: \'/vol/projects/BIFO/patric_genome\'')
    # parser.add_argument('-path_large_temp', '--path_large_temp', default='/net/sgi/metagenomics/data/khu/benchmarking/phylo', type=str,
    #                     required=False, help='path for large temp files/folders, another option: \'/vol/projects/khu/amr/benchmarking/large_temp\'')
    parser.add_argument('-path_sequence', '--path_sequence', default='/vol/projects/BIFO/patric_genome', type=str,
                        required=False,
                        help='path of sequence,another option: \'/net/projects/BIFO/patric_genome\'')

    #para for nn nestedCV
    # parser.add_argument('-f_nn', '--f_nn', dest='f_nn', action='store_true',
    #                     help='Run the NN model')
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
    parser.add_argument('-feature', '--feature', default='6mer', type=str,
                        help='kmer(k=6,8,10)  or res or s2g')
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
    parser.add_argument('-f_optimize_score', '--f_optimize_score', default='f1_macro', type=str,
                        help='optimize score for choosing the best estimator in inner loop.')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')

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
    extract_info(parsedArgs.path_sequence,parsedArgs.species,parsedArgs.level,parsedArgs.feature,
                 parsedArgs.cv_number,parsedArgs.random,parsedArgs.epochs,parsedArgs.re_epochs,
                 parsedArgs.learning,parsedArgs.f_scaler,parsedArgs.f_fixed_threshold,parsedArgs.f_nn_base,
                 parsedArgs.f_phylotree,parsedArgs.f_kma,parsedArgs.f_optimize_score,parsedArgs.n_jobs,parsedArgs.f_all)
