
import os
import numpy as np
import amr_utility.name_utility
import amr_utility.graph_utility
import amr_utility.file_utility
import argparse
import amr_utility.load_data
import pandas as pd


def extract_info(path_sequence,s,level,feature,f_phylo_prokka,f_phylo_roary,f_pre_cluster,f_cluster,f_cluster_folders,f_res,f_merge_mution_gene,f_matching_io,f_merge_species,f_nn,cv, random,
                 hidden, epochs, re_epochs, learning,f_scaler,f_fixed_threshold,f_nn_base,f_phylotree,f_optimize_score,n_jobs,f_all,debug):
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
        amr_utility.file_utility.make_dir('log/temp/' + str(level) + '/' + str(species.replace(" ", "_")))
        amr_utility.file_utility.make_dir('log/results/' + str(level) + '/' + str(species.replace(" ", "_")))
        print(species)
        run_file = None

        antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
        i_anti = 0
        for anti in antibiotics:
            id_all = ID[i_anti]  # sample name list, e.g. [1352.10013,1352.10014,1354.10,1366.10]
            y_all = Y[i_anti]
            i_anti+=1
            # id_all = np.array(id_all)
            y_all = np.array(y_all)
            if feature=='6mer':
                mer6_file = '/vol/projects/khu/amr/patric_Mar/log/feature/kmer/cano_' + str(
                    species.replace(" ", "_")) + '_6_mer.h5'
                data_feature = pd.read_hdf(mer6_file)
                data_feature = data_feature.T
                init_feature = np.zeros((len(id_all), 1), dtype='uint16')
                data_model_init = pd.DataFrame(init_feature, index=id_all, columns=['initializer'])
                X_all = pd.concat([data_model_init, data_feature.reindex(data_model_init.index)], axis=1)
                X_all = X_all.drop(['initializer'], axis=1)
            else:#'s2g'
                s2g_file='/vol/projects/khu/amr/seq2geno/log/temp/loose/'+ str(
                    species.replace(" ", "_"))+'/results/RESULTS/bin_tables'
                data_feature1 = pd.read_csv(s2g_file+'/gpa.mat_NONRDNT', index_col=0,sep="\t")
                data_feature2 = pd.read_csv(s2g_file + '/indel.mat_NONRDNT',index_col=0,sep="\t")


                init_feature = np.zeros((len(id_all), 1), dtype='uint16')
                id_all_new=['iso_'+ s  for s in id_all]
                data_model_init = pd.DataFrame(init_feature, index=id_all_new, columns=['initializer'])
                X_all = pd.concat([data_model_init, data_feature1.reindex(data_model_init.index)], axis=1)
                X_all = pd.concat([X_all, data_feature2.reindex(data_model_init.index)], axis=1)
                X_all = X_all.drop(['initializer'], axis=1)


            path_x, path_y, path_name = amr_utility.name_utility.g2pManu_GETname(species,anti,level,feature)

            X_all.to_csv(path_x, sep="\t", index=False,header=False)
            pd.DataFrame(y_all).to_csv(path_y, sep="\t", index=False,header=False)

            textfile = open(path_name, "w")
            for element in id_all:
                textfile.write(element + "\n")
            textfile.close()







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
    extract_info(parsedArgs.path_sequence,parsedArgs.species,parsedArgs.level,parsedArgs.feature,parsedArgs.f_phylo_prokka,
                 parsedArgs.f_phylo_roary,parsedArgs.f_pre_cluster,parsedArgs.f_cluster,parsedArgs.f_cluster_folders,parsedArgs.f_res,
                 parsedArgs.f_merge_mution_gene,parsedArgs.f_matching_io,parsedArgs.f_merge_species,
                 parsedArgs.f_nn,parsedArgs.cv_number,parsedArgs.random,parsedArgs.hidden,parsedArgs.epochs,parsedArgs.re_epochs,
                 parsedArgs.learning,parsedArgs.f_scaler,parsedArgs.f_fixed_threshold,parsedArgs.f_nn_base,parsedArgs.f_phylotree,parsedArgs.f_optimize_score,parsedArgs.n_jobs,parsedArgs.f_all,parsedArgs.debug)