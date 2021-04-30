import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import numpy as np
import ast
from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score, KFold,cross_val_predict,cross_validate
from sklearn import svm,preprocessing
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import multiprocessing as mp
import amr_utility.name_utility
import amr_utility.graph_utility
import classifier
import time
import pickle
import argparse
import amr_utility.load_data
import pandas as pd
import neural_networks.Neural_networks_khuModified as nn_module


def make_visualization(species,antibiotics):
    '''
    make final summary
    :return:
    '''
    # path_to_pointfinder = "Results/Point_results" + str(species.replace(" ", "_")) + "/"
    # path_to_resfinder = "Results/Res_results_" + str(species.replace(" ", "_")) + "/"
    # path_to_pr = "Results/" + str(species.replace(" ", "_")) + "/"
    print(species)
    antibiotics_selected = ast.literal_eval(antibiotics)

    print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)
    final=pd.DataFrame(index=antibiotics_selected, columns=['f1-score','precision', 'recall','accuracy'] )
    print(final)
    for anti in antibiotics_selected:
        save_name_score = amr_utility.name_utility.name_multi_bench_save_name_score(species, antibiotics)
        print(anti, '--------------------------------------------------------------------')
        try:
            data = pd.read_csv('log/results/report_'+save_name_score+'.txt', sep="\t")
            print(data)
            final.loc[str(anti),'f1-score']=data.iloc[3,3]
            final.loc[str(anti),'precision'] = data.iloc[3, 1]
            final.loc[str(anti),'recall'] = data.iloc[3, 2]
            final.loc[str(anti),'accuracy'] = data.iloc[2, 2]
            print(final)
        except:
            pass
    final=final.astype(float).round(2)
    final.to_csv('log/results/report_'+str(species.replace(" ", "_"))+'.csv', sep="\t")


def extract_info(s,xdata,ydata,p_names,p_clusters,cv_number, random, hidden, epochs, learning, level,output,n_jobs):
    #for store temp data
    logDir = os.path.join('log/temp/'+str(s.replace(" ", "_")))
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
            print("Can't create logging directory:", logDir)
    logDir = os.path.join('log/results/' + str(s.replace(" ", "_")))
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
            print("Can't create logging directory:", logDir)
    data = pd.read_csv('metadata/loose_Species_antibiotic_FineQuality.csv', index_col=0, dtype={'genome_id': object},
                       sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    # for training model on part of the dataset:-------------
    # data=data.loc[['Escherichia coli'],:]
    data = data.loc[s, :]
    # data.at['Mycobacterium tuberculosis', 'modelling antibiotics']=['capreomycin', 'ciprofloxacin']
    # --------------------------------------------------------
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    print(data)
    # pool = mp.Pool(processes=5)
    # pool.starmap(determination, zip(df_species,repeat(l),repeat(n_jobs)))
    for species in df_species:
        antibiotics_selected = ast.literal_eval(antibiotics)

        print(species)
        print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)
        antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
        # for anti in ['mupirocin', 'penicillin', 'rifampin', 'tetracycline', 'vancomycin']:

        for anti in antibiotics:
            nn_module.eval(species,anti, xdata,ydata,p_names,p_clusters,cv_number, random, hidden, epochs, learning, level,output)

        #put out final table with scores:'f1-score','precision', 'recall','accuracy'
        make_visualization(species, antibiotics)



if __name__== '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-x", "--xdata", default=None, type=float, required=True,
						help='input x data')
    parser.add_argument("-y", "--ydata", default=None, type=int, required=True,
                        help='output y data')# todo check type
    parser.add_argument("-names", "--p_names", default=None, type=str, required=True,
						help='path to list of sample names')
    parser.add_argument("-c", "--p_clusters", default=None, type=str, required=True,
                        help='path to the sample clusters')
    parser.add_argument("-cv", "--cv_number", default=10, type=int, required=True,
                        help='CV splits number')
    parser.add_argument("-r", "--random", default=42, type=int, required=True,
                        help='random state related to shuffle cluster order')
    parser.add_argument("-d", "--hidden", default=200, type=int, required=True,
                        help='dimension of hidden layer')
    parser.add_argument("-e", "--epochs", default=1000, type=int, required=True,
                        help='epochs')
    parser.add_argument("-learing", "--learning", default=0.001, type=int, required=True,
                        help='learning rate')
    parser.add_argument('--l', '--level', default=None, type=str, required=True,
                        help='Quality control: strict or loose')
    parser.add_argument("-o","--output", default=None, type=str, required=True,
						help='Output file names')
    parser.add_argument('--s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
            \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
            \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parsedArgs = parser.parse_args()
    # parser.print_help()
    # print(parsedArgs)
    extract_info(parsedArgs.s,parsedArgs.xdata,parsedArgs.ydata,parsedArgs.p_names,parsedArgs.p_clusters,parsedArgs.cv_number,
                 parsedArgs.random,parsedArgs.hidden,parsedArgs.epochs,parsedArgs.learning,parsedArgs.level,parsedArgs.output,parsedArgs.n_jobs)

