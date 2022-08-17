import sys
import os
sys.path.insert(0, os.path.abspath('../'))
import amr_utility.name_utility
import amr_utility.load_data
import analysis_results.extract_score
import analysis_results.make_table
import analysis_results.math_utility
import amr_utility.file_utility
import ast
import argparse
import pickle
import pandas as pd
import numpy as np
import statistics
from scipy.stats import ttest_rel
import math
import seaborn as sns


def get_mean_std(f1_pos_sub):
    '''
    :param f1_pos_sub: a list of scores from CV
    :return: mean and std in a str format
    '''
    f1_pos_mean = statistics.mean(f1_pos_sub)
    f1_pos_std = statistics.stdev(f1_pos_sub)
    f1_pos_m_s = str(round(f1_pos_mean,2))+'±'+str(round(f1_pos_std,2))
    return f1_pos_m_s

def extract_info_species(level,species,fscore,antibiotics,cv,f_phylotree,f_kma,old_version):
    '''
    Sep 10th, 2021.
    [Low level]for each species and each classifier
    summary_table_ByClassifier
    |'f1_macro'| 'precision_macro'|'recall_macro'|'accuracy'|
    'mcc'|'f1_positive'| 'f1_negative'| 'precision_positive'|'recall_positive'|'auc'|'selected hyperparameter'
    ceftazidime|0.85±0.20|...
    ciprofloxacin|
    '''
    antibiotics_selected = ast.literal_eval(antibiotics)
    print(species)
    print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)
    antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
    # cl_list=['svm','lr','lsvm','rf','et','ab','gb','xgboost']
    cl_list = ['resfinder']
    for chosen_cl in cl_list:
        print('---------------------',chosen_cl)
        hy_para_fre=[]
        hy_para_fren = []
        summary_table_ByClassifier_all = []
        for anti in antibiotics:

            _, _, save_name_score = amr_utility.name_utility.Pts_GETname(level, species, anti,chosen_cl)
            # save_name_score = amr_utility.name_utility.GETsave_name_score(species, anti, chosen_cl)

            score = pickle.load(open(save_name_score + '_kma_' + str(f_kma) + '_tree_' + str(f_phylotree) + '.pickle',"rb"))  # todo,check
            [f1_test, score_report_test, aucs_test, mcc_test]=score
            # print(hyperparameters_test)
            # common,ind = analysis_results.math_utility.get_most_fre_hyper(hyperparameters_test)
            # hy_para_fre.append(common.to_dict())
            # hy_para_fren.append(ind)

            summary_table_ByClassifier_ = pd.DataFrame(index=['mean', 'std', 'weighted-mean', 'weighted-std'],
                                   columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy_macro',
                                            'mcc', 'f1_positive', 'f1_negative', 'precision_neg', 'recall_neg',
                                            'auc','threshold', 'support', 'support_positive'])
            if f1_test[0]!=None:
                if f_kma:# extract infor from report
                    summary_table_ByClassifier= analysis_results.extract_score.score_summary(fscore,None, summary_table_ByClassifier_, cv, score_report_test, f1_test,aucs_test,
                                                                               mcc_test, save_name_score,
                                                                               np.zeros(cv))# the last 0: no meaning.
                else:# f_phylotree or random
                    #todo, still need check. should be fine.Sep 10.
                    summary_table_ByClassifier = analysis_results.extract_score.score_summary_Tree(fscore,None, summary_table_ByClassifier_, cv, score_report_test,
                                                                                    f1_test,aucs_test, mcc_test,
                                                                                    save_name_score,
                                                                                    np.zeros(cv))# the last 0: no meaning.

            else:
                summary_table_ByClassifier=summary_table_ByClassifier_
            summary_table_ByClassifier_all.append(summary_table_ByClassifier)


            # print(summary_table_ByClassifier)

        #finish one chosen_cl
        #[Low level]for each species and each classifier
        out_score='f' #['weighted-f1_macro','weighted-f1_positive', 'weighted-f1_negative','weighted-accuracy'
        if f_kma:
            final, final_plot,final_std = analysis_results.make_table.make_visualization(out_score, summary_table_ByClassifier_all, antibiotics,
                                                                                   level, None, None,
                                                                                   None,
                                                                                   None,
                                                                                   None)
        else:#if f_phylotree or random
            final, final_plot,final_std = analysis_results.make_table.make_visualization_Tree(out_score, summary_table_ByClassifier_all, antibiotics,
                                                                                        level,
                                                                                        None, None,
                                                                                        None,
                                                                                        None,
                                                                                        None)



        save_name_score_final,_ = amr_utility.name_utility.GETsave_name_final(fscore,species,f_kma,f_phylotree,chosen_cl)
        # print(final)
        # print(hyperparameters_test)
        # final['selected hyperparameter'] = hy_para_fre
        # final['frequency'] = hy_para_fren
        #change column names
        final_plot=final_plot.rename(columns={"weighted-f1_macro": "f1_macro", "weighted-f1_positive": "f1_positive","weighted-f1_negative": "f1_negative", "weighted-accuracy": "accuracy" })
        final=final.rename(columns={"weighted-f1_macro": "f1_macro", "weighted-f1_positive": "f1_positive","weighted-f1_negative": "f1_negative", "weighted-accuracy": "accuracy" })
        final_std=final_std.rename(columns={"weighted-f1_macro": "f1_macro", "weighted-f1_positive": "f1_positive","weighted-f1_negative": "f1_negative", "weighted-accuracy": "accuracy" })

        final.to_csv(save_name_score_final + '.txt', sep="\t")
        final_plot.to_csv(save_name_score_final + '_PLOT.txt', sep="\t")
        final_std.to_csv(save_name_score_final + '_std.txt', sep="\t")
        print(final)







def extract_info(l,s,fscore,cv,f_phylotree,f_kma,f_benchmarking,old_version,f_plot,f_all):
    data = pd.read_csv('metadata/' + str(l) + '_Species_antibiotic_FineQuality.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    # data = pd.read_csv('metadata/Species_antibiotic_FineQuality.csv', index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    if f_all == False:
        data = data.loc[s, :]
    # data.at['Mycobacterium tuberculosis', 'modelling antibiotics']=['capreomycin', 'ciprofloxacin']
    # --------------------------------------------------------
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    amr_utility.file_utility.make_dir('log/results/'+fscore)
    print(data)
    for df_species, antibiotics in zip(df_species, antibiotics):
        extract_info_species(l, df_species, fscore, antibiotics, cv,f_phylotree,f_kma,old_version)
        # plot(l, df_species, fscore, antibiotics, cv, f_phylotree, f_kma,old_version)
        # extract_best_estimator(l, df_species, fscore, antibiotics, cv,f_phylotree,f_kma)




if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--l', '--level', default='loose', type=str, required=False,
                        help='Quality control: strict or loose')
    parser.add_argument("-cv", "--cv_number", default=10, type=int,
                        help='CV splits number')
    # parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
         \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
         \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    # parser.add_argument('-cl', '--cl', default=['svm'], type=str, nargs='+',
    #                     help='classifier to train: e.g.\'svm\',\'lr\',\'lsvm\',\'rf\',\'et\',\'ab\',\'gb\',\'xgboost\'')
    parser.add_argument('-fscore', '--fscore', default='f1_macro', type=str, required=False,
                        help='the score used to choose the best classifier for each antibiotic. Can be one of: \'f1_macro\',\'f1_positive\',\'f1_negative\',\'accuracy\'')
    parser.add_argument('-old_version', '--old_version', dest='old_version', action='store_true',
                        help='Old version of scripts. model_cv.py')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                        help='kma based cv folders.')
    parser.add_argument('-f_plot', '--f_plot', dest='f_plot', action='store_true',
                        help='plot.')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument('-f_benchmarking', '--f_benchmarking', dest='f_benchmarking', action='store_true',
                        help='Extract the best estimator for benchmarking. First, run the script without this flag. Then, use this flag. ')
    # parser.add_argument('-score', '--score', default='f', type=str,required=False,
    #                     help='Scores of the final output table. f:f_macro,f_pos,f_neg,f_micro. all:all scores. f_p_r:f1_macro,precision,recall,accuracy')

    # parser.set_defaults(canonical=True)
    parsedArgs = parser.parse_args()
    # parser.print_help()
    print(parsedArgs)
    extract_info(parsedArgs.l,parsedArgs.species,parsedArgs.fscore,parsedArgs.cv_number,parsedArgs.f_phylotree,parsedArgs.f_kma,
                 parsedArgs.f_benchmarking,parsedArgs.old_version,parsedArgs.f_plot,parsedArgs.f_all)
