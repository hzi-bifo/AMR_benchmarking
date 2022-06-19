import os
import numpy as np
import ast
import amr_utility.name_utility
import amr_utility.graph_utility
import amr_utility.file_utility
import argparse
import amr_utility.load_data
import analysis_results.extract_score
import analysis_results.make_table
import analysis_results.math_utility
import pandas as pd
import pickle
import statistics
from scipy.stats import ttest_rel
import math
from collections import Counter
from pandas.plotting import table
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel


'''Paired T test of any two of single-s model mean,  discrete multiple-s model, concatenated mixed species model. concat multi-s model no left out species version.'''
#5 folds + 1 test fold
# python main_nn_analysis_hyper.py --T_test_hyper_opt -f_all #(-fscore 'f1_negative') ## T test of f1 macro between default hyperparameters and hyperparameter optimization. only f_kma. only single-s.


def extract_info(final_score,level,f_all,threshold_point,min_cov_point,learning,epochs,f_optimize_score,f_fixed_threshold,
                 f_nn_base,f_phylotree,cv):
    data = pd.read_csv('metadata/' +str(level)+'_multi-species_summary.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    # --------------------------------------------------------
    print(data)
    # print(data.index)
    merge_name = []
    if f_all:
        list_species = data.index.tolist()[:-1]
        data = data.loc[list_species, :]

    else:
        pass
    for n in list_species:
        merge_name.append(n[0] + n.split(' ')[1][0])
    merge_name = '_'.join(merge_name)  # e.g.Se_Kp_Pa
    # rearrange species order:
    list_species=['Mycobacterium tuberculosis','Campylobacter jejuni','Salmonella enterica','Escherichia coli','Streptococcus pneumoniae',\
                  'Klebsiella pneumoniae','Staphylococcus aureus','Acinetobacter baumannii','Pseudomonas aeruginosa']
    # list_species=['Mycobacterium tuberculosis','Campylobacter jejuni','Salmonella enterica','Streptococcus pneumoniae','Escherichia coli','Staphylococcus aureus','Klebsiella pneumoniae','Acinetobacter baumannii',]
    data=data.reindex(list_species)
    # data=data.loc[list_species, :]
    print(data)
    df_anti = data.dot(data.columns + ';').str.rstrip(';')
    print(df_anti)


    summary_plot = pd.DataFrame(columns=[final_score, 'antibiotic', 'model'])
    for species in list_species:

        print('------------------------------',species)
        #1. single-s model

        single_s_score = amr_utility.name_utility.GETname_multi_bench_save_name_final(final_score,merge_name,
                                                                                                 'all_possible_anti', level,
                                                                                                 learning,
                                                                                                 epochs, f_fixed_threshold,
                                                                                                 f_nn_base,
                                                                                                 f_optimize_score)

        single_results = pd.read_csv(os.path.dirname(single_s_score)+ '/single_species_f1_macro.txt', index_col=0,sep="\t")

        # 4. ---------discrete model.
        save_name_score_final = amr_utility.name_utility.GETname_multi_bench_save_name_final('f1_macro',merge_name,
                                                                                                 'all_possible_anti', level,
                                                                                                 learning,
                                                                                                 epochs, f_fixed_threshold,
                                                                                                 f_nn_base,
                                                                                                 f_optimize_score)
        dis_results=pd.read_csv(save_name_score_final+'/split_discrete_model.txt',  index_col=0,sep="\t")
        # 3 . -------------------multi-s concateM model
        save_name_score_final = amr_utility.name_utility.GETname_multi_bench_save_name_concatM_final(merge_name,merge_name,
                                                                                                        level,learning,epochs,
                                                                                                        f_fixed_threshold,
                                                                                                        f_nn_base,
                                                                                                        f_optimize_score,
                                                                                                        threshold_point,
                                                                                                        min_cov_point)

        concatM_results=pd.read_csv(save_name_score_final+'/split_concate_model.txt', index_col=0,sep="\t")
        #2.-------------------multi-s concate model

        merge_name_test = species.replace(" ", "_")
        concat_s_score=amr_utility.name_utility.GETname_multi_bench_save_name_concat_final(merge_name,
                                                                                        merge_name_test,
                                                                                        level,
                                                                                       learning,
                                                                                       epochs,
                                                                                       f_fixed_threshold,
                                                                                       f_nn_base,
                                                                                       f_optimize_score,
                                                                                        threshold_point,
                                                                                        min_cov_point)

        concat_results=pd.read_csv(concat_s_score + '_score_final.txt', sep="\t", header=0, index_col=0)













        #-------------------------------
        #Prepare dataframe for plotting.
        #-------------------------------
        antibiotics = df_anti[species].split(';')
        for each_anti in antibiotics:

            # print(single_results)
            summary_plot_single = pd.DataFrame(columns=[final_score, 'antibiotic', 'model'])
            summary_plot_single.loc['e'] = [single_results.loc[species,each_anti ],each_anti, 'single-speceis model']
            summary_plot = summary_plot.append(summary_plot_single, ignore_index=True)

            #-------discrete
            #
            summary_plot_dis = pd.DataFrame(columns=[final_score, 'antibiotic', 'model'])
            summary_plot_dis.loc['e'] = [dis_results.loc[species,each_anti ], each_anti, 'Discrete databases multi-species model']
            summary_plot = summary_plot.append(summary_plot_dis, ignore_index=True)

            #------------------------------------------
            #concat M

            summary_plot_concatM = pd.DataFrame(columns=[final_score, 'antibiotic', 'model'])
            summary_plot_concatM.loc['e'] = [concatM_results.loc[species,each_anti ],each_anti, 'Concatenated databases mixed multi-species model']
            summary_plot = summary_plot.append(summary_plot_concatM, ignore_index=True)


            #-----------concat leave-one-out
            # summary_plot_sub.loc[species, each_anti] = data_score.loc[each_anti, each_score]
            summary_plot_multi = pd.DataFrame(columns=[final_score, 'antibiotic', 'model'])
            summary_plot_multi.loc['e'] = [concat_results.loc[each_anti,final_score], each_anti, 'Concatenated databases leave-one-out multi-species model']
            summary_plot = summary_plot.append(summary_plot_multi, ignore_index=True)


    print(summary_plot)#
    df_single=summary_plot.loc[summary_plot['model']=='single-speceis model']
    list_single=df_single[final_score].to_list()
    df_dis=summary_plot.loc[summary_plot['model']=='Discrete databases multi-species model']
    list_dis=df_dis[final_score].to_list()
    df_concatM=summary_plot.loc[summary_plot['model']== 'Concatenated databases mixed multi-species model']
    list_concatM=df_concatM[final_score].to_list()
    df_concat=summary_plot.loc[summary_plot['model']== 'Concatenated databases leave-one-out multi-species model']
    list_concat=df_concat[final_score].to_list()

    result=ttest_rel(list_single, list_dis)
    pvalue = result[1]
    print('list_single, list_dis',pvalue)


    result=ttest_rel(list_single, list_concatM)
    pvalue = result[1]
    print('list_single, list_concatM',pvalue)

    result=ttest_rel(list_single, list_concat)
    pvalue = result[1]
    print('list_single, list_concat',pvalue)


    result=ttest_rel(list_dis, list_concatM)
    pvalue = result[1]
    print('list_dis, list_concatM',pvalue)


    result=ttest_rel(list_dis, list_concat)
    pvalue = result[1]
    print('list_dis, list_concat',pvalue)


    result=ttest_rel(list_concatM, list_concat)
    pvalue = result[1]
    print('list_concatM, list_concat',pvalue)







if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--level', default='loose', type=str, required=False,
                        help='Quality control: strict or loose')
    parser.add_argument('-score', '--score', default='f1_macro', type=str, required=False,
                        help='the score used to choose the best classifier for each antibiotic. Can be f1_pos'
                             'f1_neg, accuracy.')
    parser.add_argument('-t_p', '--threshold_point', default=0.8, type=float,
                        help='Threshold for identity of Pointfinder. ')
    parser.add_argument('-l_p', '--min_cov_point', default=0.6, type=float,
                        help=' Minimum (breadth-of) coverage of Pointfinder. ')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument('-f_fixed_threshold', '--f_fixed_threshold', dest='f_fixed_threshold', action='store_true',
                        help='set a fixed threshod:0.5.')
    parser.add_argument('-f_optimize_score', '--f_optimize_score', default='f1_macro', type=str,
                        help='the optimizing-score for choosing the best estimator in inner loop. Choose: auc or f1_macro.')
    parser.add_argument("-e", "--epochs", default=0, type=int,
                        help='epochs')
    parser.add_argument("-learning", "--learning", default=0.0, type=float,
                        help='learning rate')
    parser.add_argument('-f_nn_base', '--f_nn_base', dest='f_nn_base', action='store_true',
                        help='benchmarking baseline.')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    parser.add_argument("-cv", "--cv_number", default=10, type=int,
                        help='CV splits number')
    parsedArgs = parser.parse_args()
    # parser.print_help()
    # print(parsedArgs)
    extract_info(parsedArgs.score,parsedArgs.level,parsedArgs.f_all,parsedArgs.threshold_point,parsedArgs.min_cov_point,
                 parsedArgs.learning,parsedArgs.epochs,parsedArgs.f_optimize_score,parsedArgs.f_fixed_threshold,parsedArgs.f_nn_base,
                 parsedArgs.f_phylotree,parsedArgs.cv_number)


