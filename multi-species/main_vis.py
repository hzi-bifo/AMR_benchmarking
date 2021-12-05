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



#plot a comparative graph of single-s model and concatenated multiple-s model.
def extract_info(final_score,level,f_all,threshold_point,min_cov_point,learning,epochs,f_optimize_score,f_fixed_threshold,
                 f_nn_base,f_phylotree,cv):
    data = pd.read_csv('metadata/' +str(level)+'_multi-species_summary.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    # --------------------------------------------------------
    print(data)
    merge_name = []
    if f_all:
        list_species = data.index.tolist()[:-1]
        data = data.loc[list_species, :]
    else:
        pass
    df_anti = data.dot(data.columns + ';').str.rstrip(';')
    for n in list_species:
        merge_name.append(n[0] + n.split(' ')[1][0])
    merge_name = '_'.join(merge_name)  # e.g.Se_Kp_Pa
    fig, axs = plt.subplots(3, 3,figsize=(20,20))
    # handles, labels = axs.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center')
    # fig.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    fig.suptitle('Single-species VS Multi-species model', fontsize=32)


    n = 0
    for species in list_species:

        print('------------------------------',species)
        #1.
        # single_s_score = amr_utility.name_utility.GETname_multi_bench_save_name_final(species, None,
        #                                                                               level,learning,
        #                                                                                      epochs,
        #                                                                                      f_fixed_threshold,
        #                                                                                      f_nn_base,
        #                                                                                      f_optimize_score)
        # single_results = pd.read_csv(single_s_score + '_score_final_PLOT.txt', sep="\t", header=0, index_col=0)
        single_results = pd.DataFrame(columns=[ 'antibiotic','f1_macro', 'precision_macro', 'recall_macro', 'accuracy_macro',
                                            'mcc', 'f1_positive', 'f1_negative', 'precision_positive',
                                            'recall_positive', 'auc'])
        antibiotics, _, _ = amr_utility.load_data.extract_info(species, False, level)
        for anti in antibiotics:
            print(anti)
            save_name_score = amr_utility.name_utility.GETname_multi_bench_save_name_score(species, anti, level,
                                                                                           learning, epochs,
                                                                                           f_fixed_threshold,
                                                                                           f_nn_base,
                                                                                           f_optimize_score)
            if f_phylotree:
                score = pickle.load(open(save_name_score + '_all_score_Tree.pickle', "rb"))

            else:
                score = pickle.load(open(save_name_score + '_all_score.pickle', "rb"))

            aucs_test = score[4]
            score_report_test = score[3]
            mcc_test = score[2]
            thresholds_selected_test = score[0]


            summary = pd.DataFrame(columns=[ 'antibiotic','f1_macro', 'precision_macro', 'recall_macro', 'accuracy_macro',
                                            'mcc', 'f1_positive', 'f1_negative', 'precision_positive',
                                            'recall_positive', 'auc'])

            summary=analysis_results.extract_score.summary_allLoop(None,summary, cv,score_report_test, aucs_test, mcc_test,anti )

            single_results=single_results.append(summary, ignore_index=True)



        #2.-------------------

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

        summary_plot = pd.DataFrame(columns=[final_score, 'antibiotic', 'model'])

        for each_anti in antibiotics:
            # summary_plot_sub.loc[species, each_anti] = data_score.loc[each_anti, each_score]
            summary_plot_multi = pd.DataFrame(columns=[final_score, 'antibiotic', 'model'])
            summary_plot_multi.loc['e'] = [concat_results.loc[each_anti,final_score], each_anti, 'multi-species model']
            summary_plot = summary_plot.append(summary_plot_multi, ignore_index=True)
            #--------------------------------------
            # print(single_results)
            summary_plot_single=single_results.loc[single_results['antibiotic']==each_anti]
            summary_plot_single=summary_plot_single[[final_score,'antibiotic']]
            summary_plot_single['model']='single-speceis model'
            summary_plot = summary_plot.append(summary_plot_single, ignore_index=True)


        # print(summary_plot)
        # ax =sns.barplot(x="antibiotic", y=final_score, hue='model',
        #             data=summary_plot, dodge=True).set_title(species)
        # fig = ax.get_figure()
        # fig.savefig('log/results/' + str(level) + '/'+str(species)+'Compare_single_concat.png')
        # exit()
        # plot(species,n,axs,summary_plot_sub,final_score)

        row = (n // 3)
        col = n % 3
        g = sns.barplot(x="antibiotic", y=final_score, hue='model',
                        data=summary_plot, dodge=True, ax=axs[row, col])
        g.set(ylim=(0.2, 1.0))
        g.set_title(species,fontsize=20)
        # for item in g.get_xticklabels():
        #     item.set_rotation(45)
        g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
        # g.set_ylabel('')
        g.set_xlabel('')
        if n!=0:
            # handles, labels = g.get_legend_handles_labels()
            # g.legend('', '')
            g.get_legend().remove()

        else:
            handles, labels = g.get_legend_handles_labels()
            g.legend(bbox_to_anchor=(1.05,1.4), fontsize=16)

            #
            # plt.legend( handles[0:2], labels[0:2], bbox_to_anchor=(2.05, 0.1), loc=10, borderaxespad=0.,fontsize=20)
        n+=1
    # plt.xticks(rotation=45)

    fig.savefig('log/results/' + str(level) + '/Compare_single_concat.png')








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



