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

#5 folds + 1 test fold
#plot a comparative graph of single-s model and discrete multiple-s model, concatenated mixed species model. concat multi-s model no left out species version.
#

#get results of each species from discrete multiple-s model, concatenated mixed species model
# python main_nn_analysis_hyper.py -f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -f_multi -f_all -split_species
# python main_nn_analysis_hyper.py -f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -f_concat -f_all -split_species
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

    fig, axs = plt.subplots(2, 5,figsize=(30,20), gridspec_kw={'width_ratios': [1.2,1, 2,2,1.5]})#
    plt.tight_layout()
    fig.subplots_adjust(left=0.04,  right=0.98,wspace=0.1, hspace=0.3, top=0.8, bottom=0.08)
    gs = axs[1, 0].get_gridspec()
    # remove the underlying axes
    for ax in axs[1, :2]:
        ax.remove()
    axbig = fig.add_subplot(gs[1, :2])

    # handles, labels = axs.get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center')
    # fig.tight_layout()
    # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    # fig.suptitle('Single-species VS Multi-species model', fontsize=32)


    n = 0
    for species in list_species:

        print('------------------------------',species)
        #1. single-s model
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

            f1_test= score[1]
            aucs_test = score[4]
            score_report_test = score[3]
            mcc_test = score[2]
            thresholds_selected_test = score[0]


            summary = pd.DataFrame(columns=[ 'antibiotic','f1_macro', 'precision_macro', 'recall_macro', 'accuracy_macro',
                                            'mcc', 'f1_positive', 'f1_negative', 'precision_positive',
                                            'recall_positive', 'auc'])

            summary=analysis_results.extract_score.summary_allLoop(None,summary, cv,score_report_test, f1_test,aucs_test, mcc_test,anti )

            single_results=single_results.append(summary, ignore_index=True)



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

        # 3 . -------------------multi-s concateM model
        save_name_score_final = amr_utility.name_utility.GETname_multi_bench_save_name_concatM_final(merge_name,merge_name,
                                                                                                        level,learning,epochs,
                                                                                                        f_fixed_threshold,
                                                                                                        f_nn_base,
                                                                                                        f_optimize_score,
                                                                                                        threshold_point,
                                                                                                        min_cov_point)

        concatM_results=pd.read_csv(save_name_score_final+'/split_concate_model.txt', index_col=0,sep="\t")


        # 4. ---------discrete model.
        save_name_score_final = amr_utility.name_utility.GETname_multi_bench_save_name_final('f1_macro',merge_name,
                                                                                                 'all_possible_anti', level,
                                                                                                 learning,
                                                                                                 epochs, f_fixed_threshold,
                                                                                                 f_nn_base,
                                                                                                 f_optimize_score)
        dis_results=pd.read_csv(save_name_score_final+'/split_discrete_model.txt',  index_col=0,sep="\t")









        #-------------------------------
        #Prepare dataframe for plotting.
        #-------------------------------



        antibiotics = df_anti[species].split(';')

        summary_plot = pd.DataFrame(columns=[final_score, 'antibiotic', 'model'])

        for each_anti in antibiotics:

            # print(single_results)
            summary_plot_single=single_results.loc[single_results['antibiotic']==each_anti]
            summary_plot_single=summary_plot_single[[final_score,'antibiotic']]
            summary_plot_single['model']='single-species-antibiotic model'
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







        with open('../src/AntiAcronym_dict.pkl', 'rb') as f:
            map_acr = pickle.load(f)
        # spoke_labels= [map_acr[x] for x in spoke_labels]
        summary_plot['antibiotic_acr']=summary_plot['antibiotic'].apply(lambda x: map_acr[x])
        # print(summary_plot)
        # print(summary_plot)
        # ax =sns.barplot(x="antibiotic", y=final_score, hue='model',
        #             data=summary_plot, dodge=True).set_title(species)
        # fig = ax.get_figure()
        # fig.savefig('log/results/' + str(level) + '/'+str(species)+'Compare_single_concat.png')
        # exit()
        # plot(species,n,axs,summary_plot_sub,final_score)




        # color_selection=sns.color_palette()
        # color_selection=color_selection[:2]
        color_selection=['#a6611a','#dfc27d','#80cdc1','#018571']
        palette = iter(color_selection)
        row = (n //5)
        col = n % 5+1
        # print(df_anti[species])
        # print([row, col])
        species_title=(species[0] +". "+ species.split(' ')[1] )
        if species in ['Mycobacterium tuberculosis']:

            ax_ = plt.subplot(251)
            g = sns.barplot(x="antibiotic_acr", y=final_score, hue='model',
                        data=summary_plot, dodge=True, ax=ax_,palette=palette)
            n+=1
            g.set_ylabel(final_score, fontsize=25)

            g.set_title(species_title,fontsize=31,style='italic', weight='bold',pad=10)
            g.tick_params(axis='y', which='major', labelsize=25)
        elif species in ['Campylobacter jejuni']:

            ax_ = plt.subplot(252)
            g = sns.barplot(x="antibiotic_acr", y=final_score, hue='model',
                        data=summary_plot, dodge=True, ax=ax_,palette=palette)

            g.set_yticklabels([])
            g.set_yticks([])
            g.set_title(species_title,fontsize=31,style='italic', weight='bold',pad=10 )
            g.set(ylabel=None)
            n+=1
        elif species in ['Salmonella enterica','Streptococcus pneumoniae','Escherichia coli']:
            n+=1
            num=250+n
            ax_= plt.subplot(num)
            g = sns.barplot(x="antibiotic_acr", y=final_score, hue='model',
                        data=summary_plot, dodge=True, ax=ax_,palette=palette)#ax=axs[row, col]
            g.set_yticklabels([])
            g.set_yticks([])
            g.set_title(species_title,fontsize=31,style='italic', weight='bold',pad=10 )
            g.set(ylabel=None)
        elif species =='Klebsiella pneumoniae':
            g = sns.barplot(x="antibiotic_acr", y=final_score, hue='model',
                        data=summary_plot, dodge=True, ax=axbig,palette=palette)#ax=axs[row, col]
            n+=1
            g.set_ylabel(final_score, fontsize=25)
            g.set_title(species_title,fontsize=31,style='italic', weight='bold' ,pad=10)
            g.tick_params(axis='y', which='major', labelsize=25)
        else:

            # num=240+n
            # ax_= plt.subplot(num)
            print(n)
            print([row, col])
            g = sns.barplot(x="antibiotic_acr", y=final_score, hue='model',
                    data=summary_plot, dodge=True, ax=axs[row, col],palette=palette)#ax=axs[row, col]
            n+=1
            g.set_yticklabels([])
            g.set_yticks([])
            g.set_title(species_title,fontsize=29,style='italic', weight='bold',pad=10 )
            g.set(ylabel=None)
        g.set(ylim=(0, 1.0))

        # g.set_ylabel(final_score,size = 16)
        # for item in g.get_xticklabels():
        #     item.set_rotation(45)

        labels_p = [item.get_text() for item in g.get_xticklabels()]
        for i_anti in labels_p:
            if '/' in i_anti:
                posi=i_anti.find('/')
                _i_anti=i_anti[:(posi+1)] + '\n' + i_anti[(posi+1):]
                labels_p=[_i_anti if x==i_anti else x for x in labels_p]

        g.set_xticklabels(labels_p, size=32, rotation=40, horizontalalignment='right')
        # g.set_ylabel('')
        g.set_xlabel('')
        if n!=1:
            # handles, labels = g.get_legend_handles_labels()
            # g.legend('', '')
            g.get_legend().remove()

        else:
            handles, labels = g.get_legend_handles_labels()
            g.legend(bbox_to_anchor=(5,1.7), fontsize=35, ncol=1,frameon=False)

            #
            # plt.legend( handles[0:2], labels[0:2], bbox_to_anchor=(2.05, 0.1), loc=10, borderaxespad=0.,fontsize=20)

    # plt.xticks(rotation=45)

    fig.savefig('log/results/' + str(level) + '/Compare_single_dis_concatM_concat.pdf')
    fig.savefig('log/results/' + str(level) + '/Compare_single_dis_concatM_concat.png')











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


