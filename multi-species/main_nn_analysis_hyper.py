import os
import numpy as np
import ast
import amr_utility.name_utility
import amr_utility.graph_utility
import amr_utility.file_utility
import argparse
import amr_utility.load_data
import analysis_results.extract_score
import pandas as pd

import pickle
import statistics
from scipy.stats import ttest_rel
import math
from collections import Counter
from pandas.plotting import table
import matplotlib.pyplot as plt



def make_visualization_pos(out_score,summary_all,species,antibiotics,level,f_fixed_threshold,epochs,learning,f_optimize_score,f_nn_base):

    # print(species)
    # # antibiotics_selected = ast.literal_eval(antibiotics)
    # print('====> Select_antibiotic:', len(antibiotics), antibiotics)
    final=pd.DataFrame(index=antibiotics, columns=['weighted-f1_macro', 'weighted-precision_macro', 'weighted-recall_macro', 'weighted-accuracy',
                                                   'weighted-mcc','weighted-f1_positive', 'weighted-f1_negative','weighted-precision_positive','weighted-recall_positive','weighted-auc','weighted-threshold','support','support_positive'] )

    final_plot=pd.DataFrame(index=antibiotics, columns=['weighted-f1_macro', 'weighted-precision_macro', 'weighted-recall_macro', 'weighted-accuracy',
                                                   'weighted-mcc','weighted-f1_positive', 'weighted-f1_negative','weighted-precision_positive','weighted-recall_positive','weighted-auc','weighted-threshold','support','support_positive'] )
    count = 0
    for anti in antibiotics:
        # save_name_score = amr_utility.name_utility.GETname_multi_bench_save_name_score(species, anti,level,learning,epochs,f_fixed_threshold,f_nn_base,f_optimize_score)
        # print(anti, '--------------------------------------------------------------------')
        # data = pd.read_csv(save_name_score+'_score.txt', sep="\t",index_col=0, header=0)
        # print(data)
        data=summary_all[count]
        count+=1
        data=data.loc[['weighted-mean','weighted-std'],:]
        data = data.astype(float).round(2)

        m= data.loc['weighted-mean',:].apply(lambda x: "{:.2f}".format(x))
        n=data.loc['weighted-std',:].apply(lambda x: "{:.2f}".format(x))

        final.loc[anti,:]=m.str.cat(n, sep='±').values

        final_plot.loc[anti,:]=data.loc['weighted-mean',:].to_list()
    if out_score=='f':
        final=final[['weighted-f1_macro','weighted-f1_positive', 'weighted-f1_negative','weighted-accuracy']]
        final_plot=final_plot[['weighted-f1_macro','weighted-f1_positive', 'weighted-f1_negative','weighted-accuracy']]
    elif out_score=='f_p_r':
        final = final[['weighted-f1_macro', 'weighted-precision_macro', 'weighted-recall_macro', 'weighted-accuracy']]
        final_plot = final_plot[['weighted-f1_macro', 'weighted-precision_macro', 'weighted-recall_macro', 'weighted-accuracy']]
    else:#all scores
        pass
    return final,final_plot

def make_visualization_pos_Tree(out_score,summary_all,species,antibiotics,level,f_fixed_threshold,epochs,learning,f_optimize_score,f_nn_base):

    # print(species)
    # # antibiotics_selected = ast.literal_eval(antibiotics)
    # print('====> Select_antibiotic:', len(antibiotics), antibiotics)
    final=pd.DataFrame(index=antibiotics, columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy',
                                                   'wmcc','f1_positive', 'f1_negative','precision_positive','recall_positive','auc','threshold','support','support_positive'] )
    final_plot=pd.DataFrame(index=antibiotics, columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy_macro',
                                                   'wmcc','f1_positive', 'f1_negative','precision_positive','recall_positive','auc','threshold','support','support_positive'] )

    count=0
    for anti in antibiotics:
        # save_name_score = amr_utility.name_utility.GETname_multi_bench_save_name_score(species, anti,level,learning,epochs,f_fixed_threshold,f_nn_base,f_optimize_score)
        # print(anti, '--------------------------------------------------------------------')
        # data = pd.read_csv(save_name_score+'_score.txt', sep="\t",index_col=0, header=0)
        # print(data)
        data=summary_all[count]
        count+=1
        data=data.loc[['mean','std'],:]
        data = data.astype(float).round(2)

        m= data.loc['mean',:].apply(lambda x: "{:.2f}".format(x))
        n=data.loc['std',:].apply(lambda x: "{:.2f}".format(x))

        final.loc[anti,:]=m.str.cat(n, sep='±').values
        final_plot.loc[anti, :] = data.loc['mean', :]
    if out_score=='f':
        final=final[['f1_macro','f1_positive', 'f1_negative','accuracy']]
        final_plot=final_plot[['f1_macro','f1_positive', 'f1_negative','accuracy']]
    elif out_score=='f_p_r':
        final = final[['f1_macro', 'precision_macro', 'recall_macro', 'accuracy']]
        final_plot = final_plot[['f1_macro', 'precision_macro', 'recall_macro', 'accuracy']]
    else:#all scores
        pass
    return final,final_plot

def multi_make_visualization_pos(out_score,merge_name,All_antibiotics,level,f_fixed_threshold, epochs,learning,f_optimize_score,
                             f_nn_base,cv,score,save_name_score,save_name_score_final):
    #only for multi-species,multi-output models. nested CV. No use till now. Aug 15,2021.
    #Only one table as output. i.e. all species share the same one score. June 23, finished.
    aucs_test = score[4]
    score_report_test = score[3]
    mcc_test = score[2]
    thresholds_selected_test = score[0]
    hyper_para=score[6]#one-element list
    hyper_para2 = score[7]
    hyper_para3 = score[8]
    # print('check',hyper_para)
    hy_para_all=[score[6][0],score[7][0],score[8][0]]
    final = pd.DataFrame(index=All_antibiotics,
                         columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy',
                                        'mcc', 'f1_positive', 'f1_negative','precision_positive', 'recall_positive', 'auc','threshold',
                                        'support', 'support_positive'])#todo 'f1_negative' check
    count_anti = 0
    # hy_para_all=[]
    for anti in All_antibiotics:


        summary = pd.DataFrame(index=['mean', 'std', 'weighted-mean', 'weighted-std'],
                               columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy_macro',
                                        'mcc', 'f1_positive', 'f1_negative','precision_positive', 'recall_positive','auc', 'threshold',
                                        'support', 'support_positive'])
        print('count_anti----------------------:',count_anti)
        summary=analysis_results.extract_score.score_summary_pos_normalCV(count_anti,summary,cv, score_report_test, aucs_test, mcc_test, save_name_score, thresholds_selected_test)



        count_anti+=1
        data = summary.loc[['weighted-mean', 'weighted-std'], :]
        # print(data)
        data = data.astype(float).round(2)
        # print(data)
        m = data.loc['weighted-mean', :].apply(lambda x: "{:.2f}".format(x))
        # print(m)
        n = data.loc['weighted-std', :].apply(lambda x: "{:.2f}".format(x))
        # print(data.dtypes)

        final.loc[anti, :] = m.str.cat(n, sep='±').values
        # hy_para_all.append([hyper_para[count_anti],hyper_para2[count_anti],hyper_para3[count_anti]])

    final['selected hyperparameter'] = [hy_para_all]*count_anti
    final.to_csv(save_name_score_final + '_score_final.txt', sep="\t")
    print(final)
def multi_make_visualization_pos_normalCV(out_score,merge_name,All_antibiotics,level,f_fixed_threshold, epochs,learning,f_optimize_score,
                             f_nn_base,cv,score,save_name_score,save_name_score_final):
    #only for multi-species,multi-output models. normal CV.
    #Only one table as output. i.e. all species share the same one score. June 23, finished.
    aucs_test = score[4]
    score_report_test = score[3]
    mcc_test = score[2]
    print('mcc_test shape: ',len(mcc_test),len(mcc_test[0]))
    thresholds_selected_test = score[0]
    # hyper_para=score[6]#one-element list
    # hyper_para2 = score[7]
    # hyper_para3 = score[8]
    # print('check',hyper_para)
    hy_para_all=[score[6][0],score[7][0],score[8][0]]
    final = pd.DataFrame(index=All_antibiotics,
                         columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy',
                                        'mcc', 'f1_positive', 'f1_negative','precision_positive', 'recall_positive', 'auc','threshold',
                                        'support', 'support_positive'])
    count_anti = 0
    # hy_para_all=[]
    for anti in All_antibiotics:


        summary = pd.DataFrame(index=['score'],
                               columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy_macro',
                                        'mcc', 'f1_positive', 'f1_negative','precision_positive', 'recall_positive','auc', 'threshold',
                                        'support', 'support_positive'])
        # print('count_anti----------------------:',count_anti)
        summary=analysis_results.extract_score.score_summary_pos_normalCV(count_anti,summary,cv, score_report_test, aucs_test, mcc_test, save_name_score, thresholds_selected_test)



        count_anti+=1

        '''
        #only for nested CV version.
        data = summary.loc[['weighted-mean', 'weighted-std'], :]
        # print(data)
        data = data.astype(float).round(2)
        # print(data)
        m = data.loc['weighted-mean', :].apply(lambda x: "{:.2f}".format(x))
        # print(m)
        n = data.loc['weighted-std', :].apply(lambda x: "{:.2f}".format(x))
        # print(data.dtypes)

        final.loc[anti, :] = m.str.cat(n, sep='±').values
        # hy_para_all.append([hyper_para[count_anti],hyper_para2[count_anti],hyper_para3[count_anti]])
        '''

        final.loc[anti, :] = summary.loc['score', :].to_list()

    if out_score=='f':
        final=final[['f1_macro','f1_positive', 'f1_negative','accuracy']]

    elif out_score=='f_p_r':
        final = final[['f1_macro', 'precision_macro', 'recall_macro', 'accuracy']]

    else:#all scores
        pass
    final['selected hyperparameter'] = [hy_para_all] * count_anti
    final.to_csv(save_name_score_final + '_score_final.txt', sep="\t")
    print(final)
def concat_multi_make_visualization_pos(out_score,merge_name, All_antibiotics, level, f_fixed_threshold, epochs, learning,
                                 f_optimize_score,f_nn_base, cv, score,save_name_score, save_name_score_final):
    # only for multi-s concat models
    aucs_test = score[4]
    score_report_test = score[3]
    mcc_test = score[2]
    thresholds_selected_test = score[0]

    final = pd.DataFrame(index=All_antibiotics,
                         columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy',
                                  'mcc', 'f1_positive', 'precision_positive', 'recall_positive', 'auc','threshold',
                                  'support', 'support_positive'])
    count=0

    for anti in All_antibiotics:

        # for i in np.arange(cv):
        if  len(All_antibiotics) > 1:
            report = score_report_test[count]  # multi-species model. should always be this
            # mcc = mcc_test[count]
            # auc=aucs_test[count]
            # thr=thresholds_selected_test
        else:
            report = score_report_test
        report = pd.DataFrame(report).transpose()
        # print(report)
        # print('--------')

        # if 'accuracy' not in report.index.to_list():  # no resitance pheno in test folder.  should not happen
        #     accuracy='-'
        # else:
        #     accuracy=report.loc['accuracy', 'f1-score']
        # print(report)
        if not report.empty:
            summary = pd.DataFrame(index=['mean', 'std', 'weighted-mean', 'weighted-std'],
                                   columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy_macro',
                                            'mcc', 'f1_positive', 'precision_positive', 'recall_positive', 'auc',
                                            'threshold',
                                            'support', 'support_positive'])

            summary = analysis_results.extract_score.score_summary_pos(count, summary, cv, score_report_test, aucs_test, mcc_test,
                                        save_name_score, thresholds_selected_test)

            count += 1
            data = summary.loc[['weighted-mean', 'weighted-std'], :]
            # print(data)
            data = data.astype(float).round(2)
            # print(data)
            m = data.loc['weighted-mean', :].apply(lambda x: "{:.2f}".format(x))
            # print(m)
            n = data.loc['weighted-std', :].apply(lambda x: "{:.2f}".format(x))
            # print(data.dtypes)

            final.loc[anti, :] = m.str.cat(n, sep='±').values
            # accuracy = report.loc['accuracy', 'f1-score']
            # f1=(report.loc['macro avg', 'f1-score'])
            # precision=(report.loc['macro avg', 'precision'])
            # recall=(report.loc['macro avg', 'recall'])
            # support=(report.loc['macro avg', 'support'])
            #
            # f1_pos=(report.loc['1', 'f1-score'])
            # precision_pos=(report.loc['1', 'precision'])
            # recall_pos=(report.loc['1', 'recall'])
            # support_pos=(report.loc['1', 'support'])
            #
            # final.loc[anti, :] = [f1, precision, recall, accuracy,
            #                                    mcc, f1_pos, precision_pos, recall_pos, auc,
            #                                    thr, support, support_pos]
            # final = final.astype(float).round(2)
            # hy_para_all.append([hyper_para[count], hyper_para2[count], hyper_para3[count]])
        count+=1
    final=final.fillna('-')
    if out_score=='f':
        final=final[['f1_macro','f1_positive', 'f1_negative','accuracy']]

    elif out_score=='f_p_r':
        final = final[['f1_macro', 'precision_macro', 'recall_macro', 'accuracy']]

    else:#all scores
        pass

    final.to_csv(save_name_score_final + '_score_final.txt', sep="\t")
    print(final)



def extract_info(out_score,f_multi,f_concat,f_concat2,f_all,T_test,T_dis_con,f_match_single,list_species,level,cv,hidden, epochs, re_epochs, learning,f_fixed_threshold,f_nn_base,f_phylotree,f_optimize_score,threshold_point,min_cov_point):
    if f_multi==True:
        if T_test == False and f_match_single==False:
            merge_name = []
            data = pd.read_csv('metadata/' + str(level) + '_multi-species_summary.csv', index_col=0,
                               dtype={'genome_id': object}, sep="\t")
            if f_all:
                list_species = data.index.tolist()[:-1]
                data = data.loc[list_species, :]
            else:
                # --------------------------------------------------------
                # drop columns(antibotics) all zero
                # data = data.loc[:, (data != 0).any(axis=0)]
                # drop columns(antibotics) less than 2 antibiotics
                data = data.loc[list_species, :]
                data = data.loc[:, (data.sum() > 1)]
                print(data)

            # --------------------------------------------------------
            # drop columns(antibotics) all zero
            data = data.loc[:, (data != 0).any(axis=0)]
            All_antibiotics = data.columns.tolist()  # all envolved antibiotics # todo
            for n in list_species:
                merge_name.append(n[0] + n.split(' ')[1][0])
            merge_name = '_'.join(merge_name)  # e.g.Se_Kp_Pa
            multi_results = 'log/results/' + str(level) + '/multi_species/' + merge_name
            amr_utility.file_utility.make_dir(multi_results)
            save_name_score = amr_utility.name_utility.GETname_multi_bench_save_name_score(merge_name, 'all_possible_anti', level,
                                                                                           learning, epochs,
                                                                                           f_fixed_threshold, f_nn_base,
                                                                                           f_optimize_score)
            save_name_score_final = amr_utility.name_utility.GETname_multi_bench_save_name_final(merge_name,
                                                                                                 'all_possible_anti', level,
                                                                                                 learning,
                                                                                                 epochs, f_fixed_threshold,
                                                                                                 f_nn_base,
                                                                                                 f_optimize_score)
            score = pickle.load(open(save_name_score + '_all_score.pickle', "rb"))
            amr_utility.file_utility.make_dir(amr_utility.file_utility.get_directory(save_name_score_final))

            multi_make_visualization_pos_normalCV(out_score,merge_name,All_antibiotics,level,f_fixed_threshold, epochs,learning,f_optimize_score,
                                         f_nn_base,cv,score,save_name_score,save_name_score_final)

            #June 23rd, seems finished.
        elif T_test == True and T_dis_con == True: #A paired T test between discrete vs concatenated m-s model.
            pass
        elif f_match_single==True:#match the single-species model results to the multi-s model table for a comparison.
            merge_name = []
            data = pd.read_csv('metadata/' + str(level) + '_multi-species_summary.csv', index_col=0,
                               dtype={'genome_id': object}, sep="\t")
            if f_all:
                list_species = data.index.tolist()[:-1]
                data = data.loc[list_species, :]
            else:
                data = data.loc[list_species, :]
                data = data.loc[:, (data.sum() > 1)]
                print(data)
            # drop columns(antibotics) all zero
            data = data.loc[:, (data != 0).any(axis=0)]
            All_antibiotics = data.columns.tolist()  # all envolved antibiotics # todo
            for n in list_species:
                merge_name.append(n[0] + n.split(' ')[1][0])
            merge_name = '_'.join(merge_name)  # e.g.Se_Kp_Pa
            # --------------------------------------------------------
            # --------------------------------------------------------
            save_name_score_final = amr_utility.name_utility.GETname_multi_bench_save_name_final(merge_name,
                                                                                                 'all_possible_anti',
                                                                                                 level,
                                                                                                 learning,
                                                                                                 epochs,
                                                                                                 f_fixed_threshold,
                                                                                                 f_nn_base,
                                                                                                 f_optimize_score)
            save_name_score_final=os.path.dirname(save_name_score_final)
            data = pd.read_csv('metadata/' + str(level) + '_multi-species_summary.csv', index_col=0,
                               dtype={'genome_id': object}, sep="\t")
            print(data)
            list_species = data.index.tolist()[:-1]
            data = data.loc[list_species, :]
            # All_antibiotics = data.columns.tolist()
            df_anti = data.dot(data.columns + ';').str.rstrip(';')#get anti names  marked with 1
            final_init=pd.DataFrame(index=list_species,columns=data.columns.tolist())
            for species in list_species:

                anti=df_anti[species].split(';')

                #read in resutls
                single_s_score = amr_utility.name_utility.GETname_multi_bench_save_name_final(species, None,
                                                                                                     level, learning,
                                                                                                     epochs,
                                                                                                     f_fixed_threshold,
                                                                                                     f_nn_base,
                                                                                                     f_optimize_score)
                data_score=pd.read_csv(single_s_score + '_score_final_PLOT.txt', sep="\t", header=0, index_col=0)
                print(data_score)

                # f1_macro, f1-positive, f1-negative, accuracy(f1_micro)
                if out_score == 'f':
                    score= ['weighted-f1_macro', 'weighted-f1_positive', 'weighted-f1_negative', 'weighted-accuracy']
                elif out_score == 'f_p_r':

                    score= ['weighted-f1_macro', 'weighted-precision_macro', 'weighted-recall_macro', 'weighted-accuracy']
                else:  # all scores
                    print('Please choose either f or f_p_r.')
                    exit()
                for each_score in score:
                    for each_anti in anti:
                        final_init.loc[species,each_anti]=data_score.loc[each_anti,each_score]

                    # final_init = final_init.replace(np.nan, '-', regex=True)
                    final_init.to_csv(save_name_score_final+'/single_species_'+each_score+'.txt', sep="\t")
                    print(final_init)




                # todo Aug 13.
        else:#A paired T test between threshold selected and fixed theshold.

            pass

    elif f_concat==True:# use all the species for a normal CV. quite the same as f_multi
        if T_test == False:

            merge_name = []

            data = pd.read_csv('metadata/' + str(level) + '_multi-species_summary.csv', index_col=0,
                               dtype={'genome_id': object}, sep="\t")

            if f_all:
                list_species = data.index.tolist()[:-1]
                data = data.loc[list_species, :]
            else:
                # --------------------------------------------------------
                # drop columns(antibotics) all zero
                # data = data.loc[:, (data != 0).any(axis=0)]
                # drop columns(antibotics) less than 2 antibiotics
                data = data.loc[list_species, :]
                data = data.loc[:, (data.sum() > 1)]
                print(data)

            # --------------------------------------------------------
            # drop columns(antibotics) all zero
            data = data.loc[:, (data != 0).any(axis=0)]
            All_antibiotics = data.columns.tolist()  # all envolved antibiotics # todo
            for n in list_species:
                merge_name.append(n[0] + n.split(' ')[1][0])
            merge_name = '_'.join(merge_name)  # e.g.Se_Kp_Pa
            multi_results = 'log/results/' + str(level) + '/multi_concat/' + merge_name

            # amr_utility.file_utility.make_dir(multi_results)

            save_name_score_concat = amr_utility.name_utility.GETname_multi_bench_save_name_score_concat(merge_name,
                                                                                                         merge_name,
                                                                                                         level,
                                                                                                         learning,
                                                                                                         epochs,
                                                                                                         f_fixed_threshold,
                                                                                                         f_nn_base,
                                                                                                         f_optimize_score,
                                                                                                         threshold_point,
                                                                                                         min_cov_point)

            save_name_score_final = amr_utility.name_utility.GETname_multi_bench_save_name_concat_final(merge_name,merge_name,
                                                                                                        level,learning,epochs,
                                                                                                        f_fixed_threshold,
                                                                                                        f_nn_base,
                                                                                                        f_optimize_score,
                                                                                                        threshold_point,
                                                                                                        min_cov_point)



            score = pickle.load(open(save_name_score_concat + '_all_score.pickle', "rb"))
            amr_utility.file_utility.make_dir(amr_utility.file_utility.get_directory(save_name_score_final))
            multi_make_visualization_pos_normalCV(out_score,merge_name, All_antibiotics, level, f_fixed_threshold, epochs, learning,
                                         f_optimize_score,
                                         f_nn_base, cv, score, save_name_score_concat, save_name_score_final)
            #todo need checking

        elif T_test == True and T_dis_con == True: #A paired T test between 2 sets of para: -t_p 0.6 -l_p 0.4 and  -t_p 0.8 -l_p 0.6
            pass


        else:# A paired T test between threshold selected and fixed theshold.
            pass

    elif f_concat2==True:#use one stand-out species for testing.
        if T_test == False:
            merge_name = []

            data = pd.read_csv('metadata/' + str(level) + '_multi-species_summary.csv', index_col=0,
                               dtype={'genome_id': object}, sep="\t")

            if f_all:
                list_species = data.index.tolist()[:-1]
                data = data.loc[list_species, :]
            else:
                # --------------------------------------------------------
                # drop columns(antibotics) all zero
                # data = data.loc[:, (data != 0).any(axis=0)]
                # drop columns(antibotics) less than 2 antibiotics
                data = data.loc[list_species, :]
                data = data.loc[:, (data.sum() > 1)]
                print(data)

            # --------------------------------------------------------
            # drop columns(antibotics) all zero
            data = data.loc[:, (data != 0).any(axis=0)]
            All_antibiotics = data.columns.tolist()  # all envolved antibiotics # todo
            for n in list_species:
                merge_name.append(n[0] + n.split(' ')[1][0])
            merge_name = '_'.join(merge_name)  # e.g.Se_Kp_Pa
            multi_results = 'log/results/' + str(level) + '/multi_concat/' + merge_name

            # amr_utility.file_utility.make_dir(multi_results)

            count = 0
            for species_testing in list_species:
                list_species_training = list_species[:count] + list_species[count + 1:]
                count += 1
                # do a nested CV on list_species, select the best estimator for testing on the standing out species
                merge_name_train = []
                for n in list_species_training:
                    merge_name_train.append(n[0] + n.split(' ')[1][0])
                merge_name_train = '_'.join(merge_name_train)  # e.g.Se_Kp_Pa
                merge_name_test = species_testing.replace(" ", "_")
                # 1. testing on the left-out species scores
                save_name_score = amr_utility.name_utility.GETname_multi_bench_save_name_score_concat(merge_name,
                                                                                                      merge_name_test,
                                                                                                      level, learning,
                                                                                                      epochs,
                                                                                                      f_fixed_threshold,
                                                                                                      f_nn_base,
                                                                                                      f_optimize_score,
                                                                                                      threshold_point,
                                                                                                      min_cov_point)
                save_name_score_final = amr_utility.name_utility.GETname_multi_bench_save_name_concat_final(merge_name,
                                                                                                            merge_name_test,
                                                                                                            level,
                                                                                                            learning,
                                                                                                            epochs,
                                                                                                            f_fixed_threshold,
                                                                                                            f_nn_base,
                                                                                                            f_optimize_score,
                                                                                                            threshold_point,
                                                                                                            min_cov_point)

                amr_utility.file_utility.make_dir(os.path.dirname(save_name_score_final))
                score = pickle.load(open(save_name_score + '_TEST.pickle', "rb"))
                # aucs_test = score[4]
                # score_report_test = score[3]
                # mcc_test = score[2]
                # thresholds_selected_test = score[0]

                concat_multi_make_visualization_pos(out_score,merge_name, All_antibiotics, level, f_fixed_threshold, epochs,
                                                    learning,
                                                    f_optimize_score, f_nn_base, cv, score, save_name_score,
                                                    save_name_score_final)

                # 2. CV scores on the training species
                save_name_score_concat = amr_utility.name_utility.GETname_multi_bench_save_name_score_concat(merge_name,
                                                                                                             merge_name_train,
                                                                                                             level,
                                                                                                             learning,
                                                                                                             epochs,
                                                                                                             f_fixed_threshold,
                                                                                                             f_nn_base,
                                                                                                             f_optimize_score,
                                                                                                             threshold_point,
                                                                                                             min_cov_point)
                save_name_score_final = amr_utility.name_utility.GETname_multi_bench_save_name_concat_final(merge_name,
                                                                                                            merge_name_train,
                                                                                                            level,
                                                                                                            learning,
                                                                                                            epochs,
                                                                                                            f_fixed_threshold,
                                                                                                            f_nn_base,
                                                                                                            f_optimize_score,
                                                                                                            threshold_point,
                                                                                                            min_cov_point)

                score = pickle.load(open(save_name_score_concat + '_all_score.pickle', "rb"))
                # aucs_test = score[4]
                # score_report_test = score[3]
                # mcc_test = score[2]
                # thresholds_selected_test = score[0]

                multi_make_visualization_pos_normalCV(out_score,merge_name_train, All_antibiotics, level, f_fixed_threshold, epochs,
                                             learning,
                                             f_optimize_score,
                                             f_nn_base, cv, score, save_name_score, save_name_score_final)




        else:
            pass



    else:#single-species model.
        # June 22nd. for hyper-para selection version
        #June 22nd, So far the s-m output's auc score is not correct calculated, although in the codees it is right.
        # July 2nd. should be finished all.
        if T_test==False:
            data = pd.read_csv('metadata/'+str(level)+'_Species_antibiotic_FineQuality.csv', index_col=0, dtype={'genome_id': object},
                               sep="\t")
            data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
            if f_all:
                list_species = data.index.tolist()
                data = data.loc[list_species, :]
            else:
                data = data.loc[list_species, :]
            df_species = data.index.tolist()
            # antibiotics = data['modelling antibiotics'].tolist()
            print(data)

            for species in df_species:
                amr_utility.file_utility.make_dir('log/results/'+str(level)+'/'+ str(species.replace(" ", "_")))
                antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
                summary_all=[]

                #only hyper-parameter selection mode, otherwaise use the main_nn_analysis.py
                hy_para_all=[]
                hy_para_fre=[]
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

                    hy_para_all.append([score[6],score[7],score[8]])#1*n_cv
                    #vote the most frequent used hyper-para
                    hy_para_collection=score[6]#10 dimension. each reapresents one outer loop.
                    df = pd.DataFrame(hy_para_collection)
                    df_count=df.groupby(df.columns.tolist(), as_index=False).size()
                    df_count.rename(columns={'size': 'frequency'}, inplace=True)
                    ind=df_count['frequency'].argmax()
                    common=df_count.loc[ind]

                    hy_para_fre.append(common.to_dict())

                    summary = pd.DataFrame(index=['mean', 'std', 'weighted-mean', 'weighted-std'],
                                           columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy_macro',
                                                    'mcc', 'f1_positive','f1_negative', 'precision_positive', 'recall_positive', 'auc',
                                                    'threshold', 'support', 'support_positive'])
                    if f_phylotree:
                        summary = analysis_results.extract_score.score_summary_pos_Tree(None, summary, cv, score_report_test, aucs_test, mcc_test,
                                                         save_name_score,
                                                         thresholds_selected_test)
                    else:
                        summary = analysis_results.extract_score.score_summary_pos(None, summary, cv, score_report_test, aucs_test, mcc_test, save_name_score,
                                                thresholds_selected_test)
                    summary_all.append(summary)
                # put out final table with scores:'f1-score','precision', 'recall','accuracy'
                # make_visualization(species, antibiotics,level,f_fixed_threshold, epochs,learning,f_optimize_score)
                if f_phylotree:
                    final, final_plot = make_visualization_pos_Tree(out_score,summary_all, species, antibiotics, level,
                                                                    f_fixed_threshold, epochs,
                                                                    learning,
                                                                    f_optimize_score,
                                                                    f_nn_base)  # with scores only for positive class
                    save_name_score_final = amr_utility.name_utility.GETname_multi_bench_save_name_final(species, None,
                                                                                                         level,
                                                                                                         learning,
                                                                                                         epochs,
                                                                                                         f_fixed_threshold,
                                                                                                         f_nn_base,
                                                                                                         f_optimize_score)
                    final.to_csv(save_name_score_final + '_score_final_Tree.txt', sep="\t")
                    final_plot.to_csv(save_name_score_final + '_score_final_Tree_PLOT.txt', sep="\t")
                    final['the most frequent hyperparameter'] = hy_para_fre
                    final['selected hyperparameter'] = hy_para_all  # add hyperparameter information. Each antibiotic has 10 hyper-para, each for one outer loop.
                    print(final)

                else:
                    final,final_plot=make_visualization_pos(out_score,summary_all, species, antibiotics, level, f_fixed_threshold, epochs, learning,
                                           f_optimize_score, f_nn_base)  # with scores only for positive class
                    save_name_score_final = amr_utility.name_utility.GETname_multi_bench_save_name_final(species, None,
                                                                                                         level, learning,
                                                                                                         epochs,
                                                                                                         f_fixed_threshold,
                                                                                                         f_nn_base,
                                                                                                         f_optimize_score)


                    final['the most frequent hyperparameter'] = hy_para_fre
                    final['selected hyperparameter'] = hy_para_all  # add hyperparameter information. Each antibiotic has 10 hyper-para, each for one outer loop.

                    final.to_csv(save_name_score_final + '_score_final.txt', sep="\t")
                    final_plot.to_csv(save_name_score_final + '_score_final_PLOT.txt', sep="\t")
                    print(final)

        # 3 paired T tests of f1_macro between fixed-threshold and threshold selection and auc based inner loop best estimator selection
        else:

            data = pd.read_csv('metadata/' + str(level) + '_Species_antibiotic_FineQuality.csv', index_col=0,
                               dtype={'genome_id': object},
                               sep="\t")
            data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
            data = data.loc[list_species, :]
            df_species = data.index.tolist()
            # antibiotics = data['modelling antibiotics'].tolist()
            print(data)

            for species in df_species:
                amr_utility.file_utility.make_dir(
                    'log/results/' + str(level) + '/' + str(species.replace(" ", "_")))
                antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
                summary_all = []

                # only hyper-parameter selection mode, otherwaise use the main_nn_analysis.py
                fix_select = []
                fix_auc = []
                auc_select=[]
                for anti in antibiotics:
                # for anti in ['ceftazidime','ciprofloxacin','levofloxacin','meropenem']:#todo change back to normal
                    # 1. print('T tests of f1_macro between f1 fixed-threshold and threshold selection \n ----------------')
                    save_name_score_f1_fix = amr_utility.name_utility.GETname_multi_bench_save_name_score(species,
                                                                                                   anti, level,
                                                                                                   0.0,
                                                                                                   0,
                                                                                                   True,
                                                                                                   False,
                                                                                                   'f1_macro')

                    save_name_score_f1_selection = amr_utility.name_utility.GETname_multi_bench_save_name_score(
                        species,
                        anti, level,
                        0.0,
                        0,
                        False,
                        False,
                        'f1_macro')

                    score_fix = pickle.load(open(save_name_score_f1_fix + '_all_score.pickle', "rb"))
                    score_report_test_fix = score_fix[3]

                    score_select = pickle.load(open(save_name_score_f1_selection + '_all_score.pickle', "rb"))
                    score_report_test_select = score_select[3]
                    f1_macro_fix=[]
                    f1_macro_select=[]
                    f1_macro_fix_pos=[]
                    f1_macro_select_pos=[]
                    for i in np.arange(cv):
                        report_fix = score_report_test_fix[i]
                        report_select = score_report_test_select[i]
                        f1_macro_fix_sub=pd.DataFrame(report_fix).transpose().loc['macro avg','f1-score']
                        f1_macro_select_sub = pd.DataFrame(report_select).transpose().loc['macro avg','f1-score']
                        f1_macro_fix_pos_sub = pd.DataFrame(report_fix).transpose().loc['1', 'f1-score']
                        f1_macro_select_pos_sub = pd.DataFrame(report_select).transpose().loc['1', 'f1-score']
                        f1_macro_fix.append(f1_macro_fix_sub)
                        f1_macro_select.append(f1_macro_select_sub)
                        f1_macro_fix_pos.append(f1_macro_fix_pos_sub)
                        f1_macro_select_pos.append(f1_macro_select_pos_sub)



                    result=ttest_rel(f1_macro_fix, f1_macro_select)
                    result_pos=ttest_rel(f1_macro_fix_pos, f1_macro_select_pos)
                    pvalue=result[1]
                    pvalue_pos=result_pos[1]
                    # print(pvalue)
                    # print(pvalue_pos)
                    fix_select.append(pvalue)

                    # 2. print('T tests of f1_macro between f1 fixed-threshold and auc based inner loop best estimator selection')
                    save_name_score_f1_fix = amr_utility.name_utility.GETname_multi_bench_save_name_score(species,
                                                                                                          anti, level,
                                                                                                          0.0,
                                                                                                          0,
                                                                                                          True,
                                                                                                          False,
                                                                                                          'f1_macro')

                    save_name_score_auc = amr_utility.name_utility.GETname_multi_bench_save_name_score(
                        species,
                        anti, level,
                        0.0,
                        0,
                        False,
                        False,
                        'auc')
                    score_fix = pickle.load(open(save_name_score_f1_fix + '_all_score.pickle', "rb"))
                    score_report_test_fix = score_fix[3]

                    score_auc = pickle.load(open(save_name_score_auc + '_all_score.pickle', "rb"))
                    score_report_test_auc = score_auc[3]
                    f1_macro_fix = []
                    f1_macro_auc = []
                    f1_macro_fix_pos = []
                    f1_macro_auc_pos = []
                    for i in np.arange(cv):
                        report_fix = score_report_test_fix[i]
                        report_auc = score_report_test_auc[i]
                        f1_macro_fix_sub = pd.DataFrame(report_fix).transpose().loc['macro avg', 'f1-score']
                        f1_macro_auc_sub = pd.DataFrame(report_auc).transpose().loc['macro avg', 'f1-score']
                        f1_macro_fix_pos_sub = pd.DataFrame(report_fix).transpose().loc['1', 'f1-score']
                        f1_macro_auc_pos_sub = pd.DataFrame(report_auc).transpose().loc['1', 'f1-score']
                        f1_macro_fix.append(f1_macro_fix_sub)
                        f1_macro_auc.append(f1_macro_auc_sub)
                        f1_macro_fix_pos.append(f1_macro_fix_pos_sub)
                        f1_macro_auc_pos.append(f1_macro_auc_pos_sub)

                    result = ttest_rel(f1_macro_fix, f1_macro_auc)
                    result_pos = ttest_rel(f1_macro_fix_pos, f1_macro_auc_pos)
                    pvalue = result[1]
                    pvalue_pos = result_pos[1]
                    # print(pvalue)
                    # print(pvalue_pos)
                    fix_auc.append(pvalue)





                    # 3. print('T tests of f1_macro between f1 threshold selection and auc based inner loop best estimator selection')
                    save_name_score_auc = amr_utility.name_utility.GETname_multi_bench_save_name_score(species,
                                                                                                          anti, level,
                                                                                                          0.0,
                                                                                                          0,
                                                                                                          True,
                                                                                                          False,
                                                                                                          'auc')

                    save_name_score_f1_selection = amr_utility.name_utility.GETname_multi_bench_save_name_score(
                        species,
                        anti, level,
                        0.0,
                        0,
                        False,
                        False,
                        'f1_macro')
                    score_auc = pickle.load(open(save_name_score_auc + '_all_score.pickle', "rb"))
                    score_report_test_auc = score_auc[3]

                    score_select = pickle.load(open(save_name_score_f1_selection + '_all_score.pickle', "rb"))
                    score_report_test_select = score_select[3]
                    f1_macro_auc = []
                    f1_macro_select = []
                    f1_macro_auc_pos = []
                    f1_macro_select_pos = []
                    for i in np.arange(cv):
                        report_auc = score_report_test_auc[i]
                        report_select = score_report_test_select[i]
                        f1_macro_auc_sub = pd.DataFrame(report_auc).transpose().loc['macro avg', 'f1-score']
                        f1_macro_select_sub = pd.DataFrame(report_select).transpose().loc['macro avg', 'f1-score']
                        f1_macro_auc_pos_sub = pd.DataFrame(report_auc).transpose().loc['1', 'f1-score']
                        f1_macro_select_pos_sub = pd.DataFrame(report_select).transpose().loc['1', 'f1-score']
                        f1_macro_auc.append(f1_macro_auc_sub)
                        f1_macro_select.append(f1_macro_select_sub)
                        f1_macro_auc_pos.append(f1_macro_auc_pos_sub)
                        f1_macro_select_pos.append(f1_macro_select_pos_sub)

                    result = ttest_rel(f1_macro_auc, f1_macro_select)
                    result_pos = ttest_rel(f1_macro_auc_pos, f1_macro_select_pos)
                    pvalue = result[1]
                    pvalue_pos = result_pos[1]
                    # print(pvalue)
                    # print(pvalue_pos)
                    auc_select.append(pvalue)


                #--------
                print(fix_select)
                print(fix_auc)
                print(auc_select)
                # save as dataframe
                save_name_score_final = amr_utility.name_utility.GETname_multi_bench_save_name_final(species, None,
                                                                                                     level, learning,
                                                                                                     epochs,
                                                                                                     f_fixed_threshold,
                                                                                                     f_nn_base,
                                                                                                     f_optimize_score)
                # antibiotics= ['ceftazidime','ciprofloxacin','levofloxacin','meropenem']#todo delete this later
                data_p=np.array([fix_select,fix_auc,auc_select])
                final_p=pd.DataFrame(data=data_p.T,index=antibiotics, columns=['fixed threshold VS threshold selection', 'fixed threshold VS AUC','threshold selection VS AUC'])
                final_p.to_csv(save_name_score_final + '_Ttest.txt', sep="\t")




if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t_p', '--threshold_point', default=0.8, type=float,
                        help='Threshold for identity of Pointfinder. ')
    parser.add_argument('-l_p', '--min_cov_point', default=0.6, type=float,
                        help=' Minimum (breadth-of) coverage of Pointfinder. ')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')

    parser.add_argument("-f_multi", "--f_multi",  dest='f_multi', action='store_true',
                        help='flag for multi-species model')
    parser.add_argument("-f_concat", "--f_concat", dest='f_concat', action='store_true',
                        help='flag for multi-species concatenated model,all species model')
    parser.add_argument("-f_concat2", "--f_concat2", dest='f_concat2', action='store_true',
                        help='flag for multi-species concatenated model, testing on a stand-out species')
    parser.add_argument("-T", "--T_test", dest='T_test', action='store_true',
                        help='flag for T test between threshold selected and fixed theshold and auc based selection.')
    parser.add_argument("-T_dis_con", "--T_dis_con", dest='T_dis_con', action='store_true',
                        help='flag for T test between discrete vs concatenated m-s mode.')
    parser.add_argument("-f_match_single", "--f_match_single", dest='f_match_single', action='store_true',
                        help='flag for match single-species model results to multi-species model results for a comparison.')
    parser.add_argument('-out_score', '--out_score', default='f', type=str,
                        help='Scores of the final output table. f:f_macro,f_pos,f_neg,f_micro. all:all scores. f_p_r:f1_macro,precision,recall,accuracy')
    parser.add_argument("-cv", "--cv_number", default=10, type=int,
                        help='CV splits number')
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
    parser.add_argument('-f_fixed_threshold', '--f_fixed_threshold', dest='f_fixed_threshold', action='store_true',
                        help='set a fixed threshod:0.5.')
    parser.add_argument('-f_optimize_score', '--f_optimize_score', default='f1_macro', type=str,
                        help='the optimizing-score for choosing the best estimator in inner loop. Choose: auc or f1_macro.')
    parser.add_argument('-f_nn_base', '--f_nn_base', dest='f_nn_base', action='store_true',
                        help='benchmarking baseline.')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    # parser.add_argument("-o","--output", default=None, type=str, required=True,
	# 					help='Output file names')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
            \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
            \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    # parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parsedArgs = parser.parse_args()
    # parser.print_help()
    # print(parsedArgs)
    extract_info(parsedArgs.out_score,parsedArgs.f_multi,parsedArgs.f_concat,parsedArgs.f_concat2,parsedArgs.f_all,parsedArgs.T_test,parsedArgs.T_dis_con,parsedArgs.f_match_single,parsedArgs.species,parsedArgs.level,parsedArgs.cv_number,parsedArgs.hidden,parsedArgs.epochs,
                 parsedArgs.re_epochs,parsedArgs.learning,parsedArgs.f_fixed_threshold,parsedArgs.f_nn_base,parsedArgs.f_phylotree,parsedArgs.f_optimize_score,parsedArgs.threshold_point,parsedArgs.min_cov_point)



# def make_visualization(species,antibiotics,level,f_fixed_threshold,epochs,learning,f_optimize_score,f_nn_base):
#
#     print(species)
#     # antibiotics_selected = ast.literal_eval(antibiotics)
#     print('====> Select_antibiotic:', len(antibiotics), antibiotics)
#     final=pd.DataFrame(index=antibiotics, columns=['weighted-f1_macro', 'weighted-precision_macro', 'weighted-recall_macro', 'weighted-accuracy_macro',
#                                                           'weighted-auc','weighted-mcc','weighted-threshold'] )
#     # print(final)
#     for anti in antibiotics:
#         save_name_score = amr_utility.name_utility.GETname_multi_bench_save_name_score(species,anti,level,learning,epochs,f_fixed_threshold,f_nn_base,f_optimize_score)
#
#         # print(anti, '--------------------------------------------------------------------')
#
#         data = pd.read_csv(save_name_score+'_score.txt', sep="\t",index_col=0, header=0)
#         # print(data)
#         data=data.loc[['weighted-mean','weighted-std'],:]
#         # print(data)
#         data = data.astype(float).round(2)
#         # print(data)
#         m= data.loc['weighted-mean',:].apply(lambda x: "{:.2f}".format(x))
#         # print(m)
#         n=data.loc['weighted-std',:].apply(lambda x: "{:.2f}".format(x))
#         # print(data.dtypes)
#
#         final.loc[anti,:]=m.str.cat(n, sep='±').values
#
#     print(final)
#     # None means not multi-species results.
#     save_name_score_final=amr_utility.name_utility.GETname_multi_bench_save_name_final(species,None, level,learning,epochs,f_fixed_threshold,f_nn_base,f_optimize_score)
#     # final=final.astype(float).round(2)
#     final.to_csv(save_name_score_final+'_score_final.txt', sep="\t")

# def score_summary(summary,cv,score_report_test,aucs_test,mcc_test,save_name_score,thresholds_selected_test):
#
#
#
#     #
#     f1=[]
#     precision=[]
#     recall=[]
#     accuracy=[]
#     support=[]
#     f_noAccu = []
#     for i in np.arange(cv):
#         report=score_report_test[i]
#         report=pd.DataFrame(report).transpose()
#         print(report)
#         print('--------')
#
#         if 'accuracy' not in report.index.to_list():# no resitance pheno in test folder
#             accuracy.append('-')
#             f_noAccu.append(i)
#         else:
#             accuracy.append(report.loc['accuracy', 'f1-score'])
#
#         f1.append(report.loc['macro avg','f1-score'])
#         precision.append(report.loc['macro avg','precision'])
#         recall.append(report.loc['macro avg','recall'])
#         support.append(report.loc['macro avg','support'])
#
#
#
#     if f_noAccu != []:
#         #rm the iteration's results, where no resistance phenotype in the test folder.
#         f1 = [i for j, i in enumerate(f1) if j not in f_noAccu]
#         precision = [i for j, i in enumerate(precision) if j not in f_noAccu]
#         recall = [i for j, i in enumerate(recall) if j not in f_noAccu]
#         accuracy = [i for j, i in enumerate(accuracy) if j not in f_noAccu]
#         support = [i for j, i in enumerate(support) if j not in f_noAccu]
#         mcc_test = [i for j, i in enumerate(mcc_test) if j not in f_noAccu]
#         aucs_test = [i for j, i in enumerate(aucs_test) if j not in f_noAccu]
#         thresholds_selected_test = [i for j, i in enumerate(thresholds_selected_test) if j not in f_noAccu]
#
#
#     summary.loc['mean','accuracy_macro'] = statistics.mean(accuracy)
#     summary.loc['std','accuracy_macro'] = statistics.stdev(accuracy)
#     summary.loc['mean', 'f1_macro'] = statistics.mean(f1)
#     summary.loc['std', 'f1_macro'] = statistics.stdev(f1)
#     summary.loc['mean', 'precision_macro'] = statistics.mean(precision)
#     summary.loc['std', 'precision_macro'] = statistics.stdev(precision)
#     summary.loc['mean', 'recall_macro'] = statistics.mean(recall)
#     summary.loc['std', 'recall_macro'] = statistics.stdev(recall)
#     summary.loc['mean', 'auc'] = statistics.mean(aucs_test)
#     summary.loc['std', 'auc'] = statistics.stdev(aucs_test)
#     summary.loc['mean', 'mcc'] = statistics.mean(mcc_test)
#     summary.loc['std', 'mcc'] = statistics.stdev(mcc_test)
#     summary.loc['mean', 'threshold'] = statistics.mean(thresholds_selected_test)
#     summary.loc['std', 'threshold'] = statistics.stdev(thresholds_selected_test)
#     f1_average = np.average(f1, weights=support)
#     precision_average = np.average(precision, weights=support)
#     recall_average = np.average(recall, weights=support)
#     aucs_average = np.average(aucs_test, weights=support)
#     mcc_average = np.average(mcc_test, weights=support)
#     thr_average = np.average(thresholds_selected_test, weights=support)
#     accuracy_average = np.average(accuracy, weights=support)
#     summary.loc['weighted-mean', :] = [f1_average, precision_average, recall_average, accuracy_average,
#                                        aucs_average, mcc_average, thr_average]
#     summary.loc['weighted-std', :] = [math.sqrt(weithgted_var(f1, f1_average, support)),
#                                       math.sqrt(weithgted_var(precision, precision_average, support)),
#                                       math.sqrt(weithgted_var(recall, recall_average, support)),
#                                       math.sqrt(weithgted_var(accuracy, accuracy_average, support)),
#                                       math.sqrt(weithgted_var(aucs_test, aucs_average, support)),
#                                       math.sqrt(weithgted_var(mcc_test, mcc_average, support)),
#                                       math.sqrt(weithgted_var(thresholds_selected_test, thr_average, support))]
#
#
#
#     summary.to_csv(save_name_score+'_score.txt', sep="\t")
#     print(summary)
