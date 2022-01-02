import os
import numpy as np
import pandas as pd
import statistics
import math
import analysis_results.extract_score

def make_visualization(out_score,summary_all,antibiotics,level,f_fixed_threshold,epochs,learning,f_optimize_score,f_nn_base):

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

def make_visualization_Tree(out_score,summary_all,antibiotics,level,f_fixed_threshold,epochs,learning,f_optimize_score,f_nn_base):

    # print(species)
    # # antibiotics_selected = ast.literal_eval(antibiotics)
    # print('====> Select_antibiotic:', len(antibiotics), antibiotics)
    final=pd.DataFrame(index=antibiotics, columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy',
                                                   'mcc','f1_positive', 'f1_negative','precision_positive','recall_positive','auc','threshold','support','support_positive'] )
    final_plot=pd.DataFrame(index=antibiotics, columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy',
                                                   'mcc','f1_positive', 'f1_negative','precision_positive','recall_positive','auc','threshold','support','support_positive'] )

    count=0
    for anti in antibiotics:
        # save_name_score = amr_utility.name_utility.GETname_multi_bench_save_name_score(species, anti,level,learning,epochs,f_fixed_threshold,f_nn_base,f_optimize_score)
        # print(anti, '--------------------------------------------------------------------')
        # data = pd.read_csv(save_name_score+'_score.txt', sep="\t",index_col=0, header=0)
        # print(data)
        data=summary_all[count]
        count+=1
        data=data.loc[['mean','std'],:]
        final_plot.loc[anti, :] = data.loc['mean', :].to_list()
        data = data.astype(float).round(2)

        m= data.loc['mean',:].apply(lambda x: "{:.2f}".format(x))
        n=data.loc['std',:].apply(lambda x: "{:.2f}".format(x))

        final.loc[anti,:]=m.str.cat(n, sep='±').values



    if out_score=='f':
        final=final[['f1_macro','f1_positive', 'f1_negative','accuracy']]
        final_plot=final_plot[['f1_macro','f1_positive', 'f1_negative','accuracy']]
    elif out_score=='f_p_r':
        final = final[['f1_macro', 'precision_macro', 'recall_macro', 'accuracy']]
        final_plot = final_plot[['f1_macro', 'precision_macro', 'recall_macro', 'accuracy']]
    else:#all scores
        pass
    return final,final_plot

def multi_make_visualization(fscore,out_score,merge_name,All_antibiotics,level,f_fixed_threshold, epochs,learning,f_optimize_score,
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
        summary=analysis_results.extract_score.score_summary_normalCV(fscore,count_anti,summary,cv, score_report_test, aucs_test, mcc_test, save_name_score, thresholds_selected_test)



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
def multi_make_visualization_normalCV(fscore,out_score,merge_name,All_antibiotics,level,f_fixed_threshold, epochs,learning,f_optimize_score,
                             f_nn_base,cv,score,save_name_score,save_name_score_final):
    #only for multi-species,multi-output models. normal CV.
    #Only one table as output. i.e. all species share the same one score. June 23, finished.
    aucs_test = score[4][0]
    score_report_test = score[3][0]
    mcc_test = score[2][0]
    # print(score_report_test)
    thresholds_selected_test = score[0]
    # hyper_para=score[6]#one-element list
    # hyper_para2 = score[7]
    # hyper_para3 = score[8]
    # print('check',hyper_para)
    hy_para_all=[score[6][0],score[7][0],score[8][0]]
    final = pd.DataFrame(index=All_antibiotics,
                         columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy_macro',
                                        'mcc', 'f1_positive', 'f1_negative','precision_positive', 'recall_positive', 'auc','threshold',
                                        'support', 'support_positive'])
    count_anti = 0
    # hy_para_all=[]
    for anti in All_antibiotics:
        summary = pd.DataFrame(index=['score'],
                               columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy_macro',
                                        'mcc', 'f1_positive', 'f1_negative', 'precision_positive', 'recall_positive',
                                        'auc', 'threshold',
                                        'support', 'support_positive'])
        # print('count_anti----------------------:',count_anti)
        summary = analysis_results.extract_score.score_summary_normalCV(fscore,count_anti, summary, cv, score_report_test,
                                                                            aucs_test, mcc_test, save_name_score,
                                                                            thresholds_selected_test)

        count_anti += 1

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

    if out_score == 'f':
        final = final[['f1_macro', 'f1_positive', 'f1_negative', 'accuracy_macro']]

    elif out_score == 'f_p_r':
        final = final[['f1_macro', 'precision_macro', 'recall_macro', 'accuracy_macro']]

    else:  # all scores
        pass
    final['selected hyperparameter'] = [hy_para_all] * count_anti
    final.to_csv(save_name_score_final + '_score_final.txt', sep="\t")
    print(final)


def concat_multi_make_visualization(fscore,out_score,merge_name, All_antibiotics, level, f_fixed_threshold, epochs, learning,
                                 f_optimize_score,f_nn_base, cv, score,save_name_score, save_name_score_final):
    # only for multi-s concat models.
    aucs_test = score[4][0]
    score_report_test = score[3][0]
    mcc_test = score[2][0]
    thresholds_selected_test = score[0][0]

    final = pd.DataFrame(index=All_antibiotics,
                         columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy_macro',
                                  'mcc', 'f1_positive', 'f1_negative','precision_positive', 'recall_positive','auc', 'threshold',
                                  'support', 'support_positive'])
    count=0
    # print(len(aucs_test))
    for anti in All_antibiotics:

        # for i in np.arange(cv):
        if  len(All_antibiotics) > 1:
            report = score_report_test[count]  # multi-species model. should always be this

            # mcc = mcc_test[count]
            # auc=aucs_test[count]
            # thr=thresholds_selected_test
        else:
            pass
            # report = score_report_test
        report = pd.DataFrame(report).transpose()
        # print(report)
        # print('--------')

        # if 'accuracy' not in report.index.to_list():  # no resitance pheno in test folder.  should not happen
        #     accuracy='-'
        # else:
        #     accuracy=report.loc['accuracy', 'f1-score']
        # print(report)
        if not report.empty:
            summary = pd.DataFrame(index=['score'],
                                   columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy_macro',
                                            'mcc', 'f1_positive', 'f1_negative','precision_positive', 'recall_positive', 'auc',
                                            'threshold','support', 'support_positive'])

            summary = analysis_results.extract_score.score_summary_normalCV(fscore,count, summary, cv, score_report_test, aucs_test, mcc_test,
                                        save_name_score, thresholds_selected_test)

            # count += 1
            # data = summary.loc[['weighted-mean', 'weighted-std'], :]
            # # print(data)
            # data = data.astype(float).round(2)
            # # print(data)
            # m = data.loc['weighted-mean', :].apply(lambda x: "{:.2f}".format(x))
            # # print(m)
            # n = data.loc['weighted-std', :].apply(lambda x: "{:.2f}".format(x))
            # # print(data.dtypes)
            final.loc[anti, :] = summary.loc['score', :].to_list()
            # final.loc[anti, :] = m.str.cat(n, sep='±').values
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
    # final=final.fillna('-')
    if out_score=='f':
        final=final[['f1_macro','f1_positive', 'f1_negative','accuracy_macro']]

    elif out_score=='f_p_r':
        final = final[['f1_macro', 'precision_macro', 'recall_macro', 'accuracy_macro']]

    else:#all scores
        pass

    final.to_csv(save_name_score_final + '_score_final.txt', sep="\t")
    print('concat2 final------------------',final)


def concat_make_visualization2(fscore,out_score, merge_name, All_antibiotics, level, f_fixed_threshold, epochs,
                                      learning, f_optimize_score,
                                      f_nn_base, cv, score, save_name_score, save_name_score_final):
    # only for multi-species,multi-output models. normal CV. training scores
    # Only one table as output. i.e. all species share the same one score. June 23, finished.
    score_val=score[9]
    aucs_test = score_val[1]
    score_report_test = score_val[0]
    mcc_test = score_val[2]
    # print(mcc_test)
    thresholds_selected_test=[None]
    # print(score_report_test)
    # thresholds_selected_test = score[0][0]
    # hyper_para=score[6]#one-element list
    # hyper_para2 = score[7]
    # hyper_para3 = score[8]
    # print('check',hyper_para)
    # hy_para_all = [score[6][0], score[7][0], score[8][0]]
    final = pd.DataFrame(index=All_antibiotics,
                         columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy_macro',
                                  'mcc', 'f1_positive', 'f1_negative', 'precision_positive', 'recall_positive', 'auc',
                                  'threshold',
                                  'support', 'support_positive'])
    count_anti = 0
    # hy_para_all=[]
    # print(len(score_report_test))

    for anti in All_antibiotics:  # this is for the sake of concat2, validation scores.
        # if len(All_antibiotics) > 1:
        #     report = score_report_test[count_anti]  # multi-species model. should always be this
        # else:
        #     pass
        #
        # report = pd.DataFrame(report).transpose()


        summary = pd.DataFrame(index=['score'],
                               columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy_macro',
                                        'mcc', 'f1_positive', 'f1_negative', 'precision_positive',
                                        'recall_positive', 'auc', 'threshold',
                                        'support', 'support_positive'])
        # print('count_anti----------------------:',count_anti)
        summary = analysis_results.extract_score.score_summary_normalCV(fscore,count_anti, summary, cv, score_report_test,
                                                                        aucs_test, mcc_test, save_name_score,
                                                                        thresholds_selected_test)

        count_anti += 1

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

    if out_score == 'f':
        final = final[['f1_macro', 'f1_positive', 'f1_negative', 'accuracy_macro']]

    elif out_score == 'f_p_r':
        final = final[['f1_macro', 'precision_macro', 'recall_macro', 'accuracy_macro']]

    else:  # all scores
        pass
    # final['selected hyperparameter'] = [hy_para_all] * count_anti
    final.to_csv(save_name_score_final + '_score_final.txt', sep="\t")
    print('concat2,training scores:==============')
    print(final)
