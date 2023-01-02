
import os
import sys
# sys.path.append('../../')
sys.path.insert(0, os.getcwd())
import pandas as pd
from src.analysis_utility.lib import extract_score,math_utility



def make_visualization(out_score,summary_all,antibiotics ):

    column_name=summary_all[0].columns.tolist()
    column_name=['weighted-'+ name for name in column_name]
    final=pd.DataFrame(index=antibiotics, columns=column_name )
    final_plot=pd.DataFrame(index=antibiotics, columns=column_name)
    final_std=pd.DataFrame(index=antibiotics, columns=column_name)


    count = 0
    for anti in antibiotics:

        data=summary_all[count]
        count+=1
        data=data.loc[['weighted-mean','weighted-std'],:]
        data = data.astype(float).round(2)

        m= data.loc['weighted-mean',:].apply(lambda x: "{:.2f}".format(x))
        n=data.loc['weighted-std',:].apply(lambda x: "{:.2f}".format(x))

        final.loc[anti,:]=m.str.cat(n, sep='±').values
        final_std.loc[anti,:]=data.loc['weighted-std',:].to_list()
        final_plot.loc[anti,:]=data.loc['weighted-mean',:].to_list()
    if out_score=='f':
        final=final[['weighted-f1_macro','weighted-f1_positive', 'weighted-f1_negative','weighted-accuracy']]
        final_plot=final_plot[['weighted-f1_macro','weighted-f1_positive', 'weighted-f1_negative','weighted-accuracy']]
        final_std=final_std[['weighted-f1_macro','weighted-f1_positive', 'weighted-f1_negative','weighted-accuracy']]
    elif out_score=='neg':
        final = final[['weighted-f1_macro', 'weighted-f1_positive','weighted-f1_negative','weighted-precision_neg', 'weighted-recall_neg', 'weighted-accuracy']]
        final_plot = final_plot[['weighted-f1_macro', 'weighted-f1_positive','weighted-f1_negative','weighted-precision_neg', 'weighted-recall_neg', 'weighted-accuracy']]
        final_std=final_std[['weighted-f1_macro','weighted-f1_positive', 'weighted-f1_negative','weighted-precision_neg', 'weighted-recall_neg', 'weighted-accuracy']]
    else:#all scores
        pass
    return final,final_plot,final_std

def make_visualization_Tree(out_score,summary_all,antibiotics):


    column_name=summary_all[0].columns.tolist()

    final=pd.DataFrame(index=antibiotics, columns=column_name )
    final_plot=pd.DataFrame(index=antibiotics, columns=column_name)
    final_std=pd.DataFrame(index=antibiotics, columns=column_name)

    count=0
    for anti in antibiotics:

        data=summary_all[count]
        count+=1
        data=data.loc[['mean','std'],:]
        final_plot.loc[anti, :] = data.loc['mean', :].to_list()
        final_std.loc[anti, :] = data.loc['std', :].to_list()
        data = data.astype(float).round(2)

        m= data.loc['mean',:].apply(lambda x: "{:.2f}".format(x))
        n=data.loc['std',:].apply(lambda x: "{:.2f}".format(x))

        final.loc[anti,:]=m.str.cat(n, sep='±').values

    

    if out_score=='f':
        final=final[['f1_macro','f1_positive', 'f1_negative','accuracy']]
        final_plot=final_plot[['f1_macro','f1_positive', 'f1_negative','accuracy']]
        final_std=final_std[['f1_macro','f1_positive', 'f1_negative','accuracy']]
    elif out_score=='neg':
        final = final[['f1_macro', 'f1_positive','f1_negative','precision_neg', 'recall_neg', 'accuracy']]
        final_plot = final_plot[['f1_macro','f1_positive',  'f1_negative','precision_neg', 'recall_neg', 'accuracy']]
        final_std = final_std[['f1_macro', 'f1_positive', 'f1_negative','precision_neg', 'recall_neg', 'accuracy']]
    else:#all scores
        pass
    return final,final_plot,final_std

def multi_make_visualization(out_score,All_antibiotics,cv,score):
    #only for SSMA models. nested CV.
    #so far, only KMA folds
    #Only one table as output. i.e. all combinations in a species share the same one score.
    f1macro=score[0]
    score_report_test = score[1]
    aucs_test = score[2]
    mcc_test = score[3]
    thresholds_selected_test = score[4]
    hy_para_all=[score[5],score[6],score[7]]#1*n_cv
    hy_para_collection=score[5]#10 dimension. each reapresents one outer loop.
    final = pd.DataFrame(index=All_antibiotics,
                         columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy',
                                        'mcc', 'f1_positive', 'f1_negative','precision_neg', 'recall_neg', 'auc','threshold',
                                        'support', 'support_positive'])
    count_anti = 0


    for anti in All_antibiotics:

        summary = pd.DataFrame(index=['mean', 'std', 'weighted-mean', 'weighted-std'],
                                           columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy',
                                                    'mcc', 'f1_positive','f1_negative', 'precision_neg', 'recall_neg', 'auc',
                                                    'threshold', 'support', 'support_positive'])
        print('count_anti----------------------:',count_anti,anti)
        summary=extract_score.score_summary(count_anti,summary,cv, score_report_test, f1macro,aucs_test, mcc_test, thresholds_selected_test)


        count_anti+=1
        data = summary.loc[['weighted-mean', 'weighted-std'], :]
        data = data.astype(float).round(2)
        m = data.loc['weighted-mean', :].apply(lambda x: "{:.2f}".format(x))
        n = data.loc['weighted-std', :].apply(lambda x: "{:.2f}".format(x))

        final.loc[anti, :] = m.str.cat(n, sep='±').values
    if out_score == 'f':
        final = final[['f1_macro', 'f1_positive', 'f1_negative', 'accuracy']]

    # elif out_score == 'f_p_r':
    #     final = final[['f1_macro', 'precision_macro', 'recall_macro', 'accuracy']]
    elif out_score=='neg':
        final = final[['f1_macro', 'f1_positive','f1_negative','precision_neg', 'recall_neg', 'accuracy']]

    else:  # all scores
        print('oerroe! only f or f_p_r for out_score.')
        exit(1)
    common,ind= math_utility.get_most_fre_hyper(hy_para_collection,False)
    hy_para_fre=common.to_dict()
    final['the most frequent hyperparameter'] = [hy_para_fre]*count_anti
    final['selected hyperparameter'] = [hy_para_all]*count_anti
    final['frequency(out of 10)'] = [ind]*count_anti
    return final



def multi_make_visualization_normalCV(out_score,All_antibiotics,score):
    #only for multi-species,multi-output models. normal CV.
    #Only one table as output. i.e. all species share the same one score.
    f1macro=score[0][0]
    score_report_test = score[1][0]
    aucs_test = score[2][0]
    mcc_test = score[3][0]
    thresholds_selected_test = score[4][0]
    hy_para_all=[score[5][0],score[6][0],score[7][0]]#1*n_cv
    score_list=['f1_macro',  'f1_positive', 'f1_negative', 'accuracy','precision_neg', 'recall_neg',
                                  'precision_pos', 'recall_pos', 'auc','mcc','support', 'support_pos', 'support_neg']
    final = pd.DataFrame(index=All_antibiotics,
                         columns=score_list)
    count_anti = 0
    for anti in All_antibiotics:
        summary = pd.DataFrame(index=['score'],
                               columns=score_list)
        # print('count_anti----------------------:',count_anti)
        summary = extract_score.score_summary_normalCV(count_anti, summary,score_report_test,
                                                                            f1macro,aucs_test, mcc_test)

        count_anti += 1
        final.loc[anti, :] = summary.loc['score', :].to_list()
    if out_score == 'f':
        final = final[['f1_macro', 'f1_positive', 'f1_negative', 'accuracy']]
    elif out_score=='neg':
        final = final[['f1_macro', 'f1_positive','f1_negative','precision_neg', 'recall_neg', 'accuracy']]
    else:
        exit(1)
    final['selected hyperparameter'] = [hy_para_all] * count_anti
    return final



def concat_multi_make_visualization(out_score, All_antibiotics , score):
    # only for multi-s concat models.
    f1macro=score[0][0]
    score_report_test = score[1][0]
    aucs_test = score[2][0]
    mcc_test = score[3][0]
    ## thresholds_selected_test = score[4][0]
    # # hy_para_all=[score[5][0],score[6][0],score[7][0]]#1*n_cv
    score_list=['f1_macro',  'f1_positive', 'f1_negative', 'accuracy','precision_neg', 'recall_neg',
                                  'precision_pos', 'recall_pos', 'auc','mcc','support', 'support_pos', 'support_neg']
    final = pd.DataFrame(index=All_antibiotics,columns=score_list)
    count=0
    # print(len(aucs_test))
    for anti in All_antibiotics:

        # for i in np.arange(cv):
        if  len(All_antibiotics) > 1:
            report = score_report_test[count]  # multi-species model. should always be this
        else:# shouldn't happen
            exit(1)
        report = pd.DataFrame(report).transpose()


        if not report.empty:
            summary = pd.DataFrame(index=['score'],columns=score_list)
            summary = extract_score.score_summary_normalCV(count, summary,  score_report_test, f1macro,aucs_test, mcc_test)
            final.loc[anti, :] = summary.loc['score', :].to_list()
        count+=1

    if out_score=='f':
        final=final[['f1_macro','f1_positive', 'f1_negative','accuracy']]
    elif out_score=='neg':
        final = final[['f1_macro', 'f1_positive','f1_negative','precision_neg', 'recall_neg', 'accuracy']]
    else:
        exit(1)
    return final


def make_visualization_clinical( score_list, summary_all, antibiotics):
    final=pd.DataFrame(index=antibiotics, columns=score_list)

    count = 0
    for anti in antibiotics:
        data=summary_all[count]
        final.loc[anti,:]=data.values.tolist()[0]
        count+=1

    return final


