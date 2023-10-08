
import os
import sys
# sys.path.append('../../')
sys.path.insert(0, os.getcwd())
import pandas as pd
from src.analysis_utility.lib import extract_score,math_utility


whole_score_set=['f1_macro', 'f1_positive', 'f1_negative', 'accuracy',
        'precision_macro', 'recall_macro', 'precision_negative', 'recall_negative','precision_positive', 'recall_positive',
        'mcc',  'auc','threshold', 'support', 'support_positive','support_negative']
score_set=['f1_macro', 'f1_positive', 'f1_negative', 'accuracy',
        'precision_macro', 'recall_macro', 'precision_negative', 'recall_negative','precision_positive', 'recall_positive',
        'mcc', 'auc']
score_set_weighted=['weighted-'+ a for a in score_set]

def make_visualization(summary_all,antibiotics ):

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
        final_std.loc[anti,:]=data.loc['weighted-std',:].to_list()
        final_plot.loc[anti,:]=data.loc['weighted-mean',:].to_list()

        data = data.astype(float).round(2)

        m= data.loc['weighted-mean',:].apply(lambda x: "{:.2f}".format(x))
        n=data.loc['weighted-std',:].apply(lambda x: "{:.2f}".format(x))

        final.loc[anti,:]=m.str.cat(n, sep='±').values

    final=final[score_set_weighted]
    final_plot=final_plot[score_set_weighted]
    final_std=final_std[score_set_weighted]

    return final,final_plot,final_std

def make_visualization_Tree(summary_all,antibiotics):


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

    final=final[score_set]
    final_plot=final_plot[score_set]
    final_std=final_std[score_set]

    return final,final_plot,final_std

def multi_make_visualization(All_antibiotics,cv,score):
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
                         columns=whole_score_set)
    count_anti = 0


    for anti in All_antibiotics:

        summary = pd.DataFrame(index=['mean', 'std', 'weighted-mean', 'weighted-std'],
                                           columns=whole_score_set)
        print('count_anti----------------------:',count_anti,anti)
        summary=extract_score.score_summary(count_anti,summary,cv, score_report_test, f1macro,aucs_test, mcc_test, thresholds_selected_test)


        count_anti+=1
        data = summary.loc[['weighted-mean', 'weighted-std'], :]
        data = data.astype(float).round(2)
        m = data.loc['weighted-mean', :].apply(lambda x: "{:.2f}".format(x))
        n = data.loc['weighted-std', :].apply(lambda x: "{:.2f}".format(x))

        final.loc[anti, :] = m.str.cat(n, sep='±').values

    final = final[score_set]

    common,ind= math_utility.get_most_fre_hyper(hy_para_collection,False)
    hy_para_fre=common.to_dict()
    final['the most frequent hyperparameter'] = [hy_para_fre]*count_anti
    final['selected hyperparameter'] = [hy_para_all]*count_anti
    final['frequency(out of 10)'] = [ind]*count_anti
    return final



def multi_make_visualization_normalCV(All_antibiotics,score):
    #only for multi-species,multi-output models. normal CV.
    #Only one table as output. i.e. all species share the same one score.
    f1macro=score[0][0]
    score_report_test = score[1][0]
    aucs_test = score[2][0]
    mcc_test = score[3][0]
    thresholds_selected_test = score[4][0]
    hy_para_all=[score[5][0],score[6][0],score[7][0]]#1*n_cv
    final = pd.DataFrame(index=All_antibiotics,
                         columns=score_set)
    count_anti = 0
    for anti in All_antibiotics:
        summary = pd.DataFrame(index=['score'],
                               columns=score_set)

        summary = extract_score.score_summary_normalCV(count_anti, summary,score_report_test,
                                                                            f1macro,aucs_test, mcc_test)

        count_anti += 1
        final.loc[anti, :] = summary.loc['score', :].to_list()

    final = final[score_set]
    final['selected hyperparameter'] = [hy_para_all] * count_anti
    return final



def concat_multi_make_visualization(All_antibiotics , score):
    # only for multi-s concat models.
    f1macro=score[0][0]
    score_report_test = score[1][0]
    aucs_test = score[2][0]
    mcc_test = score[3][0]
    ## thresholds_selected_test = score[4][0]
    # # hy_para_all=[score[5][0],score[6][0],score[7][0]]#1*n_cv

    final = pd.DataFrame(index=All_antibiotics,columns=score_set)
    count=0

    for anti in All_antibiotics:

        if  len(All_antibiotics) > 1:
            report = score_report_test[count]  # multi-species model. should always be this
        else:# shouldn't happen
            print('please check if you are running multi-species-antibiotic model.')
            exit(1)
        report = pd.DataFrame(report).transpose()


        if not report.empty:
            summary = pd.DataFrame(index=['score'],columns=score_set)
            summary = extract_score.score_summary_normalCV(count, summary,  score_report_test, f1macro,aucs_test, mcc_test)
            final.loc[anti, :] = summary.loc['score', :].to_list()
        count+=1
    final=final[score_set ]

    return final


def make_visualization_clinical( score_list, summary_all, antibiotics):
    final=pd.DataFrame(index=antibiotics, columns=score_list)

    count = 0
    for anti in antibiotics:
        data=summary_all[count]
        final.loc[anti,:]=data.values.tolist()[0]
        count+=1

    return final


