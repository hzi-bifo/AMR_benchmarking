import os
import sys
# sys.path.append('../../')
sys.path.insert(0, os.getcwd())
import numpy as np
import pandas as pd
import statistics
import math


def weithgted_var(values,average,weights):
    n=len(values)
    p_variance = np.average((values - average) ** 2, weights=weights)#np-int, element-wise
    s_variance=p_variance * n/(n-1)
    return s_variance

def score_summary_normalCV(count_anti,summary,score_report_test,f1_test,aucs_test,mcc_test):
    #only for normal CV. So a score without std.

    if count_anti != None:#multi-species model.
        mcc_test=np.array(mcc_test)
        mcc_test=mcc_test[count_anti]
        aucs_test = np.array(aucs_test)
        aucs_test = aucs_test[count_anti]
        f1_test= np.array(f1_test)
        f1_test = f1_test[count_anti]

    for i in np.arange(1):
        if count_anti != None:
            report=score_report_test[count_anti]#multi-species model.
        else:
            report = score_report_test

        report=pd.DataFrame(report).transpose()
        accuracy=(report.iat[2,2])
        support=(report.loc['macro avg','support'])
        f1_pos=(report.loc['1', 'f1-score'])
        f1_neg=(report.loc['0', 'f1-score'])
        precision_pos=(report.loc['1', 'precision'])
        recall_pos=(report.loc['1', 'recall'])
        precision_neg=(report.loc['0', 'precision'])
        recall_neg=(report.loc['0', 'recall'])
        support_pos=(report.loc['1', 'support'])
        support_neg=(report.loc['0', 'support'])

        summary.loc['score','accuracy'] =  accuracy
        summary.loc['score', 'f1_macro'] = f1_test
        summary.loc['score', 'precision_neg'] = precision_neg
        summary.loc['score', 'recall_neg'] = recall_neg
        summary.loc['score', 'auc'] = aucs_test
        summary.loc['score', 'mcc'] =  mcc_test
        summary.loc['score', 'f1_positive'] = f1_pos
        summary.loc['score', 'f1_negative'] =  f1_neg
        summary.loc['score', 'precision_pos'] =  precision_pos
        summary.loc['score', 'recall_pos'] = recall_pos
        summary.loc['score', 'support'] = support
        summary.loc['score', 'support_pos'] =  support_pos
        summary.loc['score', 'support_neg'] =  support_neg

    return summary

def score_summary(count_anti,summary,cv,score_report_test,f1_test,aucs_test,mcc_test,thresholds_selected_test):
    #only for nested CV
    # f1=[]
    precision=[]
    recall=[]
    accuracy=[]
    f1_pos = []
    f1_neg = []
    precision_neg = []
    recall_neg = []
    support_pos=[]
    support_neg = []
    support=[]

    # print(count_anti)
    if count_anti != None:#multi-anti model.

        mcc_test=np.array(mcc_test)
        mcc_test=mcc_test[ : ,count_anti]
        ## mcc_test = [x for x in mcc_test if math.isnan(x) == False]#multi-anti MT model.
        aucs_test = np.array(aucs_test)
        aucs_test = aucs_test[:, count_anti]
        ## aucs_test = [x for x in aucs_test if math.isnan(x) == False]#multi-anti MT model.
        f1_test = np.array(f1_test)
        f1_test = f1_test[:, count_anti]
        f1_test = f1_test.tolist()
        indexNnan = [num for num, x in enumerate(f1_test, start=0) if math.isnan(x) == False]
        f1_test = [x for x in f1_test if math.isnan(x) == False]#multi-anti MT model. June 2022
        aucs_test=aucs_test[indexNnan] #multi-anti MT model.June 2022
        mcc_test=mcc_test[indexNnan]  #multi-anti MT model.June 2022
        thresholds_selected_test = np.array(thresholds_selected_test)
        thresholds_selected_test=thresholds_selected_test[indexNnan]  #multi-anti MT model.June 2022
        thresholds_selected_test=thresholds_selected_test.tolist()
        aucs_test = aucs_test.tolist()
        mcc_test=mcc_test.tolist()

    for i in np.arange(cv):
        if count_anti != None:#multi-anti model.
            report=score_report_test[i][count_anti]#multi-species model.
        else:
            report = score_report_test[i]
        # -------------------------------------------------
        if type(report) != dict: ##multi-anti MT model.
            pass
        else:

            report=pd.DataFrame(report).transpose()
            accuracy.append(report.iat[2,2])
            ## f1.append(report.loc['macro avg','f1-score'])
            precision.append(report.loc['macro avg','precision'])
            recall.append(report.loc['macro avg','recall'])
            support.append(report.loc['macro avg','support'])

            f1_pos.append(report.loc['1', 'f1-score'])
            f1_neg.append(report.loc['0', 'f1-score'])
            precision_neg.append(report.loc['0', 'precision'])
            recall_neg.append(report.loc['0', 'recall'])
            support_pos.append(report.loc['1', 'support'])
            support_neg.append(report.loc['0', 'support'])



    summary.loc['mean', 'f1_macro'] = statistics.mean(f1_test)
    summary.loc['std', 'f1_macro'] = statistics.stdev(f1_test)
    summary.loc['mean', 'precision_macro'] = statistics.mean(precision)
    summary.loc['std', 'precision_macro'] = statistics.stdev(precision)
    summary.loc['mean', 'recall_macro'] = statistics.mean(recall)
    summary.loc['std', 'recall_macro'] = statistics.stdev(recall)
    summary.loc['mean','accuracy'] = statistics.mean(accuracy)
    summary.loc['std','accuracy'] = statistics.stdev(accuracy)
    summary.loc['mean', 'mcc'] = statistics.mean(mcc_test)
    summary.loc['std', 'mcc'] = statistics.stdev(mcc_test)
    summary.loc['mean', 'f1_positive'] = statistics.mean(f1_pos)
    summary.loc['std', 'f1_positive'] = statistics.stdev(f1_pos)
    summary.loc['mean', 'f1_negative'] = statistics.mean(f1_neg)
    summary.loc['std', 'f1_negative'] = statistics.stdev(f1_neg)
    summary.loc['mean', 'precision_neg'] = statistics.mean(precision_neg)
    summary.loc['std', 'precision_neg'] = statistics.stdev(precision_neg)
    summary.loc['mean', 'recall_neg'] = statistics.mean(recall_neg)
    summary.loc['std', 'recall_neg'] = statistics.stdev(recall_neg)
    summary.loc['mean', 'auc'] = statistics.mean(aucs_test)
    summary.loc['std', 'auc'] = statistics.stdev(aucs_test)
    summary.loc['mean', 'threshold'] = statistics.mean(thresholds_selected_test)
    summary.loc['std', 'threshold'] = statistics.stdev(thresholds_selected_test)
    summary.loc['mean', 'support'] = statistics.mean(support)
    summary.loc['std', 'support'] = statistics.stdev(support)
    summary.loc['mean', 'support_positive'] = statistics.mean(support_pos)
    summary.loc['std', 'support_positive'] = statistics.stdev(support_pos)


    f1_average = np.average(f1_test, weights=support)
    precision_average = np.average(precision, weights=support)
    recall_average = np.average(recall, weights=support)
    f1_pos_average = np.average(f1_pos, weights=support)
    f1_neg_average = np.average(f1_neg, weights=support)
    precision_neg_average = np.average(precision_neg, weights=support)
    recall_neg_average = np.average(recall_neg, weights=support)
    aucs_average = np.average(aucs_test, weights=support)
    mcc_average = np.average(mcc_test, weights=support)
    thr_average = np.average(thresholds_selected_test, weights=support)
    accuracy_average = np.average(accuracy, weights=support)
    # print(summary)

    summary.loc['weighted-mean', 'f1_macro'] = f1_average
    summary.loc['weighted-std', 'f1_macro'] = math.sqrt(weithgted_var(f1_test, f1_average, support))
    summary.loc['weighted-mean', 'precision_macro'] = precision_average
    summary.loc['weighted-std', 'precision_macro'] = math.sqrt(weithgted_var(precision, precision_average, support))
    summary.loc['weighted-mean', 'recall_macro'] = recall_average
    summary.loc['weighted-std', 'recall_macro'] = math.sqrt(weithgted_var(recall, recall_average, support))
    summary.loc['weighted-mean','accuracy'] = accuracy_average
    summary.loc['weighted-std','accuracy'] =  math.sqrt(weithgted_var(accuracy, accuracy_average, support))
    summary.loc['weighted-mean', 'mcc'] = mcc_average
    summary.loc['weighted-std', 'mcc'] =  math.sqrt(weithgted_var(mcc_test, mcc_average, support))
    summary.loc['weighted-mean', 'f1_positive'] = f1_pos_average
    summary.loc['weighted-std', 'f1_positive'] =  math.sqrt(weithgted_var(f1_pos, f1_pos_average, support))
    summary.loc['weighted-mean', 'f1_negative'] = f1_neg_average
    summary.loc['weighted-std', 'f1_negative'] =  math.sqrt(weithgted_var(f1_neg, f1_neg_average, support))
    summary.loc['weighted-mean', 'precision_neg'] = precision_neg_average
    summary.loc['weighted-std', 'precision_neg'] =  math.sqrt(weithgted_var(precision_neg, precision_neg_average, support))
    summary.loc['weighted-mean', 'recall_neg'] = recall_neg_average
    summary.loc['weighted-std', 'recall_neg'] =  math.sqrt(weithgted_var(recall_neg, recall_neg_average, support))
    summary.loc['weighted-mean', 'auc'] = aucs_average
    summary.loc['weighted-std', 'auc'] =  math.sqrt(weithgted_var(aucs_test, aucs_average, support))
    summary.loc['weighted-mean', 'threshold'] = thr_average
    summary.loc['weighted-std', 'threshold'] =  math.sqrt(weithgted_var(thresholds_selected_test, thr_average, support))
    summary.loc['weighted-mean', 'support'] = statistics.mean(support)
    summary.loc['weighted-std', 'support'] = statistics.stdev(support)
    summary.loc['weighted-mean', 'support_positive'] = statistics.mean(support_pos)
    summary.loc['weighted-std', 'support_positive'] = statistics.stdev(support_pos)


    return summary

def score_summary_Tree(count_anti,summary,cv,score_report_test,f1_test,aucs_test,mcc_test,thresholds_selected_test):

    # f1=[]
    precision=[]
    recall=[]
    accuracy=[]
    f1_pos = []
    f1_neg = []
    precision_neg = []
    recall_neg = []
    support_pos=[]
    support_neg = []
    support=[]

    # print(count_anti)
    if count_anti != None:#multi-species model.
        mcc_test=np.array(mcc_test)
        # print('mcc_test shape',mcc_test.shape)
        mcc_test=mcc_test[ : ,count_anti]
        mcc_test=mcc_test.tolist()
        aucs_test = np.array(aucs_test)
        aucs_test = aucs_test[:, count_anti]
        aucs_test = aucs_test.tolist()

        f1_test = np.array(f1_test)
        f1_test = f1_test[:, count_anti]
        f1_test = f1_test.tolist()

    for i in np.arange(cv):
        if count_anti != None:
            report=score_report_test[i][count_anti]#multi-species model.
        else:
            report = score_report_test[i]
        report=pd.DataFrame(report).transpose()
        # print(report)


        accuracy.append(report.iat[2,2])
        ## f1.append(report.loc['macro avg','f1-score'])
        precision.append(report.loc['macro avg','precision'])
        recall.append(report.loc['macro avg','recall'])
        support.append(report.loc['macro avg','support'])

        f1_pos.append(report.loc['1', 'f1-score'])
        f1_neg.append(report.loc['0', 'f1-score'])
        precision_neg.append(report.loc['0', 'precision'])
        recall_neg.append(report.loc['0', 'recall'])
        support_pos.append(report.loc['1', 'support'])
        support_neg.append(report.loc['0', 'support'])


    summary.loc['mean','accuracy'] = statistics.mean(accuracy)
    summary.loc['std','accuracy'] = statistics.stdev(accuracy)
    summary.loc['mean', 'f1_macro'] = statistics.mean(f1_test)
    summary.loc['std', 'f1_macro'] = statistics.stdev(f1_test)
    summary.loc['mean', 'precision_macro'] = statistics.mean(precision)
    summary.loc['std', 'precision_macro'] = statistics.stdev(precision)
    summary.loc['mean', 'recall_macro'] = statistics.mean(recall)
    summary.loc['std', 'recall_macro'] = statistics.stdev(recall)
    summary.loc['mean', 'auc'] = statistics.mean(aucs_test)
    summary.loc['std', 'auc'] = statistics.stdev(aucs_test)
    summary.loc['mean', 'mcc'] = statistics.mean(mcc_test)
    summary.loc['std', 'mcc'] = statistics.stdev(mcc_test)
    summary.loc['mean', 'threshold'] = statistics.mean(thresholds_selected_test)
    summary.loc['std', 'threshold'] = statistics.stdev(thresholds_selected_test)

    summary.loc['mean', 'f1_positive'] = statistics.mean(f1_pos)
    summary.loc['std', 'f1_positive'] = statistics.stdev(f1_pos)
    summary.loc['mean', 'f1_negative'] = statistics.mean(f1_neg)
    summary.loc['std', 'f1_negative'] = statistics.stdev(f1_neg)
    summary.loc['mean', 'precision_neg'] = statistics.mean(precision_neg)
    summary.loc['std', 'precision_neg'] = statistics.stdev(precision_neg)
    summary.loc['mean', 'recall_neg'] = statistics.mean(recall_neg)
    summary.loc['std', 'recall_neg'] = statistics.stdev(recall_neg)
    summary.loc['mean', 'support'] = statistics.mean(support)
    summary.loc['std', 'support'] = statistics.stdev(support)
    summary.loc['mean', 'support_positive'] = statistics.mean(support_pos)
    summary.loc['std', 'support_positive'] = statistics.stdev(support_pos)

    return summary

def score_clinical(summary, cv, score_report_test):
    from sympy import symbols, Eq, solve
    ## Only for F1-negative, precision-negative. Intergrate predicted results from all folds, and compute the scores.
    ## this is done due to the uneven distribution of R and S samples across the folds under phylo-folds and kma-based folds.
    ## but for convenience of comparison, we also intergated the results evaluated under random folds for clinical oriented scores generation.
    ## Nov 24 2022.
    TP=[]
    FP=[]
    TN=[]
    FN=[]

    for i in np.arange(cv):
        report = score_report_test[i]
        i+=1
        if type(report) != dict: ##multi-anti MT model.
            pass
        else:
            report=pd.DataFrame(report).transpose()
            # print(report)
            precision_pos=report.loc['1', 'precision']
            recall_pos=report.loc['1', 'recall']
            precision_neg=report.loc['0', 'precision']
            recall_neg=report.loc['0', 'recall']
            support_neg=report.loc['0', 'support']
            support_pos=report.loc['1', 'support']

            tp,fp,tn,fn = symbols('tp,fp,tn,fn')
            eq1 = Eq((tp-precision_pos*(tp+fp)),0)
            eq2 = Eq((tp-recall_pos*(tp+fn)),0)
            eq3 = Eq((tn-precision_neg*(tn+fn)),0 )
            eq4 = Eq((tn-recall_neg*(tn+fp)),0)
            # # eq1 = Eq((tp/(tp+fp)), precision_pos)
            # # eq2 = Eq((tp/(tp+fn)),recall_pos)
            # # eq3 = Eq((tn/(tn+fn)), precision_neg)
            # # eq4 = Eq((tn/(tn+fp)),recall_neg)
            eq5 = Eq(tn+fp,support_neg)
            eq6 = Eq(tp+fn,support_pos)

            confusion=solve((eq1, eq2,eq3, eq4,eq5,eq6), (tp,fp,tn,fn))
            if confusion==[]:
                confusion=solve((eq1, eq2,eq3, eq4,eq5,eq6), (tp,fp,tn,fn), rational=False)
                if confusion==[]:
                    A=np.array([[(1-precision_pos), -precision_pos, 0,0],
                                [0,0,(1-precision_neg), -precision_neg],[0,1,1,0],[1,0,0,1]])
                    B = np.array([0, 0, support_neg,support_pos])
                    tp_,fp_,tn_,fn_ = np.linalg.inv(A).dot(B)
                    confusion={tp: tp_, fp: fp_, tn: tn_, fn: fn_}

            tp_sub=confusion[tp]
            TP.append(tp_sub)
            fp_sub=confusion[fp]
            FP.append(fp_sub)
            tn_sub=confusion[tn]
            TN.append(tn_sub)
            fn_sub=confusion[fn]
            FN.append(fn_sub)
    tp_whole=sum(TP)
    fp_whole=sum(FP)
    tn_whole=sum(TN)
    fn_whole=sum(FN)

    clinical_f1_negative=2*tn_whole/(2*tn_whole+fn_whole+fp_whole)
    if (tn_whole+fn_whole)>0 :
        clinical_precision_neg=tn_whole/(tn_whole+fn_whole)
    else:
        clinical_precision_neg=0
    clinical_recall_neg=tn_whole/(tn_whole+fp_whole)
    summary.loc['value', :] =[clinical_f1_negative,clinical_precision_neg,clinical_recall_neg]

    # print(summary)
    return summary


#
# def summary_allLoop(count_anti,summary, cv,score_report_test,f1_test, aucs_test, mcc_test,anti ):
#     '''
#     Summarize scores in each loop, for visualiztion(bar plot with sig interval) comparison of MSMA.
#     No use so far, as we changed to bar plot with no sig interval, but standard deviation.
#     '''
#
#     if count_anti != None:  # multi-species model.
#         mcc_test = np.array(mcc_test)
#         print('mcc_test shape', mcc_test.shape)
#         mcc_test = mcc_test[:, count_anti]
#         mcc_test = mcc_test.tolist()
#         aucs_test = np.array(aucs_test)
#         aucs_test = aucs_test[:, count_anti]
#         aucs_test = aucs_test.tolist()
#         f1_test = np.array(f1_test)
#         f1_test = f1_test[:, count_anti]
#         f1_test=f1_test.tolist()
#     for i in np.arange(cv):
#         if count_anti != None:
#             report = score_report_test[i][count_anti]  # multi-species model.
#             # mcc_test_anti.append(mcc_test[i][count_anti])
#         else:
#             report = score_report_test[i]
#         report = pd.DataFrame(report).transpose()
#
#         summary_sub=pd.DataFrame(columns=['antibiotic','f1_macro', 'precision_macro', 'recall_macro', 'accuracy',
#                                         'mcc', 'f1_positive', 'f1_negative', 'precision_positive',
#                                         'recall_positive', 'auc'])
#
#         summary_sub.loc['score',:]=[anti,f1_test[i],report.loc['macro avg', 'precision'],report.loc['macro avg', 'recall'],
#                                 report.iat[2,2],mcc_test[i],report.loc['1', 'f1-score'],report.loc['0', 'f1-score'],
#                                 report.loc['1', 'precision'],report.loc['1', 'recall'],aucs_test[i]]
#
#         summary = summary.append(summary_sub, ignore_index=True)
#     return summary
#






