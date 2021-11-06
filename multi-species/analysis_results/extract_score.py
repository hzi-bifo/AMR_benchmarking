import os
import numpy as np
import pandas as pd
import statistics
import math


def weithgted_var(values,average,weights):
    n=len(values)
    p_variance = np.average((values - average) ** 2, weights=weights)#np-int, element-wise
    s_variance=p_variance * n/(n-1)
    return s_variance

def score_summary_normalCV(count_anti,summary,cv,score_report_test,aucs_test,mcc_test,save_name_score,thresholds_selected_test):
    #only for normal CV. So a score without std. Aug 13th,2021.
    f1=[]
    precision=[]
    recall=[]
    accuracy=[]
    f1_pos = []
    f1_neg = []
    precision_pos = []
    recall_pos = []
    support_pos=[]
    support_neg = []
    support=[]
    f_noAccu = []
    # print(count_anti)
    if count_anti != None:#multi-species model.
        mcc_test=np.array(mcc_test)

        # print('mcc_test shape',mcc_test.shape)
        mcc_test=mcc_test[count_anti]
        # mcc_test=mcc_test.tolist()
        aucs_test = np.array(aucs_test)
        aucs_test = aucs_test[count_anti]
        # aucs_test = aucs_test.tolist()
    for i in np.arange(1):

        if count_anti != None:
            # print(len(score_report_test[0]))
            # print()
            # print(score_report_test)
            # print('count_anti',count_anti)
            # print(score_report_test)
            report=score_report_test[count_anti]#multi-species model.

            # mcc_test_anti.append(mcc_test[i][count_anti])
        else:
            report = score_report_test

        report=pd.DataFrame(report).transpose()
        print('*****',report)

        # if 'accuracy' not in report.index.to_list():# no resitance pheno in test folder
        # print(report)
        if report.loc['1', 'support']==0 or report.loc['0', 'support']==0:  #  todo only one pheno in test folder
            # accuracy.append('-')
            print('Warning! Only one phenotype in the testing set. Exit!')
            print(report)
            exit()
            f_noAccu.append(i)
        else:
            accuracy=(report.loc['accuracy', 'f1-score'])

        f1=(report.loc['macro avg','f1-score'])
        precision=(report.loc['macro avg','precision'])
        recall=(report.loc['macro avg','recall'])
        support=(report.loc['macro avg','support'])

        f1_pos=(report.loc['1', 'f1-score'])
        f1_neg=(report.loc['0', 'f1-score'])
        precision_pos=(report.loc['1', 'precision'])

        recall_pos=(report.loc['1', 'recall'])
        support_pos=(report.loc['1', 'support'])
        support_neg=(report.loc['0', 'support'])

        summary.loc['score','accuracy_macro'] =  accuracy
        summary.loc['score', 'f1_macro'] = f1
        summary.loc['score', 'precision_macro'] = precision
        summary.loc['score', 'recall_macro'] = recall
        summary.loc['score', 'auc'] = aucs_test
        summary.loc['score', 'mcc'] =  mcc_test
        summary.loc['score', 'threshold'] = thresholds_selected_test
        summary.loc['score', 'f1_positive'] = f1_pos
        summary.loc['score', 'f1_negative'] =  f1_neg

        summary.loc['score', 'precision_positive'] =  precision_pos
        summary.loc['score', 'recall_positive'] = recall_pos
        summary.loc['score', 'support'] = support
        summary.loc['score', 'support_positive'] =  support_pos


    return summary

def score_summary(count_anti,summary,cv,score_report_test,aucs_test,mcc_test,save_name_score,thresholds_selected_test):
    #only for nested CV
    f1=[]
    precision=[]
    recall=[]
    accuracy=[]
    f1_pos = []
    f1_neg = []
    precision_pos = []
    recall_pos = []
    support_pos=[]
    support_neg = []
    support=[]
    f_noAccu = []
    # print(count_anti)
    if count_anti != None:#multi-species model.
        mcc_test=np.array(mcc_test)
        print('mcc_test shape',mcc_test.shape)
        mcc_test=mcc_test[ : ,count_anti]
        mcc_test=mcc_test.tolist()
        aucs_test = np.array(aucs_test)
        aucs_test = aucs_test[:, count_anti]
        aucs_test = aucs_test.tolist()
    for i in np.arange(cv):
        # print('i:',i)
        # print(len(score_report_test),len(score_report_test[i]))
        if count_anti != None:#multi-species model.
            # print(len(score_report_test[0]))
            # print()
            report=score_report_test[i][count_anti]#multi-species model.
            # mcc_test_anti.append(mcc_test[i][count_anti])
        else:
            report = score_report_test[i]
        report=pd.DataFrame(report).transpose()
        # print(report)

        # check if only one pheno in test folder




        # if 'accuracy' not in report.index.to_list():# no resitance pheno in test folder
        if report.loc['1', 'support']==0 or report.loc['0', 'support']==0:  #  todo only one pheno in test folder
            accuracy.append('-')
            print('Please count this! Only one phenotype in the testing folder!!!!!!!!!!!!!!!!')
            # print(report)
            f_noAccu.append(i)
        else:
            accuracy.append(report.loc['accuracy', 'f1-score'])

        f1.append(report.loc['macro avg','f1-score'])
        precision.append(report.loc['macro avg','precision'])
        recall.append(report.loc['macro avg','recall'])
        support.append(report.loc['macro avg','support'])

        f1_pos.append(report.loc['1', 'f1-score'])
        f1_neg.append(report.loc['0', 'f1-score'])
        precision_pos.append(report.loc['1', 'precision'])
        recall_pos.append(report.loc['1', 'recall'])
        support_pos.append(report.loc['1', 'support'])
        support_neg.append(report.loc['0', 'support'])


    if f_noAccu != []:
        #rm the iteration's results, where no resistance phenotype in the test folder.
        f1 = [i for j, i in enumerate(f1) if j not in f_noAccu]
        precision = [i for j, i in enumerate(precision) if j not in f_noAccu]
        recall = [i for j, i in enumerate(recall) if j not in f_noAccu]
        accuracy = [i for j, i in enumerate(accuracy) if j not in f_noAccu]
        support = [i for j, i in enumerate(support) if j not in f_noAccu]
        mcc_test = [i for j, i in enumerate(mcc_test) if j not in f_noAccu]
        aucs_test = [i for j, i in enumerate(aucs_test) if j not in f_noAccu]
        thresholds_selected_test = [i for j, i in enumerate(thresholds_selected_test) if j not in f_noAccu]
        f1_pos = [i for j, i in enumerate(f1_pos) if j not in f_noAccu]
        f1_neg = [i for j, i in enumerate(f1_neg) if j not in f_noAccu]
        precision_pos = [i for j, i in enumerate(precision_pos) if j not in f_noAccu]
        recall_pos = [i for j, i in enumerate(recall_pos) if j not in f_noAccu]
        support_pos = [i for j, i in enumerate(support_pos) if j not in f_noAccu]
        support_neg = [i for j, i in enumerate(support_neg) if j not in f_noAccu]

    summary.loc['mean','accuracy_macro'] = statistics.mean(accuracy)
    summary.loc['std','accuracy_macro'] = statistics.stdev(accuracy)
    summary.loc['mean', 'f1_macro'] = statistics.mean(f1)
    summary.loc['std', 'f1_macro'] = statistics.stdev(f1)
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
    summary.loc['mean', 'precision_positive'] = statistics.mean(precision_pos)
    summary.loc['std', 'precision_positive'] = statistics.stdev(precision_pos)
    summary.loc['mean', 'recall_positive'] = statistics.mean(recall_pos)
    summary.loc['std', 'recall_positive'] = statistics.stdev(recall_pos)
    summary.loc['mean', 'support'] = statistics.mean(support)
    summary.loc['std', 'support'] = statistics.stdev(support)
    summary.loc['mean', 'support_positive'] = statistics.mean(support_pos)
    summary.loc['std', 'support_positive'] = statistics.stdev(support_pos)


    f1_average = np.average(f1, weights=support)
    precision_average = np.average(precision, weights=support)
    recall_average = np.average(recall, weights=support)


    f1_pos_average = np.average(f1_pos, weights=support_pos)
    f1_neg_average = np.average(f1_neg, weights=support_neg)
    precision_pos_average = np.average(precision_pos, weights=support_pos)
    recall_pos_average = np.average(recall_pos, weights=support_pos)


    aucs_average = np.average(aucs_test, weights=support)
    mcc_average = np.average(mcc_test, weights=support)
    thr_average = np.average(thresholds_selected_test, weights=support)
    accuracy_average = np.average(accuracy, weights=support)
    # print(summary)
    summary.loc['weighted-mean', :] = [f1_average, precision_average, recall_average, accuracy_average,
                                       mcc_average,f1_pos_average, f1_neg_average, precision_pos_average,recall_pos_average, aucs_average,thr_average,
                                       statistics.mean(support),statistics.mean(support_pos)]

    summary.loc['weighted-std', :] = [math.sqrt(weithgted_var(f1, f1_average, support)),
                                      math.sqrt(weithgted_var(precision, precision_average, support)),
                                      math.sqrt(weithgted_var(recall, recall_average, support)),
                                      math.sqrt(weithgted_var(accuracy, accuracy_average, support)),
                                      math.sqrt(weithgted_var(mcc_test, mcc_average, support)),
                                      math.sqrt(weithgted_var(f1_pos, f1_pos_average, support_pos)),
                                      math.sqrt(weithgted_var(f1_neg, f1_neg_average, support_neg)),
                                      math.sqrt(weithgted_var(precision_pos, precision_pos_average, support_pos)),
                                      math.sqrt(weithgted_var(recall_pos, recall_pos_average, support_pos)),
                                      math.sqrt(weithgted_var(aucs_test, aucs_average, support)),
                                      math.sqrt(weithgted_var(thresholds_selected_test, thr_average, support)),
                                      statistics.stdev(support),statistics.stdev(support_pos)]



    # summary.to_csv(save_name_score+'_score.txt', sep="\t")
    # print(summary)
    return summary

def score_summary_Tree(count_anti,summary,cv,score_report_test,aucs_test,mcc_test,save_name_score,thresholds_selected_test):

    f1=[]
    precision=[]
    recall=[]
    accuracy=[]
    f1_pos = []
    f1_neg = []
    precision_pos = []
    recall_pos = []
    support_pos=[]
    support_neg = []
    support=[]
    f_noAccu = []
    # print(count_anti)
    if count_anti != None:#multi-species model.
        mcc_test=np.array(mcc_test)
        print('mcc_test shape',mcc_test.shape)
        mcc_test=mcc_test[ : ,count_anti]
        mcc_test=mcc_test.tolist()
        aucs_test = np.array(aucs_test)
        aucs_test = aucs_test[:, count_anti]
        aucs_test = aucs_test.tolist()
    for i in np.arange(cv):
        if count_anti != None:
            report=score_report_test[i][count_anti]#multi-species model.
            # mcc_test_anti.append(mcc_test[i][count_anti])
        else:
            report = score_report_test[i]
        report=pd.DataFrame(report).transpose()
        # print(report)

        # check if only one pheno in test folder




        # if 'accuracy' not in report.index.to_list():# no resitance pheno in test folder
        if report.loc['1', 'support']==0 or report.loc['0', 'support']==0:  #  todo only one pheno in test folder
            accuracy.append('-')
            print('Please count this! Only one phenotype in the testing folder!!!!!!!!!!!!!!!!')
            # print(report)
            f_noAccu.append(i)
        else:
            accuracy.append(report.loc['accuracy', 'f1-score'])

        f1.append(report.loc['macro avg','f1-score'])
        precision.append(report.loc['macro avg','precision'])
        recall.append(report.loc['macro avg','recall'])
        support.append(report.loc['macro avg','support'])

        f1_pos.append(report.loc['1', 'f1-score'])
        f1_neg.append(report.loc['0', 'f1-score'])
        precision_pos.append(report.loc['1', 'precision'])
        recall_pos.append(report.loc['1', 'recall'])
        support_pos.append(report.loc['1', 'support'])
        support_neg.append(report.loc['0', 'support'])


    if f_noAccu != []:
        #rm the iteration's results, where no resistance phenotype in the test folder.
        f1 = [i for j, i in enumerate(f1) if j not in f_noAccu]
        precision = [i for j, i in enumerate(precision) if j not in f_noAccu]
        recall = [i for j, i in enumerate(recall) if j not in f_noAccu]
        accuracy = [i for j, i in enumerate(accuracy) if j not in f_noAccu]
        support = [i for j, i in enumerate(support) if j not in f_noAccu]
        mcc_test = [i for j, i in enumerate(mcc_test) if j not in f_noAccu]
        aucs_test = [i for j, i in enumerate(aucs_test) if j not in f_noAccu]
        thresholds_selected_test = [i for j, i in enumerate(thresholds_selected_test) if j not in f_noAccu]
        f1_pos = [i for j, i in enumerate(f1_pos) if j not in f_noAccu]
        f1_neg = [i for j, i in enumerate(f1_neg) if j not in f_noAccu]
        precision_pos = [i for j, i in enumerate(precision_pos) if j not in f_noAccu]
        recall_pos = [i for j, i in enumerate(recall_pos) if j not in f_noAccu]
        support_pos = [i for j, i in enumerate(support_pos) if j not in f_noAccu]
        support_neg = [i for j, i in enumerate(support_neg) if j not in f_noAccu]


    summary.loc['mean','accuracy_macro'] = statistics.mean(accuracy)
    summary.loc['std','accuracy_macro'] = statistics.stdev(accuracy)
    summary.loc['mean', 'f1_macro'] = statistics.mean(f1)
    summary.loc['std', 'f1_macro'] = statistics.stdev(f1)
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
    summary.loc['mean', 'precision_positive'] = statistics.mean(precision_pos)
    summary.loc['std', 'precision_positive'] = statistics.stdev(precision_pos)
    summary.loc['mean', 'recall_positive'] = statistics.mean(recall_pos)
    summary.loc['std', 'recall_positive'] = statistics.stdev(recall_pos)
    summary.loc['mean', 'support'] = statistics.mean(support)
    summary.loc['std', 'support'] = statistics.stdev(support)
    summary.loc['mean', 'support_positive'] = statistics.mean(support_pos)
    summary.loc['std', 'support_positive'] = statistics.stdev(support_pos)

    return summary


def summary_allLoop(count_anti,summary, cv,score_report_test, aucs_test, mcc_test,anti ):
    f1 = []
    precision = []
    recall = []
    accuracy = []
    f1_pos = []
    f1_neg = []
    precision_pos = []
    recall_pos = []
    support_pos = []
    support_neg = []
    support = []
    f_noAccu = []
    # print(count_anti)
    if count_anti != None:  # multi-species model.
        mcc_test = np.array(mcc_test)
        print('mcc_test shape', mcc_test.shape)
        mcc_test = mcc_test[:, count_anti]
        mcc_test = mcc_test.tolist()
        aucs_test = np.array(aucs_test)
        aucs_test = aucs_test[:, count_anti]
        aucs_test = aucs_test.tolist()
    for i in np.arange(cv):
        if count_anti != None:
            report = score_report_test[i][count_anti]  # multi-species model.
            # mcc_test_anti.append(mcc_test[i][count_anti])
        else:
            report = score_report_test[i]
        report = pd.DataFrame(report).transpose()
        # print(report)

        # check if only one pheno in test folder

        # if 'accuracy' not in report.index.to_list():# no resitance pheno in test folder
        if report.loc['1', 'support'] == 0 or report.loc['0', 'support'] == 0:  # todo only one pheno in test folder
            # accuracy.append('-')
            # print('Please count this! Only one phenotype in the testing folder!!!!!!!!!!!!!!!!')
            # print(report)
            # f_noAccu.append(i)
            pass
        else:
            # print(report)
            summary_sub=pd.DataFrame(columns=['antibiotic','f1_macro', 'precision_macro', 'recall_macro', 'accuracy_macro',
                                            'mcc', 'f1_positive', 'f1_negative', 'precision_positive',
                                            'recall_positive', 'auc'])

            summary_sub.loc['score',:]=[anti,report.loc['macro avg', 'f1-score'],report.loc['macro avg', 'precision'],report.loc['macro avg', 'recall'],
                                    report.loc['accuracy', 'f1-score'],mcc_test[i],report.loc['1', 'f1-score'],report.loc['0', 'f1-score'],
                                    report.loc['1', 'precision'],report.loc['1', 'recall'],aucs_test[i]]

            summary = summary.append(summary_sub, ignore_index=True)
    return summary