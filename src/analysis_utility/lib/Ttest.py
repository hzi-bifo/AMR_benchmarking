
import numpy as np
from scipy.stats import ttest_rel
import pandas as pd

'''no use so far.'''

def Get_pvalue(cv,innescore_report_test_1,score_report_test_2):
    #return P values of F1-macro, F1-pos, F1-neg based on two sets of reports. TODO, F1-macro change from using reports to F1-macro directly

    f1_macro_1=[]
    f1_macro_2=[]
    f1_macro_1_pos=[]
    f1_macro_2_pos=[]
    f1_macro_1_neg=[]
    f1_macro_2_neg=[]
    for i in np.arange(cv):
        report_1 = innescore_report_test_1[i]
        report_2 = score_report_test_2[i]
        f1_macro_1_sub=pd.DataFrame(report_1).transpose().loc['macro avg','f1-score']
        f1_macro_2_sub = pd.DataFrame(report_2).transpose().loc['macro avg','f1-score']
        f1_macro_1_pos_sub = pd.DataFrame(report_1).transpose().loc['1', 'f1-score']
        f1_macro_2_pos_sub = pd.DataFrame(report_2).transpose().loc['1', 'f1-score']
        f1_macro_1_neg_sub = pd.DataFrame(report_1).transpose().loc['0', 'f1-score']
        f1_macro_2_neg_sub = pd.DataFrame(report_2).transpose().loc['0', 'f1-score']
        f1_macro_1.append(f1_macro_1_sub)
        f1_macro_2.append(f1_macro_2_sub)
        f1_macro_1_pos.append(f1_macro_1_pos_sub)
        f1_macro_2_pos.append(f1_macro_2_pos_sub)
        f1_macro_1_neg.append(f1_macro_1_neg_sub)
        f1_macro_2_neg.append(f1_macro_2_neg_sub)


    result=ttest_rel(f1_macro_1, f1_macro_2)
    result_pos=ttest_rel(f1_macro_1_pos, f1_macro_2_pos)
    result_neg=ttest_rel(f1_macro_1_neg, f1_macro_2_neg)
    pvalue=result[1]
    pvalue_pos=result_pos[1]
    pvalue_neg=result_neg[1]
    # print(pvalue)
    # print(pvalue_pos)
    # fix_select.append(pvalue)
    return pvalue,pvalue_pos,pvalue_neg
