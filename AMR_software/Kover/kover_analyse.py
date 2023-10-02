#!/usr/bin/python
from src.amr_utility import name_utility, file_utility,load_data
import argparse,os,json
import numpy as np
from sklearn.metrics import classification_report,f1_score,roc_curve,auc,matthews_corrcoef
import statistics,math
import pandas as pd
import warnings
warnings.filterwarnings('ignore')



''' For summerize and visualize the outoputs of Kover.'''

score_set=['f1_macro', 'f1_positive', 'f1_negative', 'accuracy',
        'precision_macro', 'recall_macro', 'precision_negative', 'recall_negative','precision_positive', 'recall_positive',
        'mcc',  'auc']


def get_summary( anti,final_table,final_plot , final_std ,f1_list,accuracy,f1_pos,f1_neg,precision,recall,
                                                            precision_neg,recall_neg,precision_pos,recall_pos,mcc_test,aucs_test,
                                                            support,score_list,f_kma):
    if f_kma:
        summary=pd.DataFrame(index=['weighted-mean', 'weighted-std'],columns=score_list)

        f1_average = np.average(f1_list, weights=support)
        precision_average = np.average(precision, weights=support)
        recall_average = np.average(recall, weights=support)
        f1_pos_average = np.average(f1_pos, weights=support)
        f1_neg_average = np.average(f1_neg, weights=support)
        precision_neg_average = np.average(precision_neg, weights=support)
        recall_neg_average = np.average(recall_neg, weights=support)
        precision_pos_average = np.average(precision_pos, weights=support)
        recall_pos_average = np.average(recall_pos, weights=support)
        aucs_average = np.average(aucs_test, weights=support)
        mcc_average = np.average(mcc_test, weights=support)

        accuracy_average = np.average(accuracy, weights=support)
        summary.loc['weighted-mean', 'f1_macro'] = f1_average
        summary.loc['weighted-std', 'f1_macro'] = math.sqrt(weithgted_var(f1_list, f1_average, support))
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

        summary.loc['weighted-mean', 'precision_negative'] = precision_neg_average
        summary.loc['weighted-std', 'precision_negative'] =  math.sqrt(weithgted_var(precision_neg, precision_neg_average, support))
        summary.loc['weighted-mean', 'recall_negative'] = recall_neg_average
        summary.loc['weighted-std', 'recall_negative'] =  math.sqrt(weithgted_var(recall_neg, recall_neg_average, support))

        summary.loc['weighted-mean', 'precision_positive'] = precision_pos_average
        summary.loc['weighted-std', 'precision_positive'] =  math.sqrt(weithgted_var(precision_pos, precision_pos_average, support))
        summary.loc['weighted-mean', 'recall_positive'] = recall_pos_average
        summary.loc['weighted-std', 'recall_positive'] =  math.sqrt(weithgted_var(recall_pos, recall_pos_average, support))
        summary.loc['weighted-mean', 'auc'] = aucs_average
        summary.loc['weighted-std', 'auc'] =  math.sqrt(weithgted_var(aucs_test, aucs_average, support))


        m = summary.loc['weighted-mean',:].apply(lambda x: "{:.2f}".format(x))
        n = summary.loc['weighted-std',:].apply(lambda x: "{:.2f}".format(x))
        final_table.loc[anti,:]=m.str.cat(n, sep='±').values
        final_plot.loc[anti,:]=summary.loc['weighted-mean',:].to_list()
        final_std.loc[anti,:]=summary.loc['weighted-std',:].to_list()
    else:
        summary=pd.DataFrame(index=[' mean', ' std'],columns=score_list)

        summary.loc['mean','accuracy'] = statistics.mean(accuracy)
        summary.loc['std','accuracy'] = statistics.stdev(accuracy)
        summary.loc['mean', 'f1_macro'] = statistics.mean(f1_list)
        summary.loc['std', 'f1_macro'] = statistics.stdev(f1_list)
        summary.loc['mean', 'precision_macro'] = statistics.mean(precision)
        summary.loc['std', 'precision_macro'] = statistics.stdev(precision)
        summary.loc['mean', 'recall_macro'] = statistics.mean(recall)
        summary.loc['std', 'recall_macro'] = statistics.stdev(recall)
        summary.loc['mean', 'auc'] = statistics.mean(aucs_test)
        summary.loc['std', 'auc'] = statistics.stdev(aucs_test)
        summary.loc['mean', 'mcc'] = statistics.mean(mcc_test)
        summary.loc['std', 'mcc'] = statistics.stdev(mcc_test)


        summary.loc['mean', 'f1_positive'] = statistics.mean(f1_pos)
        summary.loc['std', 'f1_positive'] = statistics.stdev(f1_pos)
        summary.loc['mean', 'f1_negative'] = statistics.mean(f1_neg)
        summary.loc['std', 'f1_negative'] = statistics.stdev(f1_neg)
        summary.loc['mean', 'precision_negative'] = statistics.mean(precision_neg)
        summary.loc['std', 'precision_negative'] = statistics.stdev(precision_neg)
        summary.loc['mean', 'recall_negative'] = statistics.mean(recall_neg)
        summary.loc['std', 'recall_negative'] = statistics.stdev(recall_neg)

        summary.loc['mean', 'precision_positive'] = statistics.mean(precision_pos)
        summary.loc['std', 'precision_positive'] = statistics.stdev(precision_pos)
        summary.loc['mean', 'recall_positive'] = statistics.mean(recall_pos)
        summary.loc['std', 'recall_positive'] = statistics.stdev(recall_pos)



        m = summary.loc['mean',:].apply(lambda x: "{:.2f}".format(x))
        n = summary.loc['std',:].apply(lambda x: "{:.2f}".format(x))
        final_table.loc[anti,:]=m.str.cat(n, sep='±').values
        final_plot.loc[anti,:]=summary.loc['mean',:].to_list()
        final_std.loc[anti,:]=summary.loc['std',:].to_list()
    return  final_table,final_plot , final_std

def get_scores(test_corrects_list,test_errors_list,name_list2):
    ''' name_list2: ID, resistant_phenotype '''
    y_true=[]
    y_pre=[]
    for each in test_corrects_list:
        p=name_list2[name_list2['ID']==each].iat[0,1]

        y_true.append(p)
        y_pre.append(p)


    for each in test_errors_list:
        p=name_list2[name_list2['ID']==each].iat[0,1]
        if p==1:
            y_true.append(1)
            y_pre.append(0)
        else:
            y_true.append(0)
            y_pre.append(1)

    df=classification_report(y_true, y_pre, labels=[0, 1], output_dict=True,zero_division=0)
    f1_test=f1_score(y_true, y_pre, average='macro')
    report = pd.DataFrame(df).transpose()
    fpr, tpr, _ = roc_curve(y_true, y_pre, pos_label=1)
    roc_auc = auc(fpr, tpr)
    mcc=matthews_corrcoef(y_true, y_pre)

    return f1_test,report,mcc, roc_auc


def weithgted_var(values,average,weights):
    n=len(values)
    p_variance = np.average((values - average) ** 2, weights=weights)#np-int, element-wise
    s_variance=p_variance * n/(n-1)
    return s_variance

def extract_info_species( level,species,cv,f_phylotree,f_kma,temp_path,output_path):

    score_list=score_set

    antibiotics, _, _ =  load_data.extract_info(species, False, level)

    for chosen_cl in ['scm','tree']:

        if f_kma:
            final_table = pd.DataFrame(index=antibiotics,columns=["weighted-"+x for x in score_list])
            final_plot = pd.DataFrame(index=antibiotics,columns=["weighted-"+x for x in score_list])
            final_std = pd.DataFrame(index=antibiotics,columns=["weighted-"+x for x in score_list])

        else:#if f_phylotree or f_random
            final_table = pd.DataFrame(index=antibiotics,columns=score_list)
            final_plot = pd.DataFrame(index=antibiotics,columns=score_list)
            final_std = pd.DataFrame(index=antibiotics,columns=score_list)


        for anti in antibiotics:
            print(anti)

            f1_list,accuracy_list,f1_positive_list,f1_negative_list=[],[],[],[]
            support=[]
            support_pos=[]
            support_neg=[]
            precision_neg_list=[]
            recall_neg_list=[]
            precision_pos_list=[]
            recall_pos_list=[]
            precision_list=[]
            recall_list=[]
            mcc_list=[]
            auc_list=[]


            _,name,meta_txt,_ = name_utility.GETname_model2('kover',level, species, anti,'',temp_path,f_kma,f_phylotree)
            name_list = pd.read_csv(name, index_col=0, dtype={'genome_id': object}, sep="\t")
            name_list.loc[:,'ID'] = 'iso_' + name_list['genome_id'].astype(str)
            name_list2 = name_list.loc[:, ['ID', 'resistant_phenotype']]
            for outer_cv in range(cv):
                with open(meta_txt+'_temp/'+str(chosen_cl)+'_b_'+str(outer_cv)+'/results.json') as f:
                    data = json.load(f)

                test_errors_list=data["classifications"]['test_errors']
                test_corrects_list=data["classifications"]['test_correct']

                # calculate scores based on prediction infor of all samples.
                f1_test,report,mcc, auc=get_scores(test_corrects_list,test_errors_list,name_list2)

                accuracy_list.append(report.iat[2,2])#no use of this score
                f1_list.append(f1_test)
                ## f1_list.append(report.loc['macro avg','f1-score'])
                precision_list.append(report.loc['macro avg','precision'])
                recall_list.append(report.loc['macro avg','recall'])
                support.append(report.loc['macro avg','support'])
                f1_positive_list.append(report.loc['1', 'f1-score'])
                f1_negative_list.append(report.loc['0', 'f1-score'])
                precision_neg_list.append(report.loc['0', 'precision'])
                recall_neg_list.append(report.loc['0', 'recall'])
                precision_pos_list.append(report.loc['1', 'precision'])
                recall_pos_list.append(report.loc['1', 'recall'])
                # support_pos.append(report.loc['1', 'support'])
                # support_neg.append(report.loc['0', 'support'])
                mcc_list.append(mcc)
                auc_list.append(auc)

            ### update summary tables: only the column corresponding to this anti.
            final_table,final_plot , final_std=get_summary( anti,final_table,final_plot , final_std ,f1_list,accuracy_list,
                                                            f1_positive_list,f1_negative_list,precision_list,recall_list,
                                                            precision_neg_list,recall_neg_list,precision_pos_list,recall_pos_list,mcc_list,auc_list,
                                                            support,score_list,f_kma)
        save_name_score,_ = name_utility.GETname_result('kover', species, '',f_kma,f_phylotree,chosen_cl,output_path)
        file_utility.make_dir(os.path.dirname(save_name_score))
        final_table.to_csv(save_name_score + '.txt', sep="\t")
        final_plot.to_csv(save_name_score + '_PLOT.txt', sep="\t")
        final_std.to_csv(save_name_score + '_std.txt', sep="\t")
        print(final_table)




def extract_best_estimator(level,species,cv,fscore,f_phylotree,f_kma,temp_path,output_path):
    '''
     Aug 2023: select the estimator based on inner looper CV
    for each species
    final_score:the score used for classifiers comparison.
    '''
    print(species)
    antibiotics, _, _ =  load_data.extract_info(species, False, level)
    score_list=score_set

    if f_kma:
        final_table = pd.DataFrame(index=antibiotics,columns=["weighted-"+x for x in score_list])
        final_plot = pd.DataFrame(index=antibiotics,columns=["weighted-"+x for x in score_list])
        final_std = pd.DataFrame(index=antibiotics,columns=["weighted-"+x for x in score_list])

    else:#if f_phylotree or f_random
        final_table = pd.DataFrame(index=antibiotics,columns=score_list)
        final_plot = pd.DataFrame(index=antibiotics,columns=score_list)
        final_std = pd.DataFrame(index=antibiotics,columns=score_list)


    chosen_cl_all=[] #only for clinical
    for anti in antibiotics:
        # print(anti)

        f1_list,accuracy_list,f1_positive_list,f1_negative_list=[],[],[],[]
        support=[]
        support_pos=[]
        support_neg=[]
        precision_neg_list=[]
        recall_neg_list=[]
        precision_pos_list=[]
        recall_pos_list=[]
        precision_list=[]
        recall_list=[]
        mcc_list=[]
        auc_list=[]
        chosen_cl_list=[] #for clinical
        for outer_cv in range(cv):
            MEAN=[]
            STD=[]
            CL=[]
            mean_list=[]
            for cl_each in  ['scm','tree']:
                ### CV on 9 folds to only select classifier.
                _,name,meta_txt,_ = name_utility.GETname_model2_val('kover',level, species, anti,'',temp_path,f_kma,f_phylotree)
                for inner_cv in range(cv-1):
                    name_list = pd.read_csv(name, index_col=0, dtype={'genome_id': object}, sep="\t")
                    name_list.loc[:,'ID'] = 'iso_' + name_list['genome_id'].astype(str)
                    name_list2 = name_list.loc[:, ['ID', 'resistant_phenotype']]

                    with open(meta_txt+'_temp/'+str(cl_each)+'_b_'+str(outer_cv)+'_'+str(inner_cv)+'/results.json') as f:
                        data = json.load(f)

                    ##################only for check, delete later. checked correct!
                    # n2=np.genfromtxt(meta_txt + '_Train_outer_' + str(outer_cv)+'_inner_'+str(inner_cv) + '_id', dtype="str")
                    # print(len(n2))
                    # train_errors_list=data["classifications"]['train_errors']
                    # train_corrects_list=data["classifications"]['train_correct']
                    # t=train_errors_list+train_corrects_list
                    # print(len(t))
                    # exit()
                    ####################
                    #####################
                    test_errors_list=data["classifications"]['test_errors']
                    test_corrects_list=data["classifications"]['test_correct']
                    f1,_,_,_=get_scores(test_corrects_list,test_errors_list,name_list2)
                    mean_list.append(f1)

                MEAN.append(statistics.mean(mean_list))
                STD.append(statistics.stdev(mean_list))
                CL.append(cl_each)
            ##select the highest mean, resorting to the lowest std.
            combined = [(MEAN[i], STD[i]) for i in range(len(MEAN))]
            # Sort the list of tuples first by a in descending order and then by b in ascending order
            sorted_combined = sorted(combined, key=lambda x: (-x[0], x[1]))
            # Get the index of the first element in the sorted list
            optimal_index = MEAN.index(sorted_combined[0][0])
            chosen_cl=CL[optimal_index]
            chosen_cl_list.append(chosen_cl)
            # print('chose: ',chosen_cl)
            ###########################################################################################
            ### report the coresponding classifier's evaluation resulting in this outer loop iteration.
            _,name,meta_txt,_ = name_utility.GETname_model2('kover',level, species, anti,'',temp_path,f_kma,f_phylotree)
            with open(meta_txt+'_temp/'+str(chosen_cl)+'_b_'+str(outer_cv)+'/results.json') as f:
                data = json.load(f)
            name_list = pd.read_csv(name, index_col=0, dtype={'genome_id': object}, sep="\t")
            name_list.loc[:,'ID'] = 'iso_' + name_list['genome_id'].astype(str)
            name_list2 = name_list.loc[:, ['ID', 'resistant_phenotype']]

            test_errors_list=data["classifications"]['test_errors']
            test_corrects_list=data["classifications"]['test_correct']

            ### calculate scores based on prediction infor of all samples.
            f1_test,report,mcc, auc=get_scores(test_corrects_list,test_errors_list,name_list2)

            accuracy_list.append(report.iat[2,2])#no use of this score
            f1_list.append(f1_test)

            precision_list.append(report.loc['macro avg','precision'])
            recall_list.append(report.loc['macro avg','recall'])
            support.append(report.loc['macro avg','support'])
            f1_positive_list.append(report.loc['1', 'f1-score'])
            f1_negative_list.append(report.loc['0', 'f1-score'])
            precision_neg_list.append(report.loc['0', 'precision'])
            recall_neg_list.append(report.loc['0', 'recall'])
            precision_pos_list.append(report.loc['1', 'precision'])
            recall_pos_list.append(report.loc['1', 'recall'])
            # support_pos.append(report.loc['1', 'support'])
            # support_neg.append(report.loc['0', 'support'])
            mcc_list.append(mcc)
            auc_list.append(auc)



        chosen_cl_all.append(chosen_cl_list)
        ### form scores into a table for output.
        final_table,final_plot , final_std=get_summary( anti,final_table,final_plot , final_std ,f1_list,accuracy_list,
                                                            f1_positive_list,f1_negative_list,precision_list,recall_list,
                                                            precision_neg_list,recall_neg_list,precision_pos_list,recall_pos_list,mcc_list,auc_list,
                                                            support,score_list,f_kma)

        ### Add clinical_oriented.
        # final_table,final_plot=extract_info_species_clinical(final_table,final_plot,anti, chosen_cl_list,level,species,cv,f_phylotree,f_kma,temp_path)

    ####Add clinical_oriented.
    clinical_table=extract_info_species_clinical(chosen_cl_all,level,species,cv,f_phylotree,f_kma,temp_path)
    final_table = pd.concat([final_table, clinical_table], axis=1, join="inner")
    final_plot = pd.concat([final_plot, clinical_table], axis=1, join="inner")
    # # #################################


    _,save_name_final = name_utility.GETname_result('kover', species, fscore,f_kma,f_phylotree,'',output_path)
    file_utility.make_dir(os.path.dirname(save_name_final))

    final_table.to_csv(save_name_final + '_SummaryBenchmarking.txt', sep="\t")
    final_plot.to_csv(save_name_final + '_SummaryBenchmarking_PLOT.txt', sep="\t")
    final_std.to_csv(save_name_final + '_SummaryBenchmarking_std.txt', sep="\t")

    with open(save_name_final + '_classifier.json', 'w') as f:  # overwrite mode. ## only for misclassifier.py usage. Added 7 Sep 2023.
        json.dump(chosen_cl_all, f)




def extract_info_species_clinical(chosen_cl_list,level,species,cv,f_phylotree,f_kma,temp_path):
    clinical_score=['clinical_f1_negative','clinical_precision_negative', 'clinical_recall_negative']

    antibiotics, _, _ =  load_data.extract_info(species, False, level)
    final_table = pd.DataFrame(index=antibiotics,columns=clinical_score)
    i_anti=0
    for anti in antibiotics:

        y_true=[]
        y_pre=[]

        _,name,meta_txt,_ = name_utility.GETname_model2('kover',level, species, anti,'',temp_path,f_kma,f_phylotree)
        name_list = pd.read_csv(name, index_col=0, dtype={'genome_id': object}, sep="\t")
        name_list.loc[:,'ID'] = 'iso_' + name_list['genome_id'].astype(str)
        name_list2 = name_list.loc[:, ['ID', 'resistant_phenotype']]
        for outer_cv in range(cv):
            chosen_cl=chosen_cl_list[i_anti][outer_cv]

            with open(meta_txt+'_temp/'+str(chosen_cl)+'_b_'+str(outer_cv)+'/results.json') as f:
                data = json.load(f)

                test_errors_list=data["classifications"]['test_errors']
                test_corrects_list=data["classifications"]['test_correct']

                for each in test_corrects_list:
                    p=name_list2[name_list2['ID']==each].iat[0,1]
                    y_true.append(p)
                    y_pre.append(p)

                for each in test_errors_list:
                    p=name_list2[name_list2['ID']==each].iat[0,1]
                    if p==1:
                        y_true.append(1)
                        y_pre.append(0)
                    else:
                        y_true.append(0)
                        y_pre.append(1)
        i_anti+=1
        df=classification_report(y_true, y_pre, labels=[0, 1], output_dict=True,zero_division=0)
        report = pd.DataFrame(df).transpose()
        f1_negative=report.loc['0', 'f1-score']
        precision_neg=report.loc['0', 'precision']
        recall_neg=report.loc['0', 'recall']
        final_table.loc[anti,clinical_score]=[f1_negative,precision_neg,recall_neg]

    return final_table


def extract_info(level,s,f_all,cv,fscore,f_phylotree,f_kma,temp_path,output_path):
    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    if f_all == False:
        data = data.loc[s, :]
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()


    for df_species, antibiotics in zip(df_species, antibiotics):
        #### 1. extract evaluation information for each classifier.
        # extract_info_species(level, df_species, cv,f_phylotree,f_kma,temp_path,output_path)

        ### 2. extract inner CV results for choosing the best classifier between scm and carrt(tree).
        ### including also clinical-oriented scores here.
        ### select criteria: f1-macro.
        extract_best_estimator(level, df_species,cv,fscore,f_phylotree,f_kma,temp_path,output_path)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                        help='Directory to store temporary files.')
    parser.add_argument('-out', '--output_path', default='./', type=str, required=False,
                        help='Directory to store CV scores.')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                        help='kma based cv folders.')
    parser.add_argument('-l', '--level', default='loose', type=str,
                        help='Quality control: strict or loose')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument('-fscore', '--fscore', default='f1_macro', type=str, required=False,
                        help='Deprecated. No use anymore. ')
    parser.add_argument('-f_ml', '--f_ml', dest='f_ml', action='store_true',
                        help='prepare bash')
    parser.add_argument("-cv", "--cv", default=10, type=int,
                        help='CV splits number')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'Pseudomonas aeruginosa\' \
                    \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
                    \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parsedArgs = parser.parse_args()
    extract_info( parsedArgs.level,parsedArgs.species, parsedArgs.f_all,parsedArgs.cv,parsedArgs.fscore,parsedArgs.f_phylotree,
                 parsedArgs.f_kma,parsedArgs.temp_path,parsedArgs.output_path)






