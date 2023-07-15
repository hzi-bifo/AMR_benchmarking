#!/usr/bin/python
from src.amr_utility import name_utility, file_utility,load_data
import argparse,os,json,ast
import numpy as np
from sklearn.metrics import classification_report,f1_score
import statistics,math
import pandas as pd
from src.analysis_utility.result_analysis import extract_best_estimator_clinical

''' For summerize and visualize the outoputs of Kover.'''
def weithgted_var(values,average,weights):
    n=len(values)
    p_variance = np.average((values - average) ** 2, weights=weights)#np-int, element-wise
    s_variance=p_variance * n/(n-1)
    return s_variance

def extract_info_species( level,species,cv,f_phylotree,f_kma,temp_path,output_path):
    # score_list=['f1_macro','accuracy', 'f1_positive','f1_negative','precision_neg', 'recall_neg']
    score_list=['f1_macro','accuracy', 'f1_positive','f1_negative']

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
        # final_best=pd.DataFrame(index=antibiotics,columns=['f1_macro','classifier'])# the results from the better classifier of scm and tree.

        for anti in antibiotics:
            print(anti)

            f1_list,accuracy_list,f1_positive_list,f1_negative_list=[],[],[],[]
            support=[]
            support_pos=[]
            support_neg=[]
            ## precision_neg_list=[]
            ## recall_neg_list=[]

            _,name,meta_txt,_ = name_utility.GETname_model2('kover',level, species, anti,'',temp_path,f_kma,f_phylotree)
            name_list = pd.read_csv(name, index_col=0, dtype={'genome_id': object}, sep="\t")
            name_list.loc[:,'ID'] = 'iso_' + name_list['genome_id'].astype(str)
            name_list2 = name_list.loc[:, ['ID', 'resistant_phenotype']]
            for outer_cv in range(cv):
                with open(meta_txt+'_temp/'+str(chosen_cl)+'_b_'+str(outer_cv)+'/results.json') as f:
                    data = json.load(f)

                test_errors_list=data["classifications"]['test_errors']
                test_corrects_list=data["classifications"]['test_correct']


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
                accuracy_list.append(report.iat[2,2])#no use of this score
                f1_list.append(f1_test)
                ## f1_list.append(report.loc['macro avg','f1-score'])
                ## precision.append(report.loc['macro avg','precision'])
                ## recall.append(report.loc['macro avg','recall'])
                support.append(report.loc['macro avg','support'])
                f1_positive_list.append(report.loc['1', 'f1-score'])
                f1_negative_list.append(report.loc['0', 'f1-score'])
                ## precision_neg_list.append(report.loc['0', 'precision'])
                ## recall_neg_list.append(report.loc['0', 'recall'])
                support_pos.append(report.loc['1', 'support'])
                support_neg.append(report.loc['0', 'support'])


            if f_kma:
                final_table_sub=pd.DataFrame(index=['weighted-mean', 'weighted-std'],columns=score_list)
                # print(f1_list)
                f1_average=np.average(f1_list, weights=support)
                accuracy_average=np.average(accuracy_list, weights=support)
                f1_pos_average=np.average(f1_positive_list, weights=support)#change from support-pos to support. May 2022
                f1_neg_average=np.average(f1_negative_list, weights=support)# change from support-neg to support. May 2022.

                final_table_sub.loc['weighted-mean',:] = [f1_average,accuracy_average,f1_pos_average, f1_neg_average]
                final_table_sub.loc['weighted-std',:] = [math.sqrt(weithgted_var(f1_list,f1_average,support )),
                                                       math.sqrt(weithgted_var(accuracy_list, accuracy_average, support)),
                                                       math.sqrt(weithgted_var(f1_positive_list, f1_pos_average, support)), #change from support-pos to support. May 2022
                                                       math.sqrt(weithgted_var(f1_negative_list, f1_neg_average, support))]# change from support-neg to support. May 2022.




                m = final_table_sub.loc['weighted-mean',:].apply(lambda x: "{:.2f}".format(x))
                n = final_table_sub.loc['weighted-std',:].apply(lambda x: "{:.2f}".format(x))
                final_table.loc[anti,:]=m.str.cat(n, sep='±').values
                final_plot.loc[anti,:]=final_table_sub.loc['weighted-mean',:].to_list()
                final_std.loc[anti,:]=final_table_sub.loc['weighted-std',:].to_list()
            else:
                final_table_sub=pd.DataFrame(index=[' mean', ' std'],columns=score_list)
                # print(f1_list)
                f1_average=np.average(f1_list )
                accuracy_average=np.average(accuracy_list)
                f1_pos_average=np.average(f1_positive_list)
                f1_neg_average=np.average(f1_negative_list)


                final_table_sub.loc['mean',:] = [f1_average,accuracy_average,f1_pos_average, f1_neg_average]
                final_table_sub.loc['std',:] = [statistics.stdev(f1_list),
                                                statistics.stdev(accuracy_list),
                                                statistics.stdev(f1_positive_list),
                                               statistics.stdev(f1_negative_list)]

                m = final_table_sub.loc['mean',:].apply(lambda x: "{:.2f}".format(x))
                n = final_table_sub.loc['std',:].apply(lambda x: "{:.2f}".format(x))
                final_table.loc[anti,:]=m.str.cat(n, sep='±').values
                final_plot.loc[anti,:]=final_table_sub.loc['mean',:].to_list()
                final_std.loc[anti,:]=final_table_sub.loc['std',:].to_list()



        save_name_score,_ = name_utility.GETname_result('kover', species, '',f_kma,f_phylotree,chosen_cl,output_path)
        file_utility.make_dir(os.path.dirname(save_name_score))
        final_table.to_csv(save_name_score + '.txt', sep="\t")
        final_plot.to_csv(save_name_score + '_PLOT.txt', sep="\t")
        final_std.to_csv(save_name_score + '_std.txt', sep="\t")
        print(final_table)


def extract_info_species_clinical( level,species,cv,f_phylotree,f_kma,temp_path,output_path):
    ## score_list=['f1_macro','accuracy', 'f1_positive','f1_negative','precision_neg', 'recall_neg']
    ## score_list=['f1_macro','accuracy', 'f1_positive','f1_negative']
    score_list=['clinical_f1_negative','clinical_precision_neg', 'clinical_recall_neg']
    antibiotics, _, _ =  load_data.extract_info(species, False, level)

    for chosen_cl in ['scm','tree']:

        final_table = pd.DataFrame(index=antibiotics,columns=score_list)
        for anti in antibiotics:
            print(anti)

            y_true=[]
            y_pre=[]

            _,name,meta_txt,_ = name_utility.GETname_model2('kover',level, species, anti,'',temp_path,f_kma,f_phylotree)
            name_list = pd.read_csv(name, index_col=0, dtype={'genome_id': object}, sep="\t")
            name_list.loc[:,'ID'] = 'iso_' + name_list['genome_id'].astype(str)
            name_list2 = name_list.loc[:, ['ID', 'resistant_phenotype']]
            for outer_cv in range(cv):
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

            df=classification_report(y_true, y_pre, labels=[0, 1], output_dict=True,zero_division=0)
            report = pd.DataFrame(df).transpose()
            f1_negative=report.loc['0', 'f1-score']
            precision_neg=report.loc['0', 'precision']
            recall_neg=report.loc['0', 'recall']
            final_table.loc[anti,:]=[f1_negative,precision_neg,recall_neg]

        save_name_score,_ = name_utility.GETname_result('kover', species, '',f_kma,f_phylotree,chosen_cl,output_path)
        file_utility.make_dir(os.path.dirname(save_name_score))
        final_table.to_csv(save_name_score + '_clinical.txt', sep="\t")

        print(final_table)

def extract_best_estimator(level,species,fscore,f_phylotree,f_kma,output_path):
    '''
    for each species
    final_score:the score used for classifiers comparison.
    '''
    # score_list=['f1_macro','accuracy', 'f1_positive','f1_negative','precision_neg', 'recall_neg']
    score_list=['f1_macro','accuracy', 'f1_positive','f1_negative']
    antibiotics, ID, Y =  load_data.extract_info(species, False, level)
    cl_list = ['scm','tree']

    '''e.g. 1. summery_benchmarking
    
    'antibiotic'|'f1_macro'|'accuracy'| 'f1_positive'|'f1_negative'|'classifier'|'selected hyperparameter'|'frequency'
    
    'selected hyperparameter':
    {'canonical': True, 'cutting': 0.5, 'kmer': 6, 'odh': False, 'pca': False}
    2. [High level]for each species
    summary_table
      |SVM|Logistic Regression|Random Forest
    ceftazidime|0.85±0.20|...
    ciprofloxacin|  
    
        
    # hyper_table
    # |SVM|Logistic Regression|Random Forest
    # ceftazidime||...
    # ciprofloxacin| 
       
    '''

    summary_table = pd.DataFrame(index=antibiotics,columns=cl_list)
    summary_table_mean = pd.DataFrame(index=antibiotics, columns=cl_list)
    summary_table_std = pd.DataFrame(index=antibiotics, columns=cl_list)
    summary_benchmarking=pd.DataFrame(index=antibiotics,columns=score_list+['classifier', 'classifier_bymean'])

    summary_benchmarking_plot=pd.DataFrame(index=antibiotics,columns=score_list+['classifier'])
    summary_benchmarking_std=pd.DataFrame(index=antibiotics,columns=score_list+['classifier'])
    for anti in antibiotics:
        for chosen_cl in cl_list:

            score_,_ = name_utility.GETname_result('kover', species, fscore,f_kma,f_phylotree,chosen_cl,output_path)
            score_sub=pd.read_csv(score_ + '.txt', header=0, index_col=0,sep="\t")
            score_sub_mean = pd.read_csv(score_ + '_PLOT.txt', header=0, index_col=0, sep="\t")
            score_sub_std = pd.read_csv(score_ + '_std.txt', header=0, index_col=0, sep="\t")
            if f_kma:
                final_score_='weighted-'+fscore
            else:
                final_score_=fscore

            summary_table.loc[anti,chosen_cl]=score_sub.loc[anti,final_score_]
            summary_table_mean.loc[anti, chosen_cl] = score_sub_mean.loc[anti, final_score_]
            summary_table_std.loc[anti, chosen_cl] = score_sub_std.loc[anti, final_score_]


    _,save_name_final = name_utility.GETname_result('kover', species, fscore,f_kma,f_phylotree,'',output_path)
    file_utility.make_dir(os.path.dirname(save_name_final))
    summary_table.to_csv(save_name_final + '_SummaryClassifier.txt', sep="\t")

    summary_table_mean=summary_table_mean.astype(float)
    summary_table_std=summary_table_std.astype(float)
    # print(summary_table_mean)
    # print(summary_table_std)
    #-----------------------------------------------------------------------------------------------------------
    #choose the best estimator according to 1) mean. 2) std.


    cl_temp = [summary_table_mean.columns[i].tolist() for i in summary_table_mean.values == summary_table_mean.max(axis=1)[:,None]]
    summary_benchmarking['classifier_bymean']=cl_temp

    for index, row in summary_benchmarking.iterrows():
        std_list=[summary_table_std.loc[index,each] for each in row['classifier_bymean']]
        cl_chose_sub=std_list.index(min(std_list))
        row['classifier']=row['classifier_bymean'][cl_chose_sub]

    # summary_benchmarking['classifier']=summary_table_mean.idxmax(axis=1)
    print(summary_benchmarking)
    for anti in antibiotics:
        chosen_cl=summary_benchmarking.loc[anti,'classifier']
        ### print(chosen_cl)
        ### print('log/results/'+fscore+'/' +str(species.replace(" ", "_"))+'_kma_'+str(f_kma)+'_tree_'+str(f_phylotree)+'_'+chosen_cl)
        score_,_ = name_utility.GETname_result('kover', species, '',f_kma,f_phylotree,chosen_cl,output_path)
        score_sub = pd.read_csv(score_ + '.txt', header=0, index_col=0, sep="\t")
        score_sub_plot = pd.read_csv(score_ + '_PLOT.txt', header=0, index_col=0, sep="\t")
        score_sub_std= pd.read_csv(score_ + '_std.txt', header=0, index_col=0, sep="\t")
        ### summary_benchmarking.loc[anti,'selected hyperparameter']=score_sub.loc[anti,'selected hyperparameter']
        ### summary_benchmarking.loc[anti, 'frequency(out of 10)'] = score_sub.loc[anti, 'frequency']
        if f_kma:

            summary_benchmarking.loc[anti, score_list] = score_sub.loc[anti, ['weighted-'+x for x in score_list]].to_list()
            summary_benchmarking_plot.loc[[anti], score_list] = score_sub_plot.loc[
                anti, ['weighted-'+x for x in score_list]].to_list()
            summary_benchmarking_std.loc[[anti], score_list] = score_sub_std.loc[
                anti,['weighted-'+x for x in score_list]].to_list()

        else:
            summary_benchmarking.loc[anti, score_list] = score_sub.loc[anti, score_list].to_list()
            summary_benchmarking_plot.loc[[anti], score_list] = score_sub_plot.loc[anti, score_list].to_list()
            summary_benchmarking_std.loc[[anti], score_list] = score_sub_std.loc[anti, score_list].to_list()
    print(summary_benchmarking)
    summary_benchmarking.to_csv(save_name_final + '_SummaryBenchmarking.txt', sep="\t")
    summary_benchmarking_plot.to_csv(save_name_final + '_SummaryBenchmarking_PLOT.txt', sep="\t")
    summary_benchmarking_std.to_csv(save_name_final + '_SummaryBenchmarking_std.txt', sep="\t")


def extract_info(level,s,f_all,cv,fscore,f_phylotree,f_kma,temp_path,output_path):
    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    if f_all == False:
        data = data.loc[s, :]
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()


    for df_species, antibiotics in zip(df_species, antibiotics):
        if "clinical_" in fscore:
            extract_info_species_clinical(level, df_species, cv,f_phylotree,f_kma,temp_path,output_path)
            extract_best_estimator_clinical('kover',['scm','tree'],level, df_species,fscore,f_phylotree,f_kma,output_path)

        else:
            extract_info_species(level, df_species, cv,f_phylotree,f_kma,temp_path,output_path)
            extract_best_estimator(level, df_species,fscore,f_phylotree,f_kma,output_path)




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
                        help='the score used to choose the best classifier for each antibiotic. Can be one of: '
                             '\'f1_macro\',\'f1_positive\',\'f1_negative\',\'accuracy\','
                             '\'clinical_f1_negative\',\'clinical_precision_neg\',\'clinical_recall_neg\'')
    parser.add_argument('-f_ml', '--f_ml', dest='f_ml', action='store_true',
                        help='prepare bash')
    parser.add_argument("-cv", "--cv", default=10, type=int,
                        help='CV splits number')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'Pseudomonas aeruginosa\' \
                    \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
                    \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    # parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parsedArgs = parser.parse_args()
    extract_info( parsedArgs.level,parsedArgs.species, parsedArgs.f_all,parsedArgs.cv,parsedArgs.fscore,parsedArgs.f_phylotree,
                 parsedArgs.f_kma,parsedArgs.temp_path,parsedArgs.output_path)
