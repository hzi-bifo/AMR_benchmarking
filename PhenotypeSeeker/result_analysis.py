import sys
import os
sys.path.insert(0, os.path.abspath('../'))
import amr_utility.name_utility
import amr_utility.load_data
import analysis_results.extract_score
import analysis_results.make_table
import analysis_results.math_utility
import ast
import argparse
import pickle
import pandas as pd
import numpy as np
import statistics
from scipy.stats import ttest_rel
import math
import seaborn as sns


def get_mean_std(f1_pos_sub):
    '''
    :param f1_pos_sub: a list of scores from CV
    :return: mean and std in a str format
    '''
    f1_pos_mean = statistics.mean(f1_pos_sub)
    f1_pos_std = statistics.stdev(f1_pos_sub)
    f1_pos_m_s = str(round(f1_pos_mean,2))+'±'+str(round(f1_pos_std,2))
    return f1_pos_m_s

def extract_info_species(level,species,final_score,antibiotics,cv,f_phylotree,f_kma,old_version):
    '''
    Sep 10th, 2021.
    [Low level]for each species and each classifier
    summary_table_ByClassifier
    |'f1_macro'| 'precision_macro'|'recall_macro'|'accuracy'|
    'mcc'|'f1_positive'| 'f1_negative'| 'precision_positive'|'recall_positive'|'auc'|'selected hyperparameter'
    ceftazidime|0.85±0.20|...
    ciprofloxacin|
    '''
    antibiotics_selected = ast.literal_eval(antibiotics)
    print(species)
    print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)
    antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
    # cl_list=['svm','lr','lsvm','rf','et','ab','gb','xgboost']
    cl_list = ['svm', 'lr','rf']
    for chosen_cl in cl_list:
        print('---------------------',chosen_cl)
        hy_para_fre=[]
        hy_para_fren = []
        summary_table_ByClassifier_all = []
        for anti in antibiotics:
            _, _, save_name_score = amr_utility.name_utility.Pts_GETname(level, species, anti,chosen_cl)
            # save_name_score = amr_utility.name_utility.GETsave_name_score(species, anti, chosen_cl)
            if old_version:
                score = pickle.load(open(save_name_score + '.pickle', "rb"))  # old version

                [f1_test, score_report_test, aucs_test, mcc_test, hyperparameters_test]=score
                hyperparameters_test_=[]
                for hyper in hyperparameters_test:# extract informarion w.r.t. feature keys.
                    hyper_ = dict((k, hyper[k]) for k in ('pca', 'canonical', 'cutting', 'kmer', 'odh'))
                    hyperparameters_test_.append(hyper_)
                common ,ind= analysis_results.math_utility.get_most_fre_hyper(hyperparameters_test_)
                hy_para_fre.append(common.to_dict())
                hy_para_fren.append(ind)

            else:
                # print(save_name_score)
                score = pickle.load(open(save_name_score + '_kma_' + str(f_kma) + '_tree_' + str(f_phylotree) + '.pickle',"rb"))  # todo,check
                [f1_test, score_report_test, aucs_test, mcc_test, hyperparameters_test]=score
                print(hyperparameters_test)
                # common,ind = analysis_results.math_utility.get_most_fre_hyper(hyperparameters_test)
                # hy_para_fre.append(common.to_dict())
                # hy_para_fren.append(ind)
            summary_table_ByClassifier_ = pd.DataFrame(index=['mean', 'std', 'weighted-mean', 'weighted-std'],
                                   columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy_macro',
                                            'mcc', 'f1_positive', 'f1_negative', 'precision_positive', 'recall_positive',
                                            'auc','threshold', 'support', 'support_positive'])
            if f_kma:# extract infor from report
                summary_table_ByClassifier= analysis_results.extract_score.score_summary(None, summary_table_ByClassifier_, cv, score_report_test, aucs_test,
                                                                           mcc_test, save_name_score,
                                                                           np.zeros(cv))# the last 0: no meaning.
            else:# f_phylotree or random
                #todo, still need check. should be fine.Sep 10.
                summary_table_ByClassifier = analysis_results.extract_score.score_summary_Tree(None, summary_table_ByClassifier_, cv, score_report_test,
                                                                                aucs_test, mcc_test,
                                                                                save_name_score,
                                                                                np.zeros(cv))# the last 0: no meaning.

            summary_table_ByClassifier_all.append(summary_table_ByClassifier)


            print(summary_table_ByClassifier)

        #finish one chosen_cl
        #[Low level]for each species and each classifier
        out_score='f' #['weighted-f1_macro','weighted-f1_positive', 'weighted-f1_negative','weighted-accuracy'
        if f_kma:
            final, final_plot = analysis_results.make_table.make_visualization(out_score, summary_table_ByClassifier_all, antibiotics,
                                                                                   level, None, None,
                                                                                   None,
                                                                                   None,
                                                                                   None)
        else:#if f_phylotree or random
            final, final_plot = analysis_results.make_table.make_visualization_Tree(out_score, summary_table_ByClassifier_all, antibiotics,
                                                                                        level,
                                                                                        None, None,
                                                                                        None,
                                                                                        None,
                                                                                        None)



        save_name_score_final,_ = amr_utility.name_utility.GETsave_name_final(species,f_kma,f_phylotree,chosen_cl)
        # print(final)
        # print(hyperparameters_test)
        # final['selected hyperparameter'] = hy_para_fre
        # final['frequency'] = hy_para_fren
        final.to_csv(save_name_score_final + '.txt', sep="\t")
        final_plot.to_csv(save_name_score_final + '_PLOT.txt', sep="\t")
        print(final)




def extract_best_estimator(level,species,final_score,antibiotics,cv,f_phylotree,f_kma):
    '''
    for each species
    final_score:the score used for classifiers comparison.
    '''
    antibiotics_selected = ast.literal_eval(antibiotics)

    print(species)
    print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)
    antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
    # cl_list=['svm','lr','lsvm','rf','et','ab','gb','xgboost']
    cl_list = ['svm', 'lr','rf']

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
    summary_table_ = pd.DataFrame(index=antibiotics, columns=cl_list)
    summary_benchmarking=pd.DataFrame(index=antibiotics,columns=['f1_macro','accuracy', 'f1_positive','f1_negative',
                                                                 'classifier','selected hyperparameter','frequency(out of 10)'])

    summary_benchmarking_plot=pd.DataFrame(index=antibiotics,columns=['f1_macro','accuracy', 'f1_positive','f1_negative','classifier'])

    for anti in antibiotics:
        for chosen_cl in cl_list:
            score_ ,_= amr_utility.name_utility.GETsave_name_final(species, f_kma, f_phylotree,chosen_cl)
            score_sub=pd.read_csv(score_ + '.txt', header=0, index_col=0,sep="\t")
            score_sub_ = pd.read_csv(score_ + '_PLOT.txt', header=0, index_col=0, sep="\t")
            if not f_phylotree:
                final_score_='weighted-'+final_score

            summary_table.loc[anti,chosen_cl]=score_sub.loc[anti,final_score_]
            summary_table_.loc[anti, chosen_cl] = score_sub_.loc[anti, final_score_]

    _, save_name_final = amr_utility.name_utility.GETsave_name_final(species, f_kma, f_phylotree, '')
    summary_table.to_csv(save_name_final + '_SummaryClassifier.txt', sep="\t")
    print('summary_table')
    print(summary_table)
    summary_table_=summary_table_.astype(float)
    # print(summary_table_.dtypes)
    #-----------------------------------------------------------------------------------------------------------
    #choose the best estimator according to summary_table
    summary_benchmarking['classifier']=summary_table_.idxmax(axis=1)
    for anti in antibiotics:
        chosen_cl=summary_benchmarking.loc[anti,'classifier']
        score_, _ = amr_utility.name_utility.GETsave_name_final(species, f_kma, f_phylotree, chosen_cl)
        score_sub = pd.read_csv(score_ + '.txt', header=0, index_col=0, sep="\t")
        score_sub_plot = pd.read_csv(score_ + '_PLOT.txt', header=0, index_col=0, sep="\t")
        # summary_benchmarking.loc[anti,'selected hyperparameter']=score_sub.loc[anti,'selected hyperparameter']
        # summary_benchmarking.loc[anti, 'frequency(out of 10)'] = score_sub.loc[anti, 'frequency']
        if not f_phylotree:

            summary_benchmarking.loc[anti, ['f1_macro','accuracy', 'f1_positive','f1_negative']] = score_sub.loc[anti, ['weighted-f1_macro','weighted-accuracy', 'weighted-f1_positive','weighted-f1_negative']].to_list()
            summary_benchmarking_plot.loc[[anti], ['f1_macro', 'accuracy', 'f1_positive', 'f1_negative']] = score_sub_plot.loc[
                anti, ['weighted-f1_macro','weighted-accuracy', 'weighted-f1_positive','weighted-f1_negative']].to_list()

        else:
            summary_benchmarking.loc[anti, ['f1_macro', 'accuracy', 'f1_positive', 'f1_negative']] = score_sub.loc[
                anti, ['f1_macro', 'accuracy', 'f1_positive', 'f1_negative']].to_list()
            summary_benchmarking_plot.loc[[anti], ['f1_macro', 'accuracy', 'f1_positive', 'f1_negative']] = score_sub_plot.loc[
                anti, ['f1_macro', 'accuracy', 'f1_positive', 'f1_negative']].to_list()
    print(summary_benchmarking)
    summary_benchmarking.to_csv(save_name_final + '_SummeryBenchmarking.txt', sep="\t")
    summary_benchmarking_plot.to_csv(save_name_final + '_SummeryBenchmarking_PLOT.txt', sep="\t")


def plot(level, species, final_score, antibiotics, cv, f_phylotree, f_kma,old_version):
    '''
    for each species.
    summary_plot
    final_score| 'antibiotic'|'classifier'
    ceftazidime|0.85|...
    ciprofloxacin|
    '''
    antibiotics_selected = ast.literal_eval(antibiotics)

    print(species)
    print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)
    antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
    # cl_list=['svm','lr','lsvm','rf','et','ab','gb','xgboost']
    cl_list = ['svm', 'lr', 'rf']

    summary_plot = pd.DataFrame(columns=[final_score, 'antibiotic', 'classifier'])


    for chosen_cl in cl_list:
        print('---------------------', chosen_cl)

        for anti in antibiotics:

            _, _, save_name_score = amr_utility.name_utility.Pts_GETname(level, species, anti,chosen_cl)
            if old_version:
                score = pickle.load(open(save_name_score + '.pickle', "rb"))  # old version
                [f1_test, score_report_test, aucs_test, mcc_test, hyperparameters_test]=score # old version

            else:
                score = pickle.load(
                    open(save_name_score + '_kma_' + str(f_kma) + '_tree_' + str(f_phylotree) + '.pickle',
                         "rb"))  # todo,check
                [f1_test, score_report_test, aucs_test, mcc_test, hyperparameters_test]=score
            for i in np.arange(cv):
                report = score_report_test[i]
                report = pd.DataFrame(report).transpose()


                if report.loc['1', 'support']!=0 and report.loc['0', 'support']!=0:#test set with only one phenotype
                    if final_score=='f1_positive':
                        f1_pos=report.loc['1', 'f1-score']
                        final_score_ = f1_pos
                    elif final_score=='f1_negative':
                        f1_neg=report.loc['0', 'f1-score']
                        final_score_ = f1_neg
                    elif final_score=='f1_macro':
                        f1_macro=report.loc['macro avg','f1-score']
                        final_score_ = f1_macro
                    elif final_score == 'accuracy':
                        accuracy=report.loc['accuracy', 'f1-score']
                        final_score_ = accuracy

                    summary_plot_sub = pd.DataFrame(columns=[final_score, 'antibiotic', 'classifier'])
                    summary_plot_sub.loc['e'] = [final_score_, anti, chosen_cl]
                    # summary_plot_sub[final_score] = final_score_
                    # summary_plot_sub['antibiotic'] = anti
                    # summary_plot_sub['classifier'] = chosen_cl
                    print(summary_plot_sub)
                    summary_plot = summary_plot.append(summary_plot_sub, sort=False)
                    print(summary_plot)


    _, save_name_final = amr_utility.name_utility.GETsave_name_final(species, f_kma, f_phylotree, '')

    # print(hyper_table)
    print(summary_plot)
    ax = sns.boxplot(x="antibiotic", y=final_score, hue="classifier",
                     data=summary_plot, dodge=True, width=0.4)
    fig = ax.get_figure()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, horizontalalignment='right',fontsize=10)
    fig.savefig(save_name_final + '_' + final_score + ".png")



def extract_info(l,s,score,cv,f_phylotree,f_kma,f_benchmarking,old_version,f_plot):
    data = pd.read_csv('metadata/' + str(l) + '_Species_antibiotic_FineQuality.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    # data = pd.read_csv('metadata/Species_antibiotic_FineQuality.csv', index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    # for training model on part of the dataset:-------------
    # data=data.loc[['Escherichia coli'],:]
    data = data.loc[s, :]
    # data.at['Mycobacterium tuberculosis', 'modelling antibiotics']=['capreomycin', 'ciprofloxacin']
    # --------------------------------------------------------
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    print(data)
    if not f_benchmarking and not f_plot:#basic. extract scores for each species, antibiotic, and estimator.
        for df_species, antibiotics in zip(df_species, antibiotics):
            extract_info_species(l, df_species, score, antibiotics, cv,f_phylotree,f_kma,old_version)
    elif f_plot:
        for df_species, antibiotics in zip(df_species, antibiotics):
            plot(l, df_species, score, antibiotics, cv, f_phylotree, f_kma,old_version)
    else:# extract the best estimator for kmer based banchmarking comparison.
        for df_species, antibiotics in zip(df_species, antibiotics):
            extract_best_estimator(l, df_species, score, antibiotics, cv,f_phylotree,f_kma)



if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--l', '--level', default='loose', type=str, required=False,
                        help='Quality control: strict or loose')
    parser.add_argument("-cv", "--cv_number", default=10, type=int,
                        help='CV splits number')
    # parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
         \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
         \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    # parser.add_argument('-cl', '--cl', default=['svm'], type=str, nargs='+',
    #                     help='classifier to train: e.g.\'svm\',\'lr\',\'lsvm\',\'rf\',\'et\',\'ab\',\'gb\',\'xgboost\'')
    parser.add_argument('-score', '--score', default='f1_macro', type=str, required=False,
                        help='the score used to choose the best classifier for each antibiotic. Can be f1_pos'
                             'f1_neg, accuracy.')
    parser.add_argument('-old_version', '--old_version', dest='old_version', action='store_true',
                        help='Old version of scripts. model_cv.py')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                        help='kma based cv folders.')
    parser.add_argument('-f_plot', '--f_plot', dest='f_plot', action='store_true',
                        help='plot.')
    parser.add_argument('-f_benchmarking', '--f_benchmarking', dest='f_benchmarking', action='store_true',
                        help='Extract the best estimator for benchmarking. First, run the script without this flag. Then, use this flag. ')
    # parser.add_argument('-score', '--score', default='f', type=str,required=False,
    #                     help='Scores of the final output table. f:f_macro,f_pos,f_neg,f_micro. all:all scores. f_p_r:f1_macro,precision,recall,accuracy')

    # parser.set_defaults(canonical=True)
    parsedArgs = parser.parse_args()
    # parser.print_help()
    print(parsedArgs)
    extract_info(parsedArgs.l,parsedArgs.species,parsedArgs.score,parsedArgs.cv_number,parsedArgs.f_phylotree,parsedArgs.f_kma,parsedArgs.f_benchmarking,parsedArgs.old_version,parsedArgs.f_plot)
