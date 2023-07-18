

import sys,os
# sys.path.append('../')
sys.path.insert(0, os.getcwd())
from src.amr_utility import name_utility, file_utility, load_data
from src.analysis_utility.lib import extract_score,make_table,math_utility
import os
import argparse
import pickle, json
import pandas as pd
import numpy as np
import statistics



def get_mean_std(f1_pos_sub):
    '''
    :param f1_pos_sub: a list of scores from CV
    :return: mean and std in a str format
    '''

    f1_pos_mean = statistics.mean(f1_pos_sub)
    f1_pos_std = statistics.stdev(f1_pos_sub)
    f1_pos_m_s = str(round(f1_pos_mean,2))+'±'+str(round(f1_pos_std,2))
    return f1_pos_m_s

def extract_info_species_clinical2(softwareName,chosen_cl,level,species,cv,f_phylotree,f_kma, temp_path):
    antibiotics, _, _ =  load_data.extract_info(species, False, level)
    score_list=['clinical_f1_negative','clinical_precision_neg', 'clinical_recall_neg']


    summary_table_ByClassifier_all = []
    for anti in antibiotics:
        print(species,anti)
        _,_ ,save_name_score= name_utility.GETname_model(softwareName,level, species, anti,chosen_cl,temp_path)
        with open(save_name_score + '_KMA_' + str(f_kma) + '_Tree_' + str(f_phylotree) + '.json') as f:
            score = json.load(f)
        summary_table_ByClassifier_ = pd.DataFrame(index=['value'],columns=score_list)
        score_report_test=score['score_report_test']
        # if anti=='amoxicillin/clavulanic acid':
        #     print(score['ture_Y'][3])
        #     print(score['predictY_test'][3])

        if score_report_test[0] != None:
            summary_table_ByClassifier =  extract_score.score_clinical(summary_table_ByClassifier_, cv, score_report_test)
        else:
            summary_table_ByClassifier=summary_table_ByClassifier_
        summary_table_ByClassifier_all.append(summary_table_ByClassifier)

    final =  make_table.make_visualization_clinical(score_list, summary_table_ByClassifier_all, antibiotics)
    return final

def extract_info_species2(softwareName,cl_list,level,species,cv,f_phylotree,f_kma, temp_path, output_path):
    '''
    Usage for : resfinder_folds & majority. No need to select the best classifier for reporting.
    [Low level]for each species and each classifier
    summary_table_ByClassifier
    |'f1_macro'| 'precision_macro'|'recall_macro'|'accuracy'|
    'mcc'|'f1_positive'| 'f1_negative'| 'precision_positive'|'recall_positive'|'auc'|'selected hyperparameter'
    ceftazidime|0.85±0.20|...
    ciprofloxacin|
    '''
    out_score='f' #['f1_macro','f1_positive', 'f1_negative','accuracy']
    ## out_score='neg' #[f1_macro', 'f1_positive','f1_negative','precision_neg', 'recall_neg', 'accuracy']

    antibiotics, _, _ =  load_data.extract_info(species, False, level)
    chosen_cl=cl_list[0]
    print('---------------------',chosen_cl)

    summary_table_ByClassifier_all = []
    for anti in antibiotics:

        _,_ ,save_name_score= name_utility.GETname_model(softwareName,level, species, anti,chosen_cl,temp_path)

        file_utility.make_dir(os.path.dirname(save_name_score))
        with open(save_name_score + '_KMA_' + str(f_kma) + '_Tree_' + str(f_phylotree) + '.json') as f:
            score = json.load(f)
        f1_test=score['f1_test']
        score_report_test=score['score_report_test']
        aucs_test=score['aucs_test']
        mcc_test=score['mcc_test']

        ## score2= pickle.load(open(save_name_score + '_kma_' + str(f_kma) + '_tree_' + str(f_phylotree) + '.pickle',"rb"))
        ## hyperparameters_test=score2['hyperparameters_test']
        summary_table_ByClassifier_ = pd.DataFrame(index=['mean', 'std', 'weighted-mean', 'weighted-std'],
                                   columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy',
                                            'mcc', 'f1_positive', 'f1_negative', 'precision_neg', 'recall_neg',
                                            'auc','threshold', 'support', 'support_positive'])
        if f1_test[0]!=None:

            if f_kma:# extract infor from report
                summary_table_ByClassifier=  extract_score.score_summary(None, summary_table_ByClassifier_, cv, score_report_test, f1_test,aucs_test,
                                                                           mcc_test,
                                                                           np.zeros(cv))# the last 0: no meaning.
            else:# f_phylotree or random

                summary_table_ByClassifier =  extract_score.score_summary_Tree(None, summary_table_ByClassifier_, cv, score_report_test,
                                                                                f1_test,aucs_test, mcc_test,
                                                                            np.zeros(cv))# the last 0: no meaning.
        else:
            summary_table_ByClassifier = summary_table_ByClassifier_

        summary_table_ByClassifier_all.append(summary_table_ByClassifier)


        # print(summary_table_ByClassifier)

    #finish one chosen_cl
    #[Low level]for each species and each classifier

    if f_kma:
        final, final_plot,final_std =  make_table.make_visualization(out_score, summary_table_ByClassifier_all, antibiotics)
    else:#if f_phylotree or random
        final, final_plot,final_std =  make_table.make_visualization_Tree(out_score, summary_table_ByClassifier_all, antibiotics)




    _,save_name_score_final = name_utility.GETname_result(softwareName,species,'',f_kma,f_phylotree,chosen_cl,output_path)
    file_utility.make_dir(os.path.dirname(save_name_score_final))
    final_plot=final_plot.rename(columns={"weighted-f1_macro": "f1_macro", "weighted-f1_positive": "f1_positive",
                                          "weighted-f1_negative": "f1_negative", "weighted-accuracy": "accuracy",
                                          "weighted-precision_neg": "precision_neg" , "weighted-recall_neg": "recall_neg"})
    final=final.rename(columns={"weighted-f1_macro": "f1_macro", "weighted-f1_positive": "f1_positive",
                                "weighted-f1_negative": "f1_negative", "weighted-accuracy": "accuracy",
                                "weighted-precision_neg": "precision_neg" , "weighted-recall_neg": "recall_neg"})
    final_std=final_std.rename(columns={"weighted-f1_macro": "f1_macro", "weighted-f1_positive": "f1_positive",
                                        "weighted-f1_negative": "f1_negative", "weighted-accuracy": "accuracy",
                                        "weighted-precision_neg": "precision_neg" , "weighted-recall_neg": "recall_neg" })


    ##### Add clinical oriented scores ['clinical_f1_negative','clinical_precision_neg', 'clinical_recall_neg'] to the main tables and PLOT table
    ##### Nov 2022

    clinical_table=extract_info_species_clinical2(softwareName,chosen_cl,level,species,cv,f_phylotree,f_kma, temp_path)
    final = pd.concat([final, clinical_table], axis=1, join="inner")
    final_plot = pd.concat([final_plot, clinical_table], axis=1, join="inner")
    #############################################################################################################################################


    final.to_csv(save_name_score_final + '_SummaryBenchmarking.txt', sep="\t")
    final_plot.to_csv(save_name_score_final + '_SummaryBenchmarking_PLOT.txt', sep="\t")
    final_std.to_csv(save_name_score_final + '_SummaryBenchmarking_std.txt', sep="\t")
    print(final)



def extract_info_species(softwareName,cl_list,level,species,cv,f_phylotree,f_kma, temp_path, output_path):
    '''

    [Low level]for each species and each classifier
    summary_table_ByClassifier
    |'f1_macro'| 'precision_macro'|'recall_macro'|'accuracy'|
    'mcc'|'f1_positive'| 'f1_negative'| 'precision_positive'|'recall_positive'|'auc'|'selected hyperparameter'
    ceftazidime|0.85±0.20|...
    ciprofloxacin|
    '''
    out_score='f' #['f1_macro','f1_positive', 'f1_negative','accuracy']
    ##out_score='neg' #[f1_macro', 'f1_positive','f1_negative','precision_neg', 'recall_neg', 'accuracy']

    antibiotics, _, _ =  load_data.extract_info(species, False, level)

    ### cl_list = ['svm', 'lr','rf']
    for chosen_cl in cl_list:
        print('---------------------',chosen_cl)
        hy_para_fre=[]
        hy_para_fren = []
        hy_para_all=[]
        summary_table_ByClassifier_all = []
        for anti in antibiotics:

            _,_ ,save_name_score= name_utility.GETname_model(softwareName,level, species, anti,chosen_cl,temp_path)

            try: #new version of format json
                with open(save_name_score + '_KMA_' + str(f_kma) + '_Tree_' + str(f_phylotree) + '.json') as f:
                    score = json.load(f)
                score2= pickle.load(open(save_name_score + '_kma_' + str(f_kma) + '_tree_' + str(f_phylotree) + '_model.pickle',"rb"))
                summary_table_ByClassifier_ = pd.DataFrame(index=['mean', 'std', 'weighted-mean', 'weighted-std'],
                                       columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy',
                                                'mcc', 'f1_positive', 'f1_negative', 'precision_neg', 'recall_neg',
                                                'auc','threshold', 'support', 'support_positive'])

                try:# not for MT

                    f1_test=score['f1_test']
                    score_report_test=score['score_report_test']
                    aucs_test=score['aucs_test']
                    mcc_test=score['mcc_test']
                    hyperparameters_test=score2['hyperparameters_test']

                    common,ind =  math_utility.get_most_fre_hyper(hyperparameters_test,True)
                    hy_para_fre.append(common.to_dict())
                    hy_para_fren.append(ind)
                    hy_para_all.append(hyperparameters_test)

                    if f_kma:# extract infor from report
                        summary_table_ByClassifier=  extract_score.score_summary(None, summary_table_ByClassifier_, cv, score_report_test, f1_test,aucs_test,
                                                                                   mcc_test,
                                                                                   np.zeros(cv))# the last 0: no meaning.
                    else:# f_phylotree or random
                        summary_table_ByClassifier =  extract_score.score_summary_Tree(None, summary_table_ByClassifier_, cv, score_report_test,
                                                                                        f1_test,aucs_test, mcc_test,
                                                                                    np.zeros(cv))# the last 0: no meaning.
                except:#only for MT
                    summary_table_ByClassifier = summary_table_ByClassifier_
                    hy_para_fre.append(None)
                    hy_para_fren.append(None)
                    hy_para_all.append(None)

            except: #old version format pickle--------------------------------------------------------------------------------------------------------------

                summary_table_ByClassifier_ = pd.DataFrame(index=['mean', 'std', 'weighted-mean', 'weighted-std'],
                                       columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy',
                                                'mcc', 'f1_positive', 'f1_negative', 'precision_neg', 'recall_neg',
                                                'auc','threshold', 'support', 'support_positive'])
                try:# not for MT
                    score = pickle.load(open(save_name_score + '_kma_' + str(f_kma) + '_tree_' + str(f_phylotree) + '.pickle',"rb"))
                    [f1_test, score_report_test, aucs_test, mcc_test, hyperparameters_test]=score
                    common,ind =  math_utility.get_most_fre_hyper(hyperparameters_test,True)
                    hy_para_fre.append(common.to_dict())
                    hy_para_fren.append(ind)
                    hy_para_all.append(hyperparameters_test)

                    if f_kma:# extract infor from report
                        summary_table_ByClassifier=  extract_score.score_summary(None, summary_table_ByClassifier_, cv, score_report_test, f1_test,aucs_test,
                                                                                       mcc_test,
                                                                                       np.zeros(cv))# the last 0: no meaning.
                    else:# f_phylotree or random
                        summary_table_ByClassifier =  extract_score.score_summary_Tree(None, summary_table_ByClassifier_, cv, score_report_test,
                                                                                            f1_test,aucs_test, mcc_test,
                                                                                        np.zeros(cv))# the last 0: no meaning.

                except:#only for MT

                    summary_table_ByClassifier = summary_table_ByClassifier_
                    hy_para_fre.append(None)
                    hy_para_fren.append(None)
                    hy_para_all.append(None)
            ########-------------------------------------------------------------------------------------------------------------------------------------------
            summary_table_ByClassifier_all.append(summary_table_ByClassifier)


        #finish one chosen_cl
        #[Low level]for each species and each classifier

        if f_kma:
            final, final_plot,final_std =  make_table.make_visualization(out_score, summary_table_ByClassifier_all, antibiotics)
        else:#if f_phylotree or random
            final, final_plot,final_std =  make_table.make_visualization_Tree(out_score, summary_table_ByClassifier_all, antibiotics)

        save_name_score_final,_ = name_utility.GETname_result(softwareName, species, '',f_kma,f_phylotree,chosen_cl,output_path)
        file_utility.make_dir(os.path.dirname(save_name_score_final))
        final['selected hyperparameter'] = hy_para_fre
        final['frequency(out of 10)'] = hy_para_fren
        final['hyperparameter sets'] = hy_para_all
        final.to_csv(save_name_score_final + '.txt', sep="\t")
        final_plot.to_csv(save_name_score_final + '_PLOT.txt', sep="\t")
        final_std.to_csv(save_name_score_final + '_std.txt', sep="\t")
        print(final)

def extract_info_species_clinical(softwareName,cl_list,level,species,cv,f_phylotree,f_kma, temp_path, output_path):
    antibiotics, _, _ =  load_data.extract_info(species, False, level)
    score_list=['clinical_f1_negative','clinical_precision_neg', 'clinical_recall_neg']
    # out_score='neg'
    for chosen_cl in cl_list:
        print('---------------------',chosen_cl)

        summary_table_ByClassifier_all = []
        for anti in antibiotics:

            _,_ ,save_name_score= name_utility.GETname_model(softwareName,level, species, anti,chosen_cl,temp_path)
            summary_table_ByClassifier_ = pd.DataFrame(index=['value'],columns=score_list)
            try: #new version
                with open(save_name_score + '_KMA_' + str(f_kma) + '_Tree_' + str(f_phylotree) + '.json') as f:
                    score = json.load(f)

                try:# not for MT
                    score_report_test=score['score_report_test']
                    summary_table_ByClassifier =  extract_score.score_clinical(summary_table_ByClassifier_, cv, score_report_test)
                except:#only for MT
                    summary_table_ByClassifier = summary_table_ByClassifier_

            except: #old version-----------------------------------------------------------------------------------------------------------------------------


                try:# if not for MT
                    score = pickle.load(open(save_name_score + '_kma_' + str(f_kma) + '_tree_' + str(f_phylotree) + '.pickle',"rb"))
                    [f1_test, score_report_test, aucs_test, mcc_test, hyperparameters_test]=score
                    summary_table_ByClassifier =  extract_score.score_clinical(summary_table_ByClassifier_, cv, score_report_test)

                except:#only for MT
                    summary_table_ByClassifier = summary_table_ByClassifier_
            ####-------------------------------------------------------------------------------------------------------------------------------------------

            summary_table_ByClassifier_all.append(summary_table_ByClassifier)
            # print(summary_table_ByClassifier)

        #finish one chosen_cl
        final =  make_table.make_visualization_clinical(score_list, summary_table_ByClassifier_all, antibiotics)
        save_name_score_final,_ = name_utility.GETname_result(softwareName, species, '',f_kma,f_phylotree,chosen_cl,output_path)
        file_utility.make_dir(os.path.dirname(save_name_score_final))
        final.to_csv(save_name_score_final + '_clinical.txt', sep="\t")

        print(final)




def extract_best_estimator(softwareName,cl_list,level,species,fscore,f_phylotree,f_kma,  output_path):
    '''
    for each species output:
    e.g. 1. summary_benchmarking

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
    score_list=['f1_macro','accuracy', 'f1_positive','f1_negative']


    antibiotics, _, _ =  load_data.extract_info(species, False, level)
    ### cl_list = ['svm', 'lr','rf']

    summary_table = pd.DataFrame(index=antibiotics,columns=cl_list)
    summary_table_mean = pd.DataFrame(index=antibiotics, columns=cl_list)
    summary_table_std = pd.DataFrame(index=antibiotics, columns=cl_list)
    summary_benchmarking=pd.DataFrame(index=antibiotics,columns=score_list+['classifier', 'classifier_bymean','hyperparameter sets','selected hyperparameter','frequency(out of 10)'])

    summary_benchmarking_plot=pd.DataFrame(index=antibiotics,columns=score_list+['classifier'])
    summary_benchmarking_std=pd.DataFrame(index=antibiotics,columns=score_list+['classifier'])
    for anti in antibiotics:
        for chosen_cl in cl_list:

            score_,_ = name_utility.GETname_result(softwareName, species,'', f_kma,f_phylotree,chosen_cl,output_path)

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

    _,save_name_final = name_utility.GETname_result(softwareName, species, fscore,f_kma,f_phylotree,'',output_path)

    file_utility.make_dir(os.path.dirname(save_name_final))
    summary_table.to_csv(save_name_final + '_SummaryClassifier.txt', sep="\t")


    summary_table_mean=summary_table_mean.astype(float)
    summary_table_std=summary_table_std.astype(float)

    cl_temp = [summary_table_mean.columns[i].tolist() for i in summary_table_mean.values == summary_table_mean.max(axis=1)[:,None]]
    summary_benchmarking['classifier_bymean']=cl_temp

    for index, row in summary_benchmarking.iterrows():
        #if there are several classifiers with the same highest fscore, then we select those with the lowest standard deviation.
        # if several with the same highest fscore and same lowest standard deviation, then select the first classifier in the list
        std_list=[summary_table_std.loc[index,each] for each in row['classifier_bymean']]
        try:
            cl_chose_sub=std_list.index(min(std_list))
            row['classifier']=row['classifier_bymean'][cl_chose_sub]
        except:#MT issues.
            row['classifier']=np.nan

    print(summary_benchmarking)

    if species =='Mycobacterium tuberculosis':#MT issues.
        # antibiotics=['amikacin','capreomycin','ethiomide','ethionamide','kanamycin','ofloxacin','rifampin','streptomycin']
        antibiotics=['amikacin','capreomycin','ethiomide','ethionamide','kanamycin','ofloxacin','streptomycin']
    for anti in antibiotics:
        chosen_cl=summary_benchmarking.loc[anti,'classifier']

        score_,_ = name_utility.GETname_result(softwareName, species, '',f_kma,f_phylotree,chosen_cl,output_path)
        score_sub = pd.read_csv(score_ + '.txt', header=0, index_col=0, sep="\t")
        score_sub_plot = pd.read_csv(score_ + '_PLOT.txt', header=0, index_col=0, sep="\t")
        score_sub_std= pd.read_csv(score_ + '_std.txt', header=0, index_col=0, sep="\t")
        if f_kma:

            summary_benchmarking.loc[anti, score_list] = score_sub.loc[anti, ['weighted-'+x for x in score_list]].to_list()
            summary_benchmarking_plot.loc[[anti], score_list] = score_sub_plot.loc[
                anti, ['weighted-'+x for x in score_list]].to_list()
            summary_benchmarking_std.loc[[anti], score_list] = score_sub_std.loc[
                anti,['weighted-'+x for x in score_list]].to_list()

        else:
            summary_benchmarking.loc[anti, score_list+['hyperparameter sets','selected hyperparameter','frequency(out of 10)']] = score_sub.loc[
                anti,score_list+ ['hyperparameter sets','selected hyperparameter','frequency(out of 10)']].to_list()
            summary_benchmarking_plot.loc[[anti], score_list] = score_sub_plot.loc[
                anti, score_list].to_list()
            summary_benchmarking_std.loc[[anti], score_list] = score_sub_std.loc[
                anti, score_list].to_list()
    print(summary_benchmarking)
    summary_benchmarking.to_csv(save_name_final + '_SummaryBenchmarking.txt', sep="\t")
    summary_benchmarking_plot.to_csv(save_name_final + '_SummaryBenchmarking_PLOT.txt', sep="\t")
    summary_benchmarking_std.to_csv(save_name_final + '_SummaryBenchmarking_std.txt', sep="\t")


def extract_best_estimator_clinical(softwareName,cl_list,level,species,fscore,f_phylotree,f_kma,  output_path):
    score_list=['clinical_f1_negative','clinical_precision_neg', 'clinical_recall_neg']
    antibiotics, _, _ =  load_data.extract_info(species, False, level)
    ### cl_list = ['svm', 'lr','rf']

    summary_table = pd.DataFrame(index=antibiotics,columns=cl_list)
    summary_benchmarking=pd.DataFrame(index=antibiotics,columns=score_list+['classifier'])


    for anti in antibiotics:
        for chosen_cl in cl_list:

            score_,_ = name_utility.GETname_result(softwareName, species,'', f_kma,f_phylotree,chosen_cl,output_path)
            score_sub=pd.read_csv(score_ + '_clinical.txt', header=0, index_col=0,sep="\t")
            final_score_=fscore
            summary_table.loc[anti,chosen_cl]=score_sub.loc[anti,final_score_]



    _,save_name_final = name_utility.GETname_result(softwareName, species, fscore,f_kma,f_phylotree,'',output_path)
    file_utility.make_dir(os.path.dirname(save_name_final))
    summary_table.to_csv(save_name_final + '_SummaryClassifier.txt', sep="\t")


    summary_table=summary_table.astype(float)
    summary_benchmarking['classifier']=summary_table.idxmax(axis=1)#choose the best estimator according to summary_table


    # print(summary_benchmarking)
    for anti in antibiotics:
        chosen_cl=summary_benchmarking.loc[anti,'classifier']
        try:#MT cases in Phenotyperseeker
            score_,_ = name_utility.GETname_result(softwareName, species, '',f_kma,f_phylotree,chosen_cl,output_path)
            score_sub = pd.read_csv(score_ + '_clinical.txt', header=0, index_col=0, sep="\t")
            summary_benchmarking.loc[anti, score_list] = score_sub.loc[anti,score_list ].to_list()
        except:
            pass
    print(summary_benchmarking)
    summary_benchmarking.to_csv(save_name_final + '_SummaryBenchmarking.txt', sep="\t")


def extract_info(softwareName,cl_list,level,s,f_all,cv,fscore,f_phylotree,f_kma, temp_path, output_path):
    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]
    if f_all == False:
        data = data.loc[s, :]
    df_species = data.index.tolist()

    for species in  df_species:
        print(cl_list)
        if len(cl_list)>1: # phenotypeseeker & s2g2p
            if "clinical_" in fscore:
                extract_info_species_clinical(softwareName,cl_list,level, species, cv,f_phylotree,f_kma ,temp_path, output_path)
                extract_best_estimator_clinical(softwareName,cl_list,level, species, fscore,f_phylotree,f_kma, output_path)
            else:
                extract_info_species(softwareName,cl_list,level, species, cv,f_phylotree,f_kma ,temp_path, output_path)
                extract_best_estimator(softwareName,cl_list,level, species, fscore,f_phylotree,f_kma, output_path)
        else:# resfinder_folds & majority, ensemble. No need to select the best classifier for reporting.
            extract_info_species2(softwareName,cl_list,level, species, cv,f_phylotree,f_kma ,temp_path, output_path)



if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-software', '--softwareName', type=str, required=True,
                        help='Software name.')
    parser.add_argument('-cl_list', '--cl_list', default=[], type=str, nargs='+',
                        help='classifiers.')
    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                        help='Directory to store temporary files.')
    parser.add_argument('-out', '--output_path', default='./', type=str, required=False,
                        help='Directory to store CV scores.')
    parser.add_argument('-l', '--level', default='loose', type=str, required=False,
                        help='Quality control: strict or loose')
    parser.add_argument("-cv", "--cv_number", default=10, type=int,
                        help='CV splits number')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
         \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
         \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('-fscore', '--fscore', default='f1_macro', type=str, required=False,
                        help='the score used to choose the best classifier for each antibiotic. Can be one of: \'f1_macro\','
                             '\'f1_positive\',\'f1_negative\',\'accuracy\',\'clinical_f1_negative\',\'clinical_precision_neg\',\'clinical_recall_neg\'')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                        help='kma based cv folders.')

    parsedArgs = parser.parse_args()
    # parser.print_help()
    extract_info(parsedArgs.softwareName,parsedArgs.cl_list,parsedArgs.level,parsedArgs.species,parsedArgs.f_all,parsedArgs.cv_number,parsedArgs.fscore,parsedArgs.f_phylotree,
                 parsedArgs.f_kma,parsedArgs.temp_path,parsedArgs.output_path)
