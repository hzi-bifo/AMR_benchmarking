

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

score_set=['f1_macro', 'f1_positive', 'f1_negative', 'accuracy',
        'precision_macro', 'recall_macro', 'precision_negative', 'recall_negative','precision_positive', 'recall_positive',
        'mcc',  'auc','threshold', 'support', 'support_positive','support_negative']

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
    score_list=['clinical_f1_negative','clinical_precision_negative', 'clinical_recall_negative']


    summary_table_ByClassifier_all = []
    for anti in antibiotics:
        print(species,anti)
        _,_ ,save_name_score= name_utility.GETname_model(softwareName,level, species, anti,chosen_cl,temp_path)
        with open(save_name_score + '_KMA_' + str(f_kma) + '_Tree_' + str(f_phylotree) + '.json') as f:
            score = json.load(f)
        summary_table_ByClassifier_ = pd.DataFrame(index=['value'],columns=score_list)
        score_report_test=score['score_report_test']

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
    '''


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
                                   columns=score_set)
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


    #finish one chosen_cl
    #[Low level]for each species and each classifier

    if f_kma:
        final, final_plot,final_std =  make_table.make_visualization(summary_table_ByClassifier_all, antibiotics)
    else:#if f_phylotree or random
        final, final_plot,final_std =  make_table.make_visualization_Tree( summary_table_ByClassifier_all, antibiotics)


    _,save_name_score_final = name_utility.GETname_result(softwareName,species,'',f_kma,f_phylotree,chosen_cl,output_path)
    file_utility.make_dir(os.path.dirname(save_name_score_final))

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

    antibiotics, _, _ =  load_data.extract_info(species, False, level)


    for chosen_cl in cl_list:
        print('---------------------',chosen_cl)
        hy_para_fre=[]
        hy_para_fren = []
        hy_para_all=[]
        summary_table_ByClassifier_all = []
        for anti in antibiotics:

            _,_ ,save_name_score= name_utility.GETname_model(softwareName,level, species, anti,chosen_cl,temp_path)

            summary_table_ByClassifier_ = pd.DataFrame(index=['mean', 'std', 'weighted-mean', 'weighted-std'],
                                       columns=score_set)

            try:# not for MT
                with open(save_name_score + '_KMA_' + str(f_kma) + '_Tree_' + str(f_phylotree) + '.json') as f:
                    score = json.load(f)
                score2= pickle.load(open(save_name_score + '_kma_' + str(f_kma) + '_tree_' + str(f_phylotree) + '_model.pickle',"rb"))

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


            ########-------------------------------------------------------------------------------------------------------------------------------------------
            summary_table_ByClassifier_all.append(summary_table_ByClassifier)


        #finish one chosen_cl
        #[Low level]for each species and each classifier

        if f_kma:
            final, final_plot,final_std =  make_table.make_visualization( summary_table_ByClassifier_all, antibiotics)
        else:#if f_phylotree or random
            final, final_plot,final_std =  make_table.make_visualization_Tree(summary_table_ByClassifier_all, antibiotics)

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
    score_list=['clinical_f1_negative','clinical_precision_negative', 'clinical_recall_negative']
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




def extract_best_estimator(softwareName,cl_list,level,species,fscore,cv,f_phylotree,f_kma, temp_path, output_path):
    '''

     Aug 2023:
     Select the best classifier in inner loop of nested CV
     update: no need to set separate folders for different metrics.
    '''
    if species =='Mycobacterium tuberculosis':#MT issues. only for PhenotypeSeeker.
        ### antibiotics=['amikacin','capreomycin','ethiomide','ethionamide','kanamycin','ofloxacin','rifampin','streptomycin']
        antibiotics_run=['amikacin','capreomycin','ethiomide','ethionamide','kanamycin','ofloxacin','streptomycin']

    antibiotics, _, _ =  load_data.extract_info(species, False, level)

    summary_table_ByClassifier_all=[]
    hy_para_fre=[]
    hy_para_fren = []
    hy_para_all=[]
    report_all=[] ##for clinical-oriented score extraction.

    classifier_selection=[] ## only for misclassifier.py usage. Added 7 Sep 2023.
    for anti in antibiotics:
        classifier_selection_sub=[]
        print(anti)

        summary_table_ByClassifier_ = pd.DataFrame(index=['mean', 'std', 'weighted-mean', 'weighted-std'],
                                   columns=score_set)


        if species=='Mycobacterium tuberculosis' and anti not in antibiotics_run: #MT issues. PhenotyperSeeker.
            summary_table_ByClassifier = summary_table_ByClassifier_
            hy_para_fre.append(None)
            hy_para_fren.append(None)
            hy_para_all.append(None)
            report_all.append(None)
            classifier_selection.append(None)
        else:
            score_report_test, f1_test,aucs_test, mcc_test=[],[],[],[]
            hyperparameters_test=[]
            for outer_cv in range(cv):

                MEAN=[]
                STD=[]
                CL=[]
                for cl_each in cl_list:

                    _,_ ,save_name_score= name_utility.GETname_model(softwareName,level, species, anti,cl_each,temp_path)
                    score2= pickle.load(open(save_name_score + '_kma_' + str(f_kma) + '_tree_' + str(f_phylotree) + '_model.pickle',"rb"))
                    score_InnerLoop=score2['score_InnerLoop'][outer_cv]
                    index_InnerLoop=score2['index_InnerLoop'][outer_cv]
                    cv_results_InnerLoop=score2['cv_results_InnerLoop'][outer_cv]
                    std_InnerLoop=cv_results_InnerLoop['std_test_score'][index_InnerLoop]
                    MEAN.append(score_InnerLoop)
                    STD.append(std_InnerLoop)
                    CL.append(cl_each)
                ##select the highest mean, resorting to the lowest std.
                combined = [(MEAN[i], STD[i]) for i in range(len(MEAN))]
                # Sort the list of tuples first by a in descending order and then by b in ascending order
                sorted_combined = sorted(combined, key=lambda x: (-x[0], x[1]))
                # Get the index of the first element in the sorted list
                optimal_index = MEAN.index(sorted_combined[0][0])
                chosen_cl=CL[optimal_index]
                classifier_selection_sub.append(chosen_cl) ## only for misclassifier.py usage. Added 7 Sep 2023.


                # print('chose: ',chosen_cl)
                _,_ ,save_name_score= name_utility.GETname_model(softwareName,level, species, anti,chosen_cl,temp_path)
                with open(save_name_score + '_KMA_' + str(f_kma) + '_Tree_' + str(f_phylotree) + '.json') as f:
                    score = json.load(f)
                score2= pickle.load(open(save_name_score + '_kma_' + str(f_kma) + '_tree_' + str(f_phylotree) + '_model.pickle',"rb"))

                f1_macro_each=score['f1_test'][outer_cv]
                report_each=score['score_report_test'][outer_cv]
                aucs_each=score['aucs_test'][outer_cv]
                mcc_each=score['mcc_test'][outer_cv]

                hyperparameters_each=score2['hyperparameters_test'][outer_cv]
                score_report_test.append(report_each)
                f1_test.append(f1_macro_each)
                aucs_test.append(aucs_each)
                mcc_test.append(mcc_each)
                hyperparameters_test.append(hyperparameters_each)

            classifier_selection.append(classifier_selection_sub) ## only for misclassifier.py usage. Added 7 Sep 2023.
            if f_kma:# extract infor from report
                summary_table_ByClassifier=  extract_score.score_summary(None, summary_table_ByClassifier_, cv, score_report_test, f1_test,aucs_test,
                                                                           mcc_test, np.zeros(cv))# the last 0: no meaning.
            else:# f_phylotree or random
                summary_table_ByClassifier =  extract_score.score_summary_Tree(None, summary_table_ByClassifier_, cv, score_report_test,
                                                                                f1_test,aucs_test, mcc_test,np.zeros(cv))# the last 0: no meaning.

            common,ind =  math_utility.get_most_fre_hyper(hyperparameters_test,True)
            hy_para_fre.append(common.to_dict())
            hy_para_fren.append(ind)
            hy_para_all.append(hyperparameters_test)
            report_all.append(score_report_test)


        summary_table_ByClassifier_all.append(summary_table_ByClassifier)



    if f_kma:
        final, final_plot,final_std =  make_table.make_visualization(summary_table_ByClassifier_all, antibiotics)
    else:#if f_phylotree or random
        final, final_plot,final_std =  make_table.make_visualization_Tree(summary_table_ByClassifier_all, antibiotics)

    #### If you want to explore more into the classifiers used in each outer loop evaluation, then uncomment the following codes.
    ### final['selected hyperparameter'] = hy_para_fre
    ### final['frequency(out of 10)'] = hy_para_fren
    ### final['hyperparameter sets'] = hy_para_all
    ####################################################################################################################


    ####Add clinical_oriented.
    clinical_table=extract_best_estimator_clinical(report_all,cv,level,species)
    final = pd.concat([final, clinical_table], axis=1, join="inner")
    final_plot = pd.concat([final_plot, clinical_table], axis=1, join="inner")
    # #################################


    _,save_name_final = name_utility.GETname_result(softwareName, species, fscore,f_kma,f_phylotree,'',output_path)
    file_utility.make_dir(os.path.dirname(save_name_final))

    final.to_csv(save_name_final + '_SummaryBenchmarking.txt', sep="\t")
    final_plot.to_csv(save_name_final + '_SummaryBenchmarking_PLOT.txt', sep="\t")
    final_std.to_csv(save_name_final + '_SummaryBenchmarking_std.txt', sep="\t")

    with open(save_name_final + '_classifier.json', 'w') as f:  # overwrite mode. ## only for misclassifier.py usage. Added 7 Sep 2023.
        json.dump(classifier_selection, f)




def extract_best_estimator_clinical(report_all,cv,level,species):
    '''23Aug 2023.
    Select the best classifier in inner loop of nested CV
    '''
    if species =='Mycobacterium tuberculosis':#MT issues. only for PhenotypeSeeker.
        ### antibiotics=['amikacin','capreomycin','ethiomide','ethionamide','kanamycin','ofloxacin','rifampin','streptomycin']
        antibiotics_run=['amikacin','capreomycin','ethiomide','ethionamide','kanamycin','ofloxacin','streptomycin']
    antibiotics, _, _ =  load_data.extract_info(species, False, level)
    score_list=['clinical_f1_negative','clinical_precision_negative', 'clinical_recall_negative']
    summary_table_all = []
    i=0
    for anti in antibiotics:
        print(species,anti)
        summary_table_initial= pd.DataFrame(index=['value'],columns=score_list)
        if species=='Mycobacterium tuberculosis' and anti not in antibiotics_run:
            i+=1
            summary_table=summary_table_initial
        else:
            score_report_test=report_all[i]
            i+=1
            summary_table =  extract_score.score_clinical(summary_table_initial, cv, score_report_test)
        summary_table_all.append(summary_table)

    final =  make_table.make_visualization_clinical(score_list, summary_table_all, antibiotics)
    return final


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
            if "clinical_" in fscore: #### for each classifier
                extract_info_species_clinical(softwareName,cl_list,level, species, cv,f_phylotree,f_kma ,temp_path, output_path)

            else:
                ### extract CV results for each classifier
                # extract_info_species(softwareName,cl_list,level, species, cv,f_phylotree,f_kma ,temp_path, output_path)

                ###extract CV results for the software as a whole: for each outer loop, report the best classifier's results base on inner loop CV.
                ### including also clinical-oriented scores here.
                ### select criteria: f1-macro.
                extract_best_estimator(softwareName,cl_list,level, species, fscore,cv,f_phylotree,f_kma, temp_path,output_path)
        else:# resfinder_folds & majority, ensemble_voting. No need to select the best classifier for reporting.
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
                        help='No use anymore. Deprecate. ')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                        help='kma based cv folders.')

    parsedArgs = parser.parse_args()
    extract_info(parsedArgs.softwareName,parsedArgs.cl_list,parsedArgs.level,parsedArgs.species,parsedArgs.f_all,parsedArgs.cv_number,parsedArgs.fscore,parsedArgs.f_phylotree,
                 parsedArgs.f_kma,parsedArgs.temp_path,parsedArgs.output_path)


