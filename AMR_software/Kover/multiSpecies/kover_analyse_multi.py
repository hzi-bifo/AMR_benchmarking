#!/usr/bin/python
from src.amr_utility import name_utility, file_utility,load_data
import argparse,os,json,ast
import numpy as np
from sklearn.metrics import classification_report,f1_score
import statistics,math
import pandas as pd
from src.analysis_utility.result_analysis import extract_best_estimator_clinical

''' For summerize and visualize the outoputs of Kover multi-species LOSO.'''


def extract_info_species( level,species,antibiotics,f_phylotree,f_kma,temp_path,output_path):
    score_list=['f1_macro','accuracy', 'f1_positive','f1_negative','precision_neg', 'recall_neg']
    ## score_list=['f1_macro','accuracy', 'f1_positive','f1_negative']



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




def extract_info(level,list_species,f_all,temp_path,output_path):
    merge_name = []
    data = pd.read_csv('./data/PATRIC/meta/'+str(level)+'_multi-species_summary.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")

    if f_all:
        list_species = data.index.tolist()[:-1]
        data = data.loc[list_species, :]
    else:
        print('Warning. You are not using all the possible data.')
        data = data.loc[list_species, :]
        data = data.loc[:, (data.sum() > 1)]
    data = data.loc[:, (data != 0).any(axis=0)]
    df_anti = data.dot(data.columns + ';').str.rstrip(';')
    for n in list_species:
        merge_name.append(n[0] + n.split(' ')[1][0])

    score_list=['f1_macro','f1_positive','f1_negative','accuracy', 'precision_macro', 'recall_macro',
                'precision_negative', 'recall_negative','precision_positive', 'recall_positive']

    for each_species in  list_species :
        print(each_species)
        antibiotics=df_anti[each_species].split(';')
        for chosen_cl in ['scm','tree']:
            print(chosen_cl)
            final_table = pd.DataFrame(index=antibiotics,columns=score_list)
            for anti in antibiotics:
                print(anti)

                y_true=[]
                y_pre=[]

                _,name,meta_txt,_ = name_utility.GETname_model3('kover',level, each_species, anti,'',temp_path)
                name_list = pd.read_csv(name, index_col=0, dtype={'genome_id': object}, sep="\t")
                name_list.loc[:,'ID'] = 'iso_' + name_list['genome_id'].astype(str)
                name_list2 = name_list.loc[:, ['ID', 'resistant_phenotype']]


                with open(meta_txt+'_temp/'+str(chosen_cl)+'_b/results.json') as f:
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
                #####

                df=classification_report(y_true, y_pre, labels=[0, 1], output_dict=True,zero_division=0)
                report = pd.DataFrame(df).transpose()
                f1_macro=f1_score(y_true, y_pre, average='macro')
                f1_positive=report.loc['1', 'f1-score']
                accuracy=report.iat[2,2]
                f1_negative=report.loc['0', 'f1-score']
                precision=report.loc['macro avg', 'precision']
                recall=report.loc['macro avg', 'recall']
                precision_pos=report.loc['1', 'precision']
                recall_pos=report.loc['1', 'recall']
                precision_neg=report.loc['0', 'precision']
                recall_neg=report.loc['0', 'recall']



                final_table.loc[anti,:]=[f1_macro,f1_positive,f1_negative,accuracy,precision,recall,precision_neg,recall_neg,precision_pos,recall_pos]

            save_name_score  = name_utility.GETname_result2('kover',each_species,chosen_cl,output_path)
            file_utility.make_dir(os.path.dirname(save_name_score))
            final_table.to_csv(save_name_score + '.txt', sep="\t")

            print(final_table)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                        help='Directory to store temporary files.')
    parser.add_argument('-out', '--output_path', default='./', type=str, required=False,
                        help='Directory to store CV scores.')
    parser.add_argument('-l', '--level', default='loose', type=str,
                        help='Quality control: strict or loose')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument('-f_ml', '--f_ml', dest='f_ml', action='store_true',
                        help='prepare bash')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'Pseudomonas aeruginosa\' \
                    \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
                    \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parsedArgs = parser.parse_args()
    extract_info( parsedArgs.level,parsedArgs.species, parsedArgs.f_all,parsedArgs.temp_path,parsedArgs.output_path)
