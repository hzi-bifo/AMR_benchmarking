

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





def extract_info_species(softwareName,cl_list,level,species_testing,antibiotics_test, fscore,temp_path, output_path):

    score_list=['f1_macro','f1_positive', 'f1_negative','accuracy','precision_macro', 'recall_macro',
                'precision_negative', 'recall_negative','precision_positive', 'recall_positive']
    # out_score='neg'
    for chosen_cl in cl_list:
        print('---------------------',chosen_cl)

        final_table = pd.DataFrame(index=antibiotics_test,columns=score_list)
        for anti in antibiotics_test:
            _,_, _,save_name_score= name_utility.GETname_model3('phenotypeseeker',level, species_testing, anti,chosen_cl,temp_path)
            with open(save_name_score +'.json') as f:
                    score = json.load(f)
            f1_macro=score['f1_test'][0]
            report_df=score['score_report_test'][0]
            report = pd.DataFrame(report_df).transpose()
            # aucs_test=score['aucs_test']
            # mcc_test=score['mcc_test']
            f1_positive=report.loc['1', 'f1-score']
            accuracy=report.iat[2,2] #no use of this score
            f1_negative=report.loc['0', 'f1-score']
            precision=report.loc['macro avg', 'precision']
            recall=report.loc['macro avg', 'recall']
            precision_pos=report.loc['1', 'precision']
            recall_pos=report.loc['1', 'recall']
            precision_neg=report.loc['0', 'precision']
            recall_neg=report.loc['0', 'recall']
            final_table.loc[anti,:]=[f1_macro,f1_positive,f1_negative,accuracy,precision,recall,precision_neg,recall_neg,precision_pos,recall_pos]


        #finish one chosen_cl

        save_name_score_final  = name_utility.GETname_result2('phenotypeseeker',species_testing,chosen_cl,output_path)
        file_utility.make_dir(os.path.dirname(save_name_score_final))
        final_table.to_csv(save_name_score_final + '.txt', sep="\t")
        print(final_table)





def extract_info(softwareName,cl_list,level,list_species,f_all,cv,fscore,temp_path, output_path):
    merge_name = []
    data = pd.read_csv('./data/PATRIC/meta/'+str(level)+'_multi-species_summary.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")

    if f_all:
        list_species = data.index.tolist()[:-1]
        data = data.loc[list_species, :]
    else:
        print('Warning. You are not using all the possible data.')
        # data = data.loc[list_species, :]
        # data = data.loc[:, (data.sum() > 1)]
        exit(1)
    data = data.loc[:, (data != 0).any(axis=0)]
    All_antibiotics = data.columns.tolist()
    df_anti = data.dot(data.columns + ';').str.rstrip(';')
    for n in list_species:
        merge_name.append(n[0] + n.split(' ')[1][0])
    merge_name = '_'.join(merge_name)  # e.g.Se_Kp_Pa
    print(data)
    print(df_anti)

    for species_testing in list_species:
        print(species_testing)
        antibiotics_test=df_anti[species_testing].split(';')
        extract_info_species(softwareName,cl_list,level, species_testing,antibiotics_test,fscore,temp_path, output_path)




if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-software', '--softwareName', type=str,
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
    # parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
    #                     help=' phylo-tree based cv folders.')
    # parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
    #                     help='kma based cv folders.')

    parsedArgs = parser.parse_args()
    # parser.print_help()
    extract_info(parsedArgs.softwareName,parsedArgs.cl_list,parsedArgs.level,parsedArgs.species,parsedArgs.f_all,parsedArgs.cv_number,
                 parsedArgs.fscore,parsedArgs.temp_path,parsedArgs.output_path)
