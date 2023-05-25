#!/usr/bin/python
import sys,os
# sys.path.append('../')
sys.path.insert(0, os.getcwd())
from src.amr_utility import name_utility
import argparse
import pandas as pd
from openpyxl import load_workbook
from src.benchmark_utility.lib.CombineResults import combine_data

"""This script organizes the performance 4 scores(F1-macro,F1-pos,F1-neg,accuracy) and () for Aytan-Aktug SSSA, SSMA, MSMA models."""




def run(species_list,level,foldset, tool_list,score_list, f_compare,path_table_results,output_path):


    # ------------------------------------------
    # Figuring out which ML performs best.
    # ------------------------------------------
    if f_compare:
        df1 = pd.DataFrame(index=species_list)
        df1.to_excel(path_table_results+".xlsx", sheet_name='introduction')

        #--------------------------------
        #mean +- std verson
        ## for fscore in ['f1_macro','f1_positive','f1_negative','accuracy']:
        for fscore in score_list:
            for eachfold in foldset:
                i=0
                for each_tool in tool_list:
                    print(each_tool,eachfold,'-----')
                    df_final=pd.DataFrame(columns=['species', 'antibiotics', each_tool])
                    for species in species_list:

                        species_sub=[species]
                        df_score=combine_data(species_sub,level,fscore,[each_tool],[eachfold],output_path)
                        df_score=df_score.reset_index()
                        df_score=df_score.drop(columns=['index'])
                        df_score[each_tool]=df_score[fscore]
                        df_score=df_score[['species', 'antibiotics',each_tool]]
                        df_final= pd.concat([df_final,df_score])
                    if i==0:
                        df_compare=df_final
                    else:
                        df_compare=pd.merge(df_compare, df_final, how="left", on=['species', 'antibiotics'])
                    i+=1


                df_compare=df_compare[['species', 'antibiotics']+tool_list]

                # #For Supplements File 7 or 8
                wb = load_workbook(path_table_results+'.xlsx')
                ew = pd.ExcelWriter(path_table_results+'.xlsx')
                ew.book = wb
                df_compare.to_excel(ew,sheet_name = (eachfold.split(' ')[0]+'_'+fscore))
                ew.save()



def extract_info(s,level,f_compare,output_path,f_all):

    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]
    if f_all == False:
        data = data.loc[s, :]
    species_list = data.index.tolist()



    foldset=['Homology-aware folds']

    #===============
    #1. Compare Aytan-Aktug  SSSA and 4 variants. 5  tools in all.
    #===============
    print('Compare Aytan-Aktug  SSSA and 4 variants. 5  tools in all.')
    tool_list=[ 'Single-species-antibiotic Aytan-Aktug','Single-species multi-antibiotics Aytan-Aktug', 'Discrete databases multi-species model', \
                'Concatenated databases mixed multi-species model', 'Concatenated databases leave-one-out multi-species model']
    path_table_results=output_path+ 'Results/supplement_figures_tables/S8_Aytan-Aktug_multi'
    score_list=['f1_macro','f1_positive','f1_negative','accuracy','clinical_f1_negative','clinical_precision_neg','clinical_recall_neg']
    run(species_list,level,foldset,tool_list,score_list,f_compare,path_table_results,output_path)
    #===============
    # 2. Compare Aytan-Aktug SSSA and SSSA with default NN settings
    #===============
    print('Compare Aytan-Aktug SSSA and SSSA with default NN settings')
    path_table_results=output_path+ 'Results/supplement_figures_tables/S7_Aytan-Aktug_SSSAdefault'
    score_list=['f1_macro','f1_positive','f1_negative','accuracy']
    tool_list=[ 'Single-species-antibiotic Aytan-Aktug','Single-species-antibiotics default']
    run(species_list,level,foldset,tool_list,score_list, f_compare,path_table_results,output_path)




if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--level', default='loose', type=str,
                        help='Quality control: strict or loose')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'Pseudomonas aeruginosa\' \
                \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
                \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('-f_compare', '--f_compare', dest='f_compare', action='store_true',
                        help='List Aytan-Aktug SSSA, SSSA with default NN and 4 variants and Figure out which ML performs best.')
    parser.add_argument('-o', '--output_path', default='./', type=str, required=False,
                        help='Directory to store CV scores.')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species.')
    parsedArgs = parser.parse_args()
    extract_info(parsedArgs.species,parsedArgs.level,parsedArgs.f_compare,parsedArgs.output_path,parsedArgs.f_all)
