#!/usr/bin/python
import sys,os
# sys.path.append('../')
sys.path.insert(0, os.getcwd())
from src.amr_utility import file_utility
import argparse,json
import pandas as pd
import numpy as np
import itertools
from openpyxl import load_workbook
from scipy.stats import ttest_rel
from src.benchmark_utility.lib.CombineResults import combine_data

"""This script organizes the performance 4 scores(F1-macro,F1-pos,F1-neg,accuracy) for Aytan-Aktug SSSA, SSMA, MSMA models."""



def run(species_list,level,fscore,foldset, tool_list, f_compare,f_Ttest,path_table_results,temp_results,output_path):


    # ------------------------------------------
    # Figuring out which ML performs best.
    # ------------------------------------------
    if f_compare:
        print('Now generating: winner version')
        df1 = pd.DataFrame(index=species_list)
        df1.to_excel(temp_results+"_winner.xlsx", sheet_name='introduction')
        for eachfold in foldset:

            i=0
            for each_tool in tool_list:
                df_final=pd.DataFrame(columns=['species', 'antibiotics', each_tool])

                for species in species_list :
                    species_sub=[species]
                    df_score=combine_data(species_sub,level,fscore,[each_tool],[eachfold],output_path)
                    df_score=df_score.reset_index()
                    df_score=df_score.drop(columns=['index'])

                    df_score[fscore] = df_score[fscore].astype(str)
                    df_score[each_tool]=df_score[fscore].apply(lambda x:x.split('±')[0] ) #df_final['f1_macro']
                    #### df_score[each_tool+'_std']=df_score[fscore].apply(lambda x: x.split('±')[1] if (len(x.split('±'))==2) else np.nan)
                    df_score[each_tool+'_std']=df_score['f1_macro'].apply(lambda x: x.split('±')[1] if (len(x.split('±'))==2) else 10)
                    df_score[each_tool] = df_score[each_tool] .astype(float)
                    df_score[each_tool+'_std'] = df_score[each_tool+'_std'] .astype(float)
                    df_score=df_score[['species', 'antibiotics',each_tool,each_tool+'_std']]
                    df_final= pd.concat([df_final,df_score])

                if i==0:
                    df_compare=df_final
                else:
                    df_compare=pd.merge(df_compare, df_final, how="left", on=['species', 'antibiotics'])
                i+=1


            df_std=df_compare[ [i+'_std' for i in tool_list]]
            df_compare['max_'+fscore]=df_compare[tool_list].max(axis=1)
            a = df_compare[tool_list]
            df = a.eq(a.max(axis=1), axis=0)

            for index, row in df.iterrows():

                winner=[]
                winner_std=[]
                for columnIndex, value in row.items():

                    if value==True:
                        winner.append(columnIndex)
                        winner_std.append(df_std.loc[index,columnIndex+'_std'])
                if len(winner)>1: #more than two winner, check std

                    min_std = min(winner_std)
                    winner_index=[i for i, x in enumerate(winner_std) if x == min_std]
                    winner_filter=np.array(winner)[winner_index]
                    filter=list(set(winner) - set(winner_filter))

                    for each in filter:
                        row[each]=False

            #for further plotting graphs
            df_compare[['species', 'antibiotics']+tool_list+[i+'_std' for i in tool_list]].to_csv(temp_results+'_multi.csv',index=True,header=True, sep="\t")
            # print(df_compare)
            #for Supplemental File 7 or 8, and compare to Supplemental File 6, add results here to Supplemental File 6 with caution, manually.
            df_compare['winner'] = df.mul(df.columns.to_series()).apply(','.join, axis=1).str.strip(',')

            df_compare=df_compare.replace({10: np.nan})
            df_compare=df_compare[['species', 'antibiotics']+tool_list+[ 'max_'+fscore]+[i+'_std' for i in tool_list]+['winner' ]]
            wb = load_workbook(temp_results+'_winner.xlsx')
            ew = pd.ExcelWriter(temp_results+'_winner.xlsx')
            ew.book = wb
            df_compare.to_excel(ew,sheet_name = (eachfold))
            ew.save()
        #--------------------------------
        #mean +- std verson
        print('Now generating: mean +- std version')
        for eachfold in foldset:
            i=0
            for each_tool in tool_list:

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

            wb = load_workbook(temp_results+'_winner.xlsx')
            ew = pd.ExcelWriter(temp_results+'_winner.xlsx')
            ew.book = wb
            df_compare.to_excel(ew,sheet_name = (eachfold+'_'+fscore))
            ew.save()


    if f_Ttest:
        # paired T-test
        print('Now paired T-test')
        for eachfold in foldset:
            i=0
            for each_tool in tool_list:
                print(each_tool)
                df_final=pd.DataFrame(columns=['species', 'antibiotics', each_tool])
                for species in  species_list:

                    species_sub=[species]
                    df_score=combine_data(species_sub,level,fscore,[each_tool],[eachfold],output_path)
                    df_score=df_score.reset_index()
                    df_score=df_score.drop(columns=['index'])

                    df_score[fscore] = df_score[fscore].astype(str)
                    df_score[each_tool]=df_score[fscore].apply(lambda x:x.split('±')[0]) #get mean if there is mean
                    df_score[each_tool] = df_score[each_tool] .astype(float)
                    # rounded to 2
                    df_score=df_score.round(2)
                    #---
                    df_score=df_score[['species', 'antibiotics',each_tool]]
                    df_final= pd.concat([df_final,df_score])

                if i==0:
                    df_compare=df_final
                else:
                    df_compare=pd.merge(df_compare, df_final, how="left", on=['species', 'antibiotics'])
                i+=1
            print(df_compare)
            df_mean=df_compare[tool_list]
            df_mean = df_mean.dropna() #Nan means no multi-models for that combination.
            print(df_mean)
            print('Paired T-test:')
            #T-test
            Presults={}
            for each_com in list(itertools.combinations(tool_list, 2)):
                mean1 = df_mean[each_com[0]]
                mean2 = df_mean[each_com[1]]
                result=ttest_rel(mean1, mean2)
                pvalue = result[1]
                print(each_com,pvalue)
                Presults[str(each_com)]=pvalue
            with open(path_table_results + '_Pvalue.json', 'w') as f:
                json.dump(Presults, f)




def extract_info(level,fscore,f_compare,f_Ttest,f_kover,f_pts,output_path):

    #
    foldset=['Homology-aware folds']
    if f_kover==False and f_pts==False:
        #===============
        #1. Compare Aytan-Aktug  SSSA and MSMA (3 variants)
        # #===============
        print('1. Compare Aytan-Aktug  SSSA and MSMA (3 variants)')
        species_list=['Escherichia coli','Staphylococcus aureus','Salmonella enterica','Klebsiella pneumoniae','Pseudomonas aeruginosa',
                      'Acinetobacter baumannii','Streptococcus pneumoniae','Mycobacterium tuberculosis','Campylobacter jejuni']
        tool_list=[ 'Single-species-antibiotic Aytan-Aktug','Discrete databases multi-species model',
                    'Concatenated databases mixed multi-species model', 'Concatenated databases leave-one-out multi-species model']
        path_table_results=output_path+ 'Results/supplement_figures_tables/S8_Aytan-Aktug_MSMA'
        temp_results=output_path+ 'Results/other_figures_tables/S8_Aytan-Aktug_MSMA'
        file_utility.make_dir(os.path.dirname(temp_results))
        file_utility.make_dir(os.path.dirname(path_table_results))
        run(species_list,level,fscore,foldset,tool_list,f_compare,f_Ttest,path_table_results,temp_results,output_path)
        # #===============
        #2. Compare Aytan-Aktug  SSSA and SSMA
        #===============
        # print('2. Compare Aytan-Aktug  SSSA and SSMA')
        # species_list=['Escherichia coli','Staphylococcus aureus','Salmonella enterica','Klebsiella pneumoniae','Pseudomonas aeruginosa',
        #               'Acinetobacter baumannii','Streptococcus pneumoniae','Mycobacterium tuberculosis', 'Neisseria gonorrhoeae']
        # tool_list=[ 'Single-species-antibiotic Aytan-Aktug','Single-species multi-antibiotics Aytan-Aktug']
        # path_table_results=output_path+ 'Results/supplement_figures_tables/S8_Aytan-Aktug_SSMA'
        # temp_results=output_path+ 'Results/other_figures_tables/S8_Aytan-Aktug_SSMA'
        # run(species_list,level,fscore,foldset,tool_list,f_compare,f_Ttest,path_table_results,temp_results,output_path)

        # #===============
        # # 3. Compare Aytan-Aktug SSSA and SSSA with default NN settings
        # #===============
        # print('3.  Compare Aytan-Aktug SSSA and SSSA with default NN settings')
        # species_list=['Escherichia coli','Staphylococcus aureus','Salmonella enterica','Klebsiella pneumoniae','Pseudomonas aeruginosa',
        #               'Acinetobacter baumannii','Streptococcus pneumoniae','Mycobacterium tuberculosis', 'Campylobacter jejuni',
        #               'Enterococcus faecium','Neisseria gonorrhoeae']
        #
        # path_table_results=output_path+ 'Results/supplement_figures_tables/S7_Aytan-Aktug_SSSAdefault'
        # temp_results=output_path+ 'Results/other_figures_tables/S8_Aytan-Aktug_SSSAdefault'
        # tool_list=[ 'Single-species-antibiotic Aytan-Aktug','Single-species-antibiotics default']
        # run(species_list,level,fscore,foldset,tool_list, f_compare,f_Ttest,path_table_results,temp_results,output_path)
        #
        # #TODO:maybe add 5 tools+ AAmulti-models; maybe not.

    #===============
    # 4. Compare Kover
    #===============
    if f_kover:
        print('4. Paired T test betweeen Kover multi- & single- models')
        species_list=['Escherichia coli','Staphylococcus aureus','Salmonella enterica','Klebsiella pneumoniae','Pseudomonas aeruginosa',
                      'Acinetobacter baumannii','Streptococcus pneumoniae','Mycobacterium tuberculosis', 'Campylobacter jejuni',
                      'Enterococcus faecium','Neisseria gonorrhoeae']
        tool_list=[ 'Kover single-species SCM','Kover single-species CART','Kover cross-species SCM','Kover cross-species CART']
        path_table_results=output_path+ 'Results/supplement_figures_tables/S8_Kover_multi'
        temp_results=output_path+ 'Results/other_figures_tables/S8_Kover_multi'
        run(species_list,level,fscore,foldset,tool_list, f_compare,f_Ttest,path_table_results,temp_results,output_path)



    #===============
    # 5. Compare PhenotypeSeeker
    #===============
    if f_pts:
        print('5. Paired T test betweeen PhenotypeSeeker multi- & single- models')
        species_list=['Escherichia coli','Staphylococcus aureus','Salmonella enterica','Klebsiella pneumoniae','Pseudomonas aeruginosa',
                      'Acinetobacter baumannii','Streptococcus pneumoniae','Mycobacterium tuberculosis', 'Campylobacter jejuni',
                      'Enterococcus faecium','Neisseria gonorrhoeae']
        tool_list=[ 'PhenotypeSeeker single-species LR','PhenotypeSeeker multi-species LR']
        path_table_results=output_path+ 'Results/supplement_figures_tables/S8_PTS_multi'
        temp_results=output_path+ 'Results/other_figures_tables/S8_PTS_multi'
        run(species_list,level,fscore,foldset,tool_list, f_compare,f_Ttest,path_table_results,temp_results,output_path)


        ##### cross tools
        tool_list=['Discrete databases multi-species model','Concatenated databases mixed multi-species model',\
                   'Concatenated databases leave-one-out multi-species model','Kover cross-species SCM','Kover cross-species CART',\
                   'PhenotypeSeeker multi-species LR']
        path_table_results=output_path+ 'Results/supplement_figures_tables/S8_crossT_multi'
        temp_results=output_path+ 'Results/other_figures_tables/S8_crossT_multi'
        run(species_list,level,fscore,foldset,tool_list, f_compare,f_Ttest,path_table_results,temp_results,output_path)






if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--level', default='loose', type=str,
                        help='Quality control: strict or loose')
    parser.add_argument('-fscore', '--fscore', default='f1_macro', type=str, required=False,
                        help='The score used for final comparison. Can be one of: \'f1_macro\',\'f1_positive\',\'f1_negative\',\'accuracy\'.')
    parser.add_argument('-f_Ttest', '--f_Ttest', dest='f_Ttest', action='store_true',
                        help=' Perform paired T test.')
    parser.add_argument('-f_kover', '--f_kover', dest='f_kover', action='store_true',
                        help=' Perform paired T test for Kover.')
    parser.add_argument('-f_pts', '--f_pts', dest='f_pts', action='store_true',
                        help=' Perform paired T test for PhenotypeSeeker.')
    parser.add_argument('-f_compare', '--f_compare', dest='f_compare', action='store_true',
                        help='List Aytan-Aktug SSSA, SSSA with default NN and 4 variants and Figure out which ML performs best.')
    parser.add_argument('-o', '--output_path', default='./', type=str, required=False,
                        help='Directory to store CV scores.')


    parsedArgs = parser.parse_args()
    extract_info(parsedArgs.level,parsedArgs.fscore,parsedArgs.f_compare,parsedArgs.f_Ttest,parsedArgs.f_kover,parsedArgs.f_pts,parsedArgs.output_path)
