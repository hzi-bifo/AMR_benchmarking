#!/usr/bin/python
import sys,os
# sys.path.append('../')
sys.path.insert(0, os.getcwd())
from src.amr_utility import name_utility,load_data,file_utility
import pandas as pd
from openpyxl import load_workbook
from src.benchmark_utility.lib.CombineResults import combine_data
"""
Performance(F1-macro, negative F1-score, positive F1-score, accuracy) of five methods alongside with the baseline method (Majority) 
w.r.t. random folds, phylogeny-aware folds, and homology-aware folds, in the 10-fold nested cross-validation. 
For Kover, Seq2Geno2Pheno, and PhenotypeSeeker, the four scores were not necessarily resulted from the same classifier, 
as multiple classifiers were used, and we reported the best result regarding each corresponding score among all the possible classifiers 
for each species-antibiotic combination. 


"""

score_set=['f1_macro', 'f1_positive', 'f1_negative', 'accuracy',
        'precision_macro', 'recall_macro', 'precision_negative', 'recall_negative','precision_positive', 'recall_positive',
        'mcc',  'auc', 'clinical_f1_negative','clinical_precision_negative', 'clinical_recall_negative']


def extract_info(level,s, f_all,output_path,tool_list,foldset,save_file_name):
    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    if f_all == False:#should not use it.
        data = data.loc[s, :]
    species_list=['Escherichia coli','Staphylococcus aureus','Salmonella enterica','Klebsiella pneumoniae','Pseudomonas aeruginosa',
                  'Acinetobacter baumannii','Streptococcus pneumoniae','Mycobacterium tuberculosis', 'Campylobacter jejuni',
                  'Enterococcus faecium','Neisseria gonorrhoeae']
    data=data.loc[species_list,:]
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()


    path_table_results=save_file_name
    file_utility.make_dir(os.path.dirname(path_table_results))
    df1 = pd.DataFrame(index=species_list)
    df1.to_excel(path_table_results, sheet_name='introduction')




    for species, antibiotics_selected in zip(df_species, antibiotics):
        print(species,foldset)

        species_sub=[species]
        df_main=combine_data(species_sub,level,'f1_macro',tool_list,foldset,output_path)
        for each in score_set[1:]:
            df_each=combine_data(species_sub,level,each,tool_list,foldset,output_path)
            df_main[each]=df_each[each]

        df_main=df_main.reset_index()
        df_main=df_main.drop(columns=['index'])
        df_main = df_main[['species', 'antibiotics', 'folds', 'software']+score_set]
        wb = load_workbook(path_table_results)
        ew = pd.ExcelWriter(path_table_results)
        ew.book = wb
        df_main.to_excel(ew,sheet_name = species)
        ew.save()
