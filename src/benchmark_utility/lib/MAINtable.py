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
    ### tool_list=['Point-/ResFinder' ,'Aytan-Aktug', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover','ML Baseline (Majority)']
    ### folds=['Random folds', 'Phylogeny-aware folds','Homology-aware folds']


    # path_table_results=output_path+ 'Results/supplement_figures_tables/S1_cv_results.xlsx' # output.
    path_table_results=save_file_name
    file_utility.make_dir(os.path.dirname(path_table_results))
    df1 = pd.DataFrame(index=species_list)
    df1.to_excel(path_table_results, sheet_name='introduction')




    for species, antibiotics_selected in zip(df_species, antibiotics):
        print(species,foldset)

        species_sub=[species]
        df_macro=combine_data(species_sub,level,'f1_macro',tool_list,foldset,output_path)
        df_acu=combine_data(species_sub,level,'accuracy',tool_list,foldset,output_path)
        df_neg=combine_data(species_sub,level,'f1_negative',tool_list,foldset,output_path)
        df_pos=combine_data(species_sub,level,'f1_positive',tool_list,foldset,output_path)

        df_cli_neg=combine_data(species_sub,level,'clinical_f1_negative',tool_list,foldset,output_path)
        df_cli_pre=combine_data(species_sub,level,'clinical_precision_neg',tool_list,foldset,output_path)
        df_cli_rec=combine_data(species_sub,level,'clinical_recall_neg',tool_list,foldset,output_path)


        df_macro['f1_negative']=df_neg['f1_negative']
        df_macro['f1_positive']=df_pos['f1_positive']
        df_macro['accuracy']=df_acu['accuracy']
        df_macro['clinical_f1_negative']=df_cli_neg['clinical_f1_negative']
        df_macro['clinical_precision_neg']=df_cli_pre['clinical_precision_neg']
        df_macro['clinical_recall_neg']=df_cli_rec['clinical_recall_neg']


        df_macro=df_macro.reset_index()
        df_macro=df_macro.drop(columns=['index'])
        df_macro = df_macro[['species', 'antibiotics', 'folds', 'software','f1_macro', 'f1_positive', 'f1_negative',
                             'accuracy','clinical_f1_negative','clinical_precision_neg', 'clinical_recall_neg']]
        wb = load_workbook(path_table_results)
        ew = pd.ExcelWriter(path_table_results)
        ew.book = wb
        df_macro.to_excel(ew,sheet_name = species)
        ew.save()
