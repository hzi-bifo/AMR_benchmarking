import amr_utility.name_utility
import amr_utility.load_data
import amr_utility.file_utility
import pickle
import pandas as pd
import numpy as np
import statistics
from openpyxl import load_workbook

"""This script organizes the performance for KMA AND Blastn version of resfinder for Supplemental 5 ."""

def combine_data(species_list,fscore):

    # This function makes a matrix of all tools' results.
    df_plot = pd.DataFrame(columns=['species','antibiotics', 'software']+fscore)
    columns_name=df_plot.columns.tolist()
    df_plot_sub = pd.DataFrame(columns=columns_name)


    for species in species_list:#'Point-/ResFinder':
            antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, 'loose')
            for anti in antibiotics:
                if species != 'Neisseria gonorrhoeae':
                    results_file='./benchmarking2_kma/resfinder/Results/summary/loose/'+str(species.replace(" ", "_"))+'.csv'
                    results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
                    score=results.loc[anti,fscore].to_list()

                    df_plot_sub.loc['s'] = [species,anti,'KMA-based Point-/ResFinder']+score
                    df_plot = df_plot.append(df_plot_sub, sort=False)
                results_file='./benchmarking2_kma/resfinder/Results/blast_version/summary/loose/'+str(species.replace(" ", "_"))+'.csv'
                results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
                score=results.loc[anti,fscore].to_list()
                df_plot_sub.loc['s'] =[species,anti,'Blastn-based Point-/ResFinder'] +score
                df_plot = df_plot.append(df_plot_sub, sort=False)
    return df_plot

def extract_info(level,s, fscore, f_all):
    data = pd.read_csv('metadata/' + str(level) + '_Species_antibiotic_FineQuality.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    if f_all == False:#should not use it.
        data = data.loc[s, :]
    species_list=['Escherichia coli','Staphylococcus aureus','Salmonella enterica','Klebsiella pneumoniae','Pseudomonas aeruginosa','Acinetobacter baumannii','Streptococcus pneumoniae','Mycobacterium tuberculosis', 'Campylobacter jejuni','Enterococcus faecium','Neisseria gonorrhoeae']
    data=data.loc[species_list,:]
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    # tool_list=[ 'Neural networks', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover','ML Baseline (Majority)']

    # initialize
    # f = open('log/results/cv_results.xlsx', 'w+')#The file is created if it does not exist, otherwise it is truncated.
    df1 = pd.DataFrame(index=species_list)
    df1.to_excel("log/results/cv_results_resfinder.xlsx", sheet_name='introduction')



    for species, antibiotics_selected in zip(df_species, antibiotics):
        # summary= pd.DataFrame(columns=['species', 'antibiotics', 'folds', 'f1_macro', 'f1_positive', 'f1_negative', 'accuracy'])
        antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)

        species_sub=[species]
        df_macro=combine_data(species_sub,['f1_macro','f1_negative','f1_positive','accuracy','mcc'])
        df_macro=df_macro.reset_index()
        df_macro=df_macro.drop(columns=['index'])
        df_macro = df_macro[['species', 'antibiotics', 'software','f1_macro', 'f1_positive', 'f1_negative', 'accuracy','mcc']]
        wb = load_workbook('log/results/cv_results_resfinder.xlsx')
        ew = pd.ExcelWriter('log/results/cv_results_resfinder.xlsx')
        ew.book = wb
        df_macro.to_excel(ew,sheet_name = species)
        ew.save()
