import amr_utility.name_utility
import amr_utility.load_data
import amr_utility.file_utility
import pickle
import pandas as pd
import numpy as np
import statistics
from openpyxl import load_workbook

"""This script organizes the performance 4 scores for Supplementary materials."""

def combine_data(species_list,fscore,tool_list):
    out_score=['f1_macro', 'accuracy', 'f1_positive', 'f1_negative', 'precision_neg', 'recall_neg']
    # This function makes a matrix of all tools' results.
    df_plot = pd.DataFrame(columns=out_score+['species', 'software','antibiotics','folds'])
    columns_name=df_plot.columns.tolist()
    df_plot_sub = pd.DataFrame(columns=columns_name)
    # print(tool_list)
    folds=['random folds','phylo-tree-based folds','KMA-based folds']

    for species in species_list:#'Point-/ResFinder':
            antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, 'loose')
            for anti in antibiotics:
                results_file='./benchmarking2_kma/resfinder/Results/summary/loose/'+str(species.replace(" ", "_"))+'.csv'
                results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
                score=results.loc[anti,out_score].to_list()
                df_plot_sub.loc['s'] = score+[species,'Point-/ResFinder',anti,'-']
                df_plot = df_plot.append(df_plot_sub, sort=False)
    # -----------------------------------
    for fold in folds:
        if fold=='phylo-tree-based folds':
            f_phylotree=True
            f_kma=False
        elif fold=='KMA-based folds':
            f_phylotree=False
            f_kma=True
        else:
            f_phylotree=False
            f_kma=False
        for species in species_list:
            if species !='Mycobacterium tuberculosis' or f_phylotree==False:
                antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, 'loose')
                for anti in antibiotics:

                    for tool in tool_list:
                        # if tool=='Point-/ResFinder':
                        #
                        #     results_file='./benchmarking2_kma/resfinder/Results/summary/loose/'+str(species.replace(" ", "_"))+'.csv'
                        #     results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
                        #     score=results.loc[anti,fscore]
                        if tool=='Neural networks':
                            results_file = amr_utility.name_utility.GETname_multi_bench_save_name_final(fscore,species, None,
                                                                                                                         'loose',
                                                                                                                         0.0,
                                                                                                                         0,
                                                                                                                         True,
                                                                                                                         False,
                                                                                                                         'f1_macro')
                            results_file='./benchmarking2_kma/'+results_file
                            if f_phylotree :
                                results_file=results_file+'_score_final_Tree.txt'
                                results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
                                score=results.loc[anti,fscore]
                            elif f_kma:
                                results_file=results_file+'_score_final.txt'
                                fscore_="weighted-"+fscore
                                results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
                                score=results.loc[anti,fscore_]
                            else:
                                results_file=results_file+ '_score_final_Random.txt'
                                results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
                                score=results.loc[anti,fscore]

                        # if tool=='KmerC':
                        #     if species !='Mycobacterium tuberculosis' :#no MT information.
                        #         _, results_file = amr_utility.name_utility.GETsave_name_final(fscore,species, f_kma, f_phylotree, '')
                        #         results_file='./patric_2022/'+results_file
                        #         results=pd.read_csv(results_file + '_SummeryBenchmarking.txt', header=0, index_col=0,sep="\t")
                        #         score=results.loc[anti,fscore]
                        #     else:
                        #         score=np.nan
                        if tool=='Seq2Geno2Pheno':
                            if species !='Mycobacterium tuberculosis':#no MT information.
                                _, results_file = amr_utility.name_utility.GETsave_name_final(fscore,species, f_kma, f_phylotree, '')
                                results_file='./seq2geno/'+results_file
                                results=pd.read_csv(results_file + '_SummeryBenchmarking.txt', header=0, index_col=0,sep="\t")
                                score=results.loc[anti,fscore]
                            else:

                                score=np.nan
                        if tool=='PhenotypeSeeker':
                            # if species !='Mycobacterium tuberculosis':
                            _, results_file = amr_utility.name_utility.GETsave_name_final(fscore,species, f_kma, f_phylotree, '')
                            if f_kma:
                                results_file='./PhenotypeSeeker_Nov08/'+results_file
                            elif f_phylotree:
                                results_file='./PhenotypeSeeker_tree/'+results_file
                            else:
                                results_file='./PhenotypeSeeker_random/'+results_file

                            results=pd.read_csv(results_file + '_SummeryBenchmarking.txt', header=0, index_col=0,sep="\t")
                            # print(results.loc[anti,out_score].to_list())
                            score=results.loc[anti,out_score].to_list()
                            # else:
                            #     score=np.nan
                        if tool=='Kover':

                            _, results_file = amr_utility.name_utility.GETsave_name_final(fscore,species, f_kma, f_phylotree, '')
                            if f_kma:
                                results_file='./kover/'+results_file
                            elif f_phylotree:
                                results_file='./kover_tree/'+results_file
                            else:
                                results_file='./kover_random/'+results_file
                            results=pd.read_csv(results_file + '_SummeryBenchmarking.txt', header=0, index_col=0,sep="\t")
                            score=results.loc[anti,fscore]

                        if tool=='Baseline (Majority)':

                            results_file,_ = amr_utility.name_utility.GETsave_name_final(fscore,species, f_kma, f_phylotree, 'majority')
                            if f_kma:
                                results_file='./majority/'+results_file
                            elif f_phylotree:
                                results_file='./majority/'+results_file
                            else:
                                results_file='./majority/'+results_file

                            results=pd.read_csv(results_file + '.txt', header=0, index_col=0,sep="\t")

                            score=results.loc[anti,fscore]
                        #[fscore, 'species', 'software','anti','folds']
                        df_plot_sub.loc['s'] = score+[species,tool,anti,fold]
                        df_plot = df_plot.append(df_plot_sub, sort=False)
    return df_plot

def extract_info(level,s, fscore, f_all):
    data = pd.read_csv('metadata/' + str(level) + '_Species_antibiotic_FineQuality.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    if f_all == False:#should not use it.
        data = data.loc[s, :]
    species_list=['Escherichia coli','Staphylococcus aureus','Salmonella enterica','Klebsiella pneumoniae','Pseudomonas aeruginosa','Acinetobacter baumannii','Streptococcus pneumoniae','Campylobacter jejuni','Enterococcus faecium','Neisseria gonorrhoeae']
    data=data.loc[species_list,:]
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    # tool_list=[ 'Neural networks', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover']
    tool_list=['PhenotypeSeeker']

    # initialize
    # f = open('log/results/cv_results.xlsx', 'w+')#The file is created if it does not exist, otherwise it is truncated.
    df1 = pd.DataFrame(index=species_list)
    name="log/results/cv_results_clinical.xlsx"
    df1.to_excel(name, sheet_name='introduction')



    for species, antibiotics_selected in zip(df_species, antibiotics):
        # summary= pd.DataFrame(columns=['species', 'antibiotics', 'folds', 'f1_macro', 'f1_positive', 'f1_negative', 'accuracy'])
        antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
        print(species)
        species_sub=[species]
        # df_macro=combine_data(species_sub,'f1_macro',tool_list)
        # if species=='Klebsiella pneumoniae':
        #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #         print(df_macro)
        # df_acu=combine_data(species_sub,'accuracy',tool_list)
        df_neg=combine_data(species_sub,'f1_negative',tool_list)
        # df_pos=combine_data(species_sub,'f1_positive',tool_list)
        # df_macro.insert(loc=0, column='species', value=[species] * (df_macro.shape[0]))
        # df_macro['f1_negative']=df_neg['f1_negative']
        # df_macro['f1_positive']=df_pos['f1_positive']
        # df_macro['accuracy']=df_acu['accuracy']


        df_neg=df_neg.reset_index()
        df_neg=df_neg.drop(columns=['index'])
        # df_macro = df_macro[['species', 'antibiotics', 'folds', 'software','f1_macro', 'f1_positive', 'f1_negative', 'accuracy']]
        wb = load_workbook(name)
        ew = pd.ExcelWriter(name)
        ew.book = wb
        df_neg.to_excel(ew,sheet_name = species)
        ew.save()
