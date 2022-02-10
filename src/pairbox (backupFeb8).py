import sys
import os
sys.path.insert(0, os.path.abspath('../'))
import amr_utility.name_utility
import amr_utility.load_data
import amr_utility.file_utility
import src.BySpecies
import src.ByAnti
import ast
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib.patches import Patch
import pickle
import pandas as pd
import seaborn as sns

def combine_data(species,fscore,tool_list):
    antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, 'loose')
    # This function makes a matrix of all tools' results.
    df_plot = pd.DataFrame(columns=[fscore, 'species', 'software','anti','folds'])
    columns_name=df_plot.columns.tolist()
    df_plot_sub = pd.DataFrame(columns=columns_name)
    print(tool_list)
    folds=['random folds','phylo-tree-based folds','KMA-based folds']
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
        if species !='Mycobacterium tuberculosis' or f_phylotree==False:
            for anti in antibiotics:

                for tool in tool_list:
                    if tool=='Point-/ResFinder':
                        results_file='./benchmarking2_kma/resfinder/Results/summary/loose/'+str(species.replace(" ", "_"))+'.csv'
                        results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
                        score=results.loc[anti,fscore]
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
                            results_file=results_file+'_score_final_Tree_PLOT.txt'
                            results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
                            score=results.loc[anti,fscore]
                        elif f_kma:
                            results_file=results_file+'_score_final_PLOT.txt'
                            fscore_="weighted-"+fscore
                            results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
                            score=results.loc[anti,fscore_]
                        else:
                            results_file=results_file+ '_score_final_Random_PLOT.txt'
                            results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
                            score=results.loc[anti,fscore]

                    if tool=='kmerC':
                        if species !='Mycobacterium tuberculosis' :#no MT information.
                            _, results_file = amr_utility.name_utility.GETsave_name_final(fscore,species, f_kma, f_phylotree, '')
                            results_file='./patric_2022/'+results_file
                            results=pd.read_csv(results_file + '_SummeryBenchmarking_PLOT.txt', header=0, index_col=0,sep="\t")
                            score=results.loc[anti,fscore]
                        else:
                            score=np.nan



                    if tool=='Seq2Geno2Pheno':
                        if species !='Mycobacterium tuberculosis':#no MT information.
                            _, results_file = amr_utility.name_utility.GETsave_name_final(fscore,species, f_kma, f_phylotree, '')
                            results_file='./seq2geno/'+results_file
                            results=pd.read_csv(results_file + '_SummeryBenchmarking_PLOT.txt', header=0, index_col=0,sep="\t")
                            score=results.loc[anti,fscore]
                        else:

                            score=np.nan
                    if tool=='PhenotypeSeeker':
                        if species !='Mycobacterium tuberculosis':
                            _, results_file = amr_utility.name_utility.GETsave_name_final(fscore,species, f_kma, f_phylotree, '')
                            if f_kma:
                                results_file='./PhenotypeSeeker_Nov08/'+results_file
                            elif f_phylotree:
                                results_file='./PhenotypeSeeker_tree/'+results_file
                            else:
                                results_file='./PhenotypeSeeker_random/'+results_file

                            results=pd.read_csv(results_file + '_SummeryBenchmarking_PLOT.txt', header=0, index_col=0,sep="\t")
                            score=results.loc[anti,fscore]
                        else:
                            score=np.nan
                    if tool=='Kover':

                        _, results_file = amr_utility.name_utility.GETsave_name_final(fscore,species, f_kma, f_phylotree, '')
                        if f_kma:
                            results_file='./kover/'+results_file
                        elif f_phylotree:
                            results_file='./kover_tree/'+results_file
                        else:
                            results_file='./kover_random/'+results_file
                        results=pd.read_csv(results_file + '_SummeryBenchmarking_PLOT.txt', header=0, index_col=0,sep="\t")
                        score=results.loc[anti,fscore]

                    if tool=='Majority':

                        results_file,_ = amr_utility.name_utility.GETsave_name_final(fscore,species, f_kma, f_phylotree, 'majority')
                        if f_kma:
                            results_file='./majority/'+results_file
                        elif f_phylotree:
                            results_file='./majority/'+results_file
                        else:
                            results_file='./majority/'+results_file

                        results=pd.read_csv(results_file + '_PLOT.txt', header=0, index_col=0,sep="\t")

                        score=results.loc[anti,fscore]


                    #[fscore, 'species', 'software','anti','folds']
                    df_plot_sub.loc['s'] = [score,species,tool,anti,fold]
                    df_plot = df_plot.append(df_plot_sub, sort=False)
    return df_plot


def extract_info(level,s, fscore, cv_number, f_phylotree, f_kma,f_all):
    data = pd.read_csv('metadata/' + str(level) + '_Species_antibiotic_FineQuality.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    if f_all == False:#should not use it.
        data = data.loc[s, :]

    # data=data.loc[['Escherichia coli','Staphylococcus aureus','Salmonella enterica','Klebsiella pneumoniae','Pseudomonas aeruginosa','Acinetobacter baumannii','Streptococcus pneumoniae','Mycobacterium tuberculosis'],:]
    data=data.loc[['Escherichia coli','Staphylococcus aureus','Salmonella enterica','Klebsiella pneumoniae','Pseudomonas aeruginosa','Acinetobacter baumannii','Streptococcus pneumoniae','Mycobacterium tuberculosis', 'Campylobacter jejuni','Enterococcus faecium','Neisseria gonorrhoeae'],:]

    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    tool_list=['Point-/ResFinder', 'Neural networks', 'KmerC', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover']
    # --------------
    #1. Is it different for species?
    # --------------
    fig, axs = plt.subplots(4,3,figsize=(20, 25))
    plt.tight_layout(pad=4)
    fig.subplots_adjust(wspace=0.25, hspace=0.3, top=0.92, bottom=0.03)
    title='Performance change w.r.t. different folds ('+fscore+')'
    labels = tool_list
    fig.suptitle(title,size=17, weight='bold')

    axs[3,2].axis('off')
    i=0
    for species, antibiotics_selected in zip(df_species, antibiotics):
        print(species)

        row = (i //3)
        col = i % 3
        i+=1
        data_plot=combine_data(species,fscore,tool_list)
        #[fscore, 'species', 'software','anti','folds']
        print(data_plot)
        data_plot= data_plot.astype({fscore:float})
        ax = sns.violinplot(x="folds", y=fscore, ax=axs[row, col],data=data_plot,
                    inner=None, color=".9")
        ax = sns.stripplot(x="folds", y=fscore,ax=axs[row, col],  data=data_plot)
        trans = ax.get_xaxis_transform()
        ax.set_title(species,style='italic', weight='bold')
        #--
        #connect dots representing the same tool+anti combination
        for i in data_plot['folds'].to_list():
            if species!='Mycobacterium tuberculosis':
                data1=data_plot[(data_plot['folds'] != 'random folds')]
                data2=data_plot[(data_plot['folds'] != 'phylo-tree-based fold')]

                ax.plot(data_plot[(data_plot['folds'] != 'random folds')[]], data_plot.loc[idx,['condition 1','condition 2']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
                ax.plot(data_plot.loc[idx,['condition 3','condition 4']], data_plot.loc[idx,['condition 3','condition 4']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
            else:
                pass

    fig.savefig('log/results/FoldsChange_'+fscore+'.png')









