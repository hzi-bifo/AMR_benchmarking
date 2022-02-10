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


def combine_data(species_list,anti,fscore, f_phylotree, f_kma,tool_list,merge_name):
    # This function makes a matrix of all tools' results.
    df_plot = pd.DataFrame(columns=[fscore, 'species', 'software'])
    columns_name=df_plot.columns.tolist()
    df_plot_sub = pd.DataFrame(columns=columns_name)
    print(tool_list)
    for species in species_list:
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
            if tool=='Neural networks Multi-species':
                # todo
                results_file = amr_utility.name_utility.GETname_multi_bench_save_name_final(fscore,merge_name,
                                                                                                     'all_possible_anti', 'loose',
                                                                                                     0.0,
                                                                                                     0, True,
                                                                                                     False,
                                                                                                     'f1_macro')

                results_file='./benchmarking2_kma/'+results_file

                if f_phylotree :
                    print('error')
                    exit()
                elif f_kma:
                    results_file=results_file+'_score_final.txt'
                    # fscore_="weighted-"+fscore
                    results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
                    score=results.loc[anti,fscore]
                else:
                    print('error')
                    exit()
            if tool=='KmerC':
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



            df_plot_sub.loc['s'] = [score,species,tool]
            df_plot = df_plot.append(df_plot_sub, sort=False)
    return df_plot


def ComByAnti(level,s, fscore, cv_number, f_phylotree, f_kma,f_all):
    '''
    Plot benchmarking resutls by antibiotics. Only those antibiotics that are with data of multi-species.
    Tool:
    '''

    data = pd.read_csv('metadata/' + str(level) + '_multi-species_summary.csv', index_col=0,
                               dtype={'genome_id': object}, sep="\t")

    if f_phylotree:
        list_species=data.index.tolist()[1:-1]#MT no data.

    else:
        list_species = data.index.tolist()[:-1]
    data = data.loc[list_species, :]
    data = data.loc[:, (data != 0).any(axis=0)]
    print(data)
    merge_name = []
    for n in list_species:
        merge_name.append(n[0] + n.split(' ')[1][0])
    merge_name = '_'.join(merge_name)  # e.g.Se_Kp_Pa
    #NN_discreate multi-species.


    fig, axs = plt.subplots(4,5,figsize=(25, 20))
    # fig.subplots_adjust(top=0.88)
    plt.tight_layout(pad=4)
    fig.subplots_adjust(wspace=0.25, hspace=0.3, top=0.92, bottom=0.05)
    if f_phylotree:
        title='Performance w.r.t. phylo-tree-based folds ('+fscore+')'
    elif f_kma:
        title='Performance w.r.t. KMA-based folds ('+fscore+')'
    else:
        title='Performance w.r.t. random folds ('+fscore+')'

    fig.suptitle(title,size='large', weight='bold')
    blue=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
    orange=(1.0, 0.4980392156862745, 0.054901960784313725)
    green= (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
    purp=(0.5803921568627451, 0.403921568627451, 0.7411764705882353)
    red=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392)
    red='#ef3b2c'
    brown=(0.5490196078431373, 0.33725490196078434, 0.29411764705882354)

    if (fscore=='f1_macro' or fscore=='accuracy') and f_kma==True:
        tool_list=['Point-/ResFinder', 'Neural networks','Neural networks Multi-species', 'KmerC', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover','Majority']
        colors = [blue,brown,"#FFD33C", orange,purp , green , red, "black"]# #ffd343
    elif (fscore=='f1_macro' or fscore=='accuracy') and f_kma==False:
        tool_list=['Point-/ResFinder', 'Neural networks' , 'KmerC', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover','Majority']
        colors = [blue,brown, orange,purp , green , red, "black"]# #ffd343
    elif f_kma==True:
        tool_list=['Point-/ResFinder', 'Neural networks','Neural networks Multi-species', 'KmerC', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover']
        colors = [blue,brown,'#FFD33C', orange,purp , green , red]# #ffd343
    else:
        tool_list=['Point-/ResFinder', 'Neural networks', 'KmerC', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover']
        colors = [blue,brown, orange,purp , green , red]



    i=0
    df_s = data.T.dot(data.T.columns + ';').str.rstrip(';')#get anti names  marked with 1
    print(df_s)
    All_antibiotics = data.columns.tolist()
    for anti in All_antibiotics:

        species=df_s[anti].split(';')



        data=combine_data(species,anti,fscore, f_phylotree, f_kma,tool_list,merge_name)



        print(data)
        # print(df1)
        row = (i // 5)
        col = i % 5
        i+=1

        # g = df.plot(ax=axs[row, col],kind="bar",color=colors, x='software',y=antibiotics)
        g = sns.barplot(x="species", y=fscore, hue='software',
                        data=data, dodge=True, ax=axs[row, col],palette=colors)
        g.set_title(anti,style='italic', weight='bold')
        if anti == 'tetracycline':
            g.set_xticklabels(g.get_xticklabels(), rotation=20, horizontalalignment='right')
        else:
            g.set_xticklabels(g.get_xticklabels(), rotation=10, horizontalalignment='center')
        g.set(ylim=(0, 1.0))
        g.set_xlabel('')
        if i!=1:
            # handles, labels = g.get_legend_handles_labels()
            # g.legend('', '')
            g.get_legend().remove()

        else:
            handles, labels = g.get_legend_handles_labels()
            g.legend(bbox_to_anchor=(0.5,1.25), ncol=8,fontsize=16,frameon=False)
    fig.savefig('log/results/ByAnti_'+'kma_'+str(f_kma)+'_tree_'+str(f_phylotree)+'_'+fscore+'.png')
