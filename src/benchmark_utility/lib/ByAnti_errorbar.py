import sys
import os
sys.path.insert(0, os.path.abspath('../'))
import numpy as np
import copy
import matplotlib.pyplot as plt
import json
import pandas as pd
from src.benchmark_utility.lib.CombineResults import combine_data_ByAnti


''' Compare performance on Antibitoics shared by multiple species.'''





def ComByAnti(level,tool_list,fscore, f_phylotree,f_kma,output_path):
    '''
    Plot benchmarking resutls by antibiotics. Only those antibiotics that are with data of multi-species.

    '''

    data = pd.read_csv('./data/PATRIC/meta/'+str(level)+'_multi-species_summary.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")

    if f_phylotree:
        list_species=data.index.tolist()[1:-1]#MT no data.

    else:
        list_species = data.index.tolist()[:-1]
    data = data.loc[list_species, :]
    data = data.loc[:, (data != 0).any(axis=0)]



    fig, axs = plt.subplots(5,4,figsize=(20, 25))
    plt.tight_layout(pad=6)
    fig.subplots_adjust(wspace=0.25, hspace=0.5, top=0.95, bottom=0.08)
    blue=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
    # orange=(1.0, 0.4980392156862745, 0.054901960784313725)
    green= (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
    purp=(0.5803921568627451, 0.403921568627451, 0.7411764705882353)
    # red=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392)
    red='#ef3b2c'
    brown=(0.5490196078431373, 0.33725490196078434, 0.29411764705882354)
    colors = [blue,"orange", purp , green , red, brown]# #ffd343


    #Add acronym
    with open('./data/AntiAcronym_dict.json') as f:
        map_acr = json.load(f)
    i=0
    df_s = data.T.dot(data.T.columns + ';').str.rstrip(';')#get anti names  marked with 1

    All_antibiotics = data.columns.tolist()
    for anti in All_antibiotics:
        species_list_sub=df_s[anti].split(';')
        species_list_each = copy.deepcopy(species_list_sub)
        if f_phylotree and 'Mycobacterium tuberculosis' in species_list_sub:
            species_list_each.remove('Mycobacterium tuberculosis')



        width = 0.4/3
        row = (i // 4)
        col = i % 4
        i+=1
        x_lable = np.arange(len(species_list_each)) # the label locations

        j=0
        position=[ -5*width/2,-3*width/2,- width/2, width/2, 3*width/2, 5*width/2]
        for tool in tool_list:
            data_mean, data_std=combine_data_ByAnti(species_list_each,anti,fscore, f_phylotree, f_kma,tool,output_path)
            axs[row, col].bar(x_lable + position[j], data_mean, width, yerr=data_std, align='center', alpha=1, ecolor='black', color=colors[j],capsize=3,label=tool)
            j+=1


        axs[row, col].set_xticklabels(x_lable)
        anti=anti+'('+map_acr[anti]+')'
        axs[row, col].set_title(anti, weight='bold',size=18)
        axs[row, col].set_xticks(x_lable)
        species_list_each=[x[0] +". "+ x.split(' ')[1] for x in species_list_each]
        if anti in[ 'tetracycline(TE)']:
            axs[row, col].set_xticklabels(species_list_each, rotation=30,size=18, horizontalalignment='right',style='italic')
        elif anti in['gentamicin(GM)' ]:
            axs[row, col].set_xticklabels(species_list_each, rotation=20,size=18, horizontalalignment='center',style='italic')
        elif anti in['amikacin(AN)','ciprofloxacin(CIP)' ]:
            axs[row, col].set_xticklabels(species_list_each, rotation=15,size=18, horizontalalignment='center',style='italic')
        else:
            axs[row, col].set_xticklabels(species_list_each, rotation=10, size=18,horizontalalignment='center',style='italic')


        axs[row, col].set(ylim=(0, 1.1))
        plt.yticks([0,0.2,0.4,0.6,0.8, 1])
        axs[row, col].set_xlabel('')
        axs[row, col].set_ylabel(fscore,size = 18)


        if i!=2:

            pass
        else:
            axs[row, col].legend()

            if f_kma:
                axs[row, col].legend(bbox_to_anchor=(3.5,1.3), ncol=8,fontsize=18,frameon=False)
            if f_phylotree:
                axs[row, col].legend(bbox_to_anchor=(3.5,1.3), ncol=8,fontsize=18,frameon=False)
            else:
                axs[row, col].legend(bbox_to_anchor=(3,5.3), ncol=8,fontsize=18,frameon=False)
    fig.savefig(output_path+'Results/supplement_figures_tables/S2_ByAnti_'+'kma_'+str(f_kma)+'_tree_'+str(f_phylotree)+'_'+fscore+'.pdf')
