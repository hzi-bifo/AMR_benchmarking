import sys
import os
sys.path.insert(0, os.path.abspath('../'))
import numpy as np
import copy
import matplotlib.pyplot as plt
import json
import pandas as pd
from src.benchmark_utility.lib.CombineResults import combine_data_ByAnti


''' Compare performance on Antibitoics shared by multiple species. e.g. E. coli-ATM and K. pneumoniae-ATM combinations.'''




def ComByAnti(level,tool_list,fscore, f_phylotree,f_kma,output_path):
    '''
    Plot benchmarking resutls by antibiotics. Only those antibiotics that are with data of multi-species.

    '''




    fig, axs = plt.subplots(1,1,figsize=(10, 10))
    plt.tight_layout(pad=5)
    # fig, axs = plt.subplots(5,4,figsize=(20, 25))
    # plt.tight_layout(pad=6)
    # fig.subplots_adjust(wspace=0, hspace=0, top=0.9, bottom=0.1)
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



    anti='aztreonam'
    # species_list_sub=df_s[anti].split(';')
    species_list_sub=['Escherichia coli', 'Klebsiella pneumoniae']
    species_list_each = copy.deepcopy(species_list_sub)
    if f_phylotree and 'Mycobacterium tuberculosis' in species_list_sub:
        species_list_each.remove('Mycobacterium tuberculosis')





    x_lable = np.arange(len(species_list_each)) # the label locations

    j=0
    width = 0.4/3
    position=[ -5*width/2,-3*width/2,- width/2, width/2, 3*width/2, 5*width/2]
    for tool in tool_list:
        data_mean, data_std=combine_data_ByAnti(species_list_each,anti,fscore, f_phylotree, f_kma,tool,output_path)
        axs.bar(x_lable + position[j], data_mean, width, yerr=data_std, align='center', alpha=1, ecolor='black', color=colors[j],error_kw=dict(lw=5, capsize=5, capthick=5),label=tool)
        j+=1


    axs.set_xticklabels(x_lable)
    anti=anti+'('+map_acr[anti]+')'
    axs.set_title(anti, weight='bold',size=25)
    axs.set_xticks(x_lable)
    species_list_each=[x[0] +". "+ x.split(' ')[1] for x in species_list_each]
    if anti in[ 'tetracycline(TE)']:
        axs.set_xticklabels(species_list_each, rotation=30,size=20, horizontalalignment='right',style='italic')
    elif anti in['gentamicin(GM)' ]:
        axs.set_xticklabels(species_list_each, rotation=20,size=20, horizontalalignment='center',style='italic')
    elif anti in['amikacin(AN)','ciprofloxacin(CIP)' ]:
        axs.set_xticklabels(species_list_each, rotation=15,size=20, horizontalalignment='center',style='italic')
    else:
        axs.set_xticklabels(species_list_each, rotation=10, size=25,horizontalalignment='center',style='italic')


    axs.set(ylim=(0, 1.01))
    plt.yticks([0,0.2,0.4,0.6,0.8, 1])
    axs.set_xlabel('')
    axs.set_ylabel(fscore,size =25)



    axs.legend()


    axs.legend(ncol=1,fontsize=18,frameon=True,loc='upper right',bbox_to_anchor=(1, 1)) #bbox_to_anchor=(0.2,1.01),
    fig.savefig(output_path+'Results/final_figures_tables/F4_ByAnti_'+'kma_'+str(f_kma)+'_tree_'+str(f_phylotree)+'_'+fscore+'_eg.pdf')
    fig.savefig(output_path+'Results/final_figures_tables/F4_ByAnti_'+'kma_'+str(f_kma)+'_tree_'+str(f_phylotree)+'_'+fscore+'_eg.png')
