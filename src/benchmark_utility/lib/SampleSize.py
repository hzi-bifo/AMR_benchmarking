#!/usr/bin/python
import sys
import os
sys.path.insert(0, os.path.abspath('../'))
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd


def rearrange(mylist):
    newlist=[]
    for i in mylist:
        newlist.insert(0,i)
    return newlist


def AddRatio(df,ax,var1,var2):
    df_total = df[var1]+df[var2]
    df = df.iloc[:, 0:4]
    df_rel = df[[var1,var2]].min(axis=1) /df[[var1,var2]].max(axis=1)
    for i, tot in enumerate( df_total):
        ax.text(tot, i, np.round(df_rel[i],2), va='center',size=20)
    return df_rel
def extract_info(level,save_file_name ):

    species_list=['Escherichia coli','Staphylococcus aureus','Salmonella enterica','Klebsiella pneumoniae','Pseudomonas aeruginosa',
                  'Acinetobacter baumannii','Streptococcus pneumoniae', 'Campylobacter jejuni','Enterococcus faecium',
                  'Neisseria gonorrhoeae']

    #1st subplot rearrange the list
    species_list=rearrange(species_list)


    #2nd subplot, due to varied sample size between MT combinations and the others.
    mt=['Mycobacterium tuberculosis']


    df_plot = pd.DataFrame(columns=[ 'anti', 'Resistant','Susceptible'])

    fig, axs = plt.subplots(2,1,figsize=(20, 30), gridspec_kw={"height_ratios":[5.5, 1]})
    plt.tight_layout(pad=4)
    fig.subplots_adjust(left=0.48,  right=0.96,wspace=0.25, hspace=0.1, top=0.98, bottom=0.02)
    for species in  species_list :
        df=pd.read_csv('./data/PATRIC/meta/'+str(level)+'_genomeNumber/log_' + str(species.replace(" ", "_")) + '_pheno_summary' + '.txt', sep="\t" )

        df=df.rename(columns={"Unnamed: 0": "anti"})
        df=df[['anti','Resistant','Susceptible']]
        # reverse the order of antibiotics list. to make it consistent with other tables.
        df=df.loc[::-1]

        df_plot=df_plot.append(df, sort=False)

    df_plot=df_plot.reset_index(drop=True)


    #add acronym
    with open('./data/AntiAcronym_dict.json') as f:
                map_acr = json.load(f)
    pd_mech=pd.read_csv('./data/acronym_mech.csv')
    map_mech= dict(zip(pd_mech['Antibiotic'], pd_mech['Mechanism classification by target site']))
    df_plot['anti_new']=df_plot['anti'].apply(lambda x:x+'('+ map_acr[x]+')'+'['+map_mech[x]+']')


    ax=df_plot.plot(
    x = 'anti_new',
    kind = 'barh',
    stacked = True,
    mark_right = True,ax=axs[0])
    fig=ax.get_figure()
    ax.yaxis.set_label_text('')
    ax.legend(bbox_to_anchor=(0.9, 1.03 ),ncol=2,fontsize=24,frameon=False)
    ax.set_yticklabels(ax.get_yticklabels(),fontsize = 20 )
    ax.tick_params(axis='x', which='major', labelsize=20)

    #add ratio
    df_rel=AddRatio(df_plot,axs[0],'Resistant','Susceptible')
    count_ratio=df_rel[df_rel<0.5].count()


    ##------------------------------------------------------------------------------------------------------------------------
    #plot MT
    df=pd.read_csv('./data/PATRIC/meta/'+str(level)+'_genomeNumber/log_' + str(mt[0].replace(" ", "_")) + '_pheno_summary' + '.txt', sep="\t" )
    df=df.rename(columns={"Unnamed: 0": "anti"})
    df=df[['anti','Resistant','Susceptible']]
    df=df.loc[::-1]
    df['anti_new']=df['anti'].apply(lambda x:x+'('+ map_acr[x]+')'+'['+map_mech[x]+']')

    ax=df.plot(
    x = 'anti_new',
    kind = 'barh',
    stacked = True,
    mark_right = True,ax=axs[1])
    fig=ax.get_figure()
    ax.yaxis.set_label_text('')
    ax.get_legend().remove()
    ax.set_yticklabels(ax.get_yticklabels(),fontsize = 20 )
    ax.tick_params(axis='x', which='major', labelsize=20)
    ax.set(xlim=(0, 14000))
    df_rel=AddRatio(df,axs[1],'Resistant','Susceptible')


    print('Number of combinations with im balance ratio < 0.5:',df_rel[df_rel<0.5].count()+count_ratio)
    fig.savefig(save_file_name)

