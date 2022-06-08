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
import pickle
import pandas as pd
import seaborn as sns

def rearrange(mylist):
    newlist=[]
    for i in mylist:
        newlist.insert(0,i)

    return newlist

def extract_info(level,s, f_all ):
    data = pd.read_csv('metadata/' + str(level) + '_Species_antibiotic_FineQuality.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    if f_all == False:#should not use it.
        data = data.loc[s, :]
    species_list=['Escherichia coli','Staphylococcus aureus','Salmonella enterica','Klebsiella pneumoniae','Pseudomonas aeruginosa','Acinetobacter baumannii','Streptococcus pneumoniae', 'Campylobacter jejuni','Enterococcus faecium','Neisseria gonorrhoeae']
    # species_list=['Escherichia coli','Staphylococcus aureus']

    #rearrange the list
    species_list=rearrange(species_list)

    # data=data.loc[['Escherichia coli','Staphylococcus aureus','Salmonella enterica','Klebsiella pneumoniae','Pseudomonas aeruginosa','Acinetobacter baumannii','Streptococcus pneumoniae','Mycobacterium tuberculosis'],:]
    data_main=data.loc[species_list,:]
    antibiotics_main = data_main['modelling antibiotics'].tolist()
    # print(data_main)
    mt=['Mycobacterium tuberculosis']
    data_mt=data.loc[species_list,:]
    antibiotics_mt = data_mt['modelling antibiotics'].tolist()

    df_plot = pd.DataFrame(columns=[ 'anti', 'Resistant','Susceptible'])

    fig, axs = plt.subplots(2,1,figsize=(20, 30), gridspec_kw={"height_ratios":[5.5, 1]})
    plt.tight_layout(pad=4)
    fig.subplots_adjust(left=0.48,  right=0.96,wspace=0.25, hspace=0.1, top=0.98, bottom=0.02)
    for species, antibiotics_selected in zip(species_list, antibiotics_main):
        df=pd.read_csv('metadata/balance/'+str(level)+'/log_' + str(species.replace(" ", "_")) + '_pheno_summary' + '.txt', sep="\t" )

        df=df.rename(columns={"Unnamed: 0": "anti"})
        df=df[['anti','Resistant','Susceptible']]
        df_plot=df_plot.append(df, sort=False)

    df_plot=df_plot.reset_index(drop=True)
    # print(df_plot)


    #add acronym
    with open('./src/AntiAcronym_dict.pkl', 'rb') as f:
        map_acr = pickle.load(f)
    pd_mech=pd.read_csv('./src/acronym_mech.csv')

    map_mech= dict(zip(pd_mech['Antibiotic'], pd_mech['Mechanism classification by target site']))
        # pd_mech[['Antibiotic','Mechanism classification by target site']].to_dict()
    # print(map_mech)
    df_plot['anti_new']=df_plot['anti'].apply(lambda x:x+'('+ map_acr[x]+')'+'['+map_mech[x]+']')
    # print(df_plot)


    # fig = plt.figure(figsize=(40, 10))

    ax=df_plot.plot(
    x = 'anti_new',
    kind = 'barh',
    stacked = True,
    mark_right = True,ax=axs[0])
    fig=ax.get_figure()
    ax.yaxis.set_label_text('')
    ax.legend(bbox_to_anchor=(0.9, 1.03 ),ncol=2,fontsize=24,frameon=False)
    # ax.set_ylabel(fscore,size=16)
    ax.set_yticklabels(ax.get_yticklabels(),fontsize = 20 )
    ax.tick_params(axis='x', which='major', labelsize=20)

    #plot MT
    df=pd.read_csv('metadata/balance/'+str(level)+'/log_' + str(mt[0].replace(" ", "_")) + '_pheno_summary' + '.txt', sep="\t" )
    df=df.rename(columns={"Unnamed: 0": "anti"})
    df=df[['anti','Resistant','Susceptible']]
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
    # plt.annotate(r"$\{$",fontsize=500,
    #         xy=(0.2, 0.9), xycoords='figure fraction',xytext =(0.01, 0.77))
    # fig.savefig('log/results/samplesize.eps', format='eps',dpi=1200)
    fig.savefig('log/results/samplesize.png')

