import sys
import os,ast,json
sys.path.insert(0, os.path.abspath('../'))
from src.amr_utility import name_utility
import numpy as np
from src.benchmark_utility.lib.CombineResults import  combine_data_meanstd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.patches as mpatches


def change_layout(data_plot,fscore,species):
    if species !='Mycobacterium tuberculosis' :
        data=pd.DataFrame(columns=['Random folds', 'Phylogeny-aware folds','Homology-aware folds'])
        data_plot=data_plot[(data_plot['species'] == species)] #newly added

        data1=data_plot[(data_plot['folds'] == 'Random folds')]
        data2=data_plot[(data_plot['folds'] == 'Phylogeny-aware folds')]
        data3=data_plot[(data_plot['folds'] == 'Homology-aware folds')]
        data['Random folds']=data1[fscore]
        data['Phylogeny-aware folds']=data2[fscore]
        data['Homology-aware folds']=data3[fscore]
    else:
        data=pd.DataFrame(columns=['Random folds', 'Phylogeny-aware folds','Homology-aware folds'])

        data_plot=data_plot[(data_plot['species'] == species)]#newly added

        data1=data_plot[(data_plot['folds'] == 'Random folds')]
        ### data2=data_plot[(data_plot['folds'] == 'Phylogeny-aware folds')]
        data3=data_plot[(data_plot['folds'] == 'Homology-aware folds')]
        data['Random folds']=data1[fscore]
        # data['Phylogeny-aware folds']= np.nan
        data['Homology-aware folds']=data3[fscore]
    return data
# def change_layoutByTool(data_plot,fscore):
#     '''connect dots representing the same tool+anti combination'''
#     data=pd.DataFrame(columns=['species','anti','Random folds', 'Phylogeny-aware folds','Homology-aware folds'])
#     df = data_plot.reset_index()
#
#     df=df.groupby(['species', 'antibiotics']).size().reset_index().rename(columns={0:'count'})
#
#     data['species']=df['species']
#     data['anti']=df['antibiotics']
#     for index, row in data.iterrows():
#         s=row['species']
#         a=row['anti']
#
#         if s!='Mycobacterium tuberculosis':
#                 row['Phylogeny-aware folds']=data_plot[(data_plot['species'] ==s)&(data_plot['antibiotics'] ==a)&(data_plot['folds'] =='Phylogeny-aware folds')].iloc[0][fscore]
#         row['Random folds']=data_plot[(data_plot['species'] ==s)&(data_plot['antibiotics'] ==a)&(data_plot['folds'] =='Random folds')].iloc[0][fscore]
#         row['Homology-aware folds']=data_plot[(data_plot['species'] ==s)&(data_plot['antibiotics'] ==a)&(data_plot['folds'] =='Homology-aware folds')].iloc[0][fscore]
#
#     data_mt=data[(data['species'] == 'Mycobacterium tuberculosis')]
#     data_else=data[(data['species'] != 'Mycobacterium tuberculosis')]
#     return data,data_else,data_mt
# def ranking(data_plot):
#     df = data_plot.reset_index()
#     df['anti+folds']=df['antibiotics']+'\n'+ df['folds']
#     df=df.sort_values(['antibiotics','folds'], ascending = (True,False ))
#     return df
# def change_layoutWithinSpecies(data_plot,fscore):
#     col=data_plot.groupby(by=["anti+folds"],sort=False).count().index.to_list()
#     data=pd.DataFrame(columns=col)
#     for c in col:
#         data_sub=data_plot[(data_plot["anti+folds"] == c)]
#         data[c]=data_sub[fscore].to_list()
#     return data




def extract_info(level,s, fscore,f_all,f_step,f_mean_std,output_path):
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
    antibiotics_whole = data['modelling antibiotics'].tolist()

    # get the list of all antibiotics
    antibiotics_unique=[]
    for antibiotics in antibiotics_whole:
        antibiotics = ast.literal_eval(antibiotics)

        for anti in antibiotics:
            if anti not in antibiotics_unique:
                antibiotics_unique.append(anti)


    print(antibiotics_unique)
    print(len(antibiotics_unique))



    foldset=['Random folds', 'Phylogeny-aware folds','Homology-aware folds']
    if f_mean_std=='mean':
        flag='_PLOT'
    elif f_mean_std=='std':
        flag='_std'
    else:
        print('Wrong parameters set, please reset.')
        exit(1)
    np.random.seed(0)

    Summary_table= pd.DataFrame(columns=['antibiotic']+foldset)#summary table for antibiotic rankings.
    Summary_table_md= pd.DataFrame(columns=['antibiotic']+foldset)#summary table for antibiotic rankings.
    # --------------
    #1. by antibiotic
    # --------------

    tool_list=[ 'Aytan-Aktug', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover']

    fig, axs = plt.subplots(9,5,figsize=(25, 34))
    plt.tight_layout(pad=4)
    fig.subplots_adjust(wspace=0.1, hspace=0.25, top=0.90, bottom=0.03,left=0.06)
    # fig.suptitle('A', weight='bold',size=35,x=0.02,y=0.97 )

    palette_tab10 = sns.color_palette("tab10", 10)
    palette_tab10.append('#01665e')

    i=0
    # for species, antibiotics_selected in zip(df_species, antibiotics):
    for each_anti in antibiotics_unique:
        # print(each_anti)
        species_p=[]
        #get the species that was subscribed by this grug
        for species in df_species:
            anti_temp=data.loc[species,"modelling antibiotics"]
            anti_temp=ast.literal_eval(anti_temp)

            if each_anti in anti_temp:
                species_p.append(species)


        row = (i //5)
        col = i % 5
        i+=1

        data_plot=combine_data_meanstd(species_p,level,fscore,tool_list,foldset,output_path,flag)
        # print(data_plot)
        #[fscore, 'species', 'software','anti','folds']
        data_plot=data_plot.replace(np. nan,0) # add in zeros into the plots!!! April 2023.

        data_plot=data_plot[data_plot["antibiotics"] == each_anti]
        # print(data_plot)
        ####summary table for antibiotic rankings.#################################################################
        average_table=data_plot.groupby(['folds']).mean()
        # print(average_table)
        if 'Phylogeny-aware folds' in average_table.index.tolist():
            row_table= [each_anti,average_table.loc['Random folds',fscore],average_table.loc['Phylogeny-aware folds',fscore],average_table.loc['Homology-aware folds',fscore]]
        else:
            row_table= [each_anti,average_table.loc['Random folds',fscore],np.nan,average_table.loc['Homology-aware folds',fscore]]


        Summary_table.loc[i] =row_table

        ##########################################################################################################
        ####summary table for antibiotic rankings.#################################################################
        average_table_md=data_plot.groupby(['folds']).median()
        # print(average_table)
        if 'Phylogeny-aware folds' in average_table.index.tolist():
            row_table= [each_anti,average_table_md.loc['Random folds',fscore],average_table_md.loc['Phylogeny-aware folds',fscore],average_table_md.loc['Homology-aware folds',fscore]]
        else:
            row_table= [each_anti,average_table_md.loc['Random folds',fscore],np.nan,average_table_md.loc['Homology-aware folds',fscore]]


        Summary_table_md.loc[i] =row_table
        #
        ##########################################################################################################


        data_plot= data_plot.astype({fscore:float})
        if species_p==['Mycobacterium tuberculosis']:
            data_plot.loc[len(data_plot)]=[np.nan,'Mycobacterium tuberculosis','',each_anti,'Phylogeny-aware folds']
        # ax = sns.violinplot(x="folds", y=fscore, ax=axs[row, col],data=data_plot,
        #         inner=None, color="0.95",order=['Random folds', 'Phylogeny-aware folds','Homology-aware folds'])
        ax=sns.boxplot(data=data_plot, x="folds", ax=axs[row, col],y=fscore,\
                       order=['Random folds', 'Phylogeny-aware folds','Homology-aware folds'])
        for i_temp,box in enumerate(ax.artists):
            box.set_edgecolor('black')
            box.set_facecolor('white')
            # iterate over whiskers and median lines
            for j in range(6*i,6*(i_temp+1)):
                 ax.lines[j].set_color('black')

        if f_mean_std=='mean':
            ax.set(ylim=(0, 1.0))
        else:
            ax.set(ylim=(0, 0.5))

        if col==0:
            ax.set_ylabel(fscore.replace("_", "-").capitalize(),size = 25)
            ax.tick_params(axis='y', which='major', labelsize=20)
        else:
            ax.set_yticks([])
            ax.set(ylabel=None)
            ax.tick_params(axis='y',bottom=False)

        # if row==8:
        #     # ax.tick_params(axis='x', which='major', labelsize=20,rotate=10)
        #     ax.set_xticklabels(ax.get_xticklabels(),rotation=20,fontsize=25,ha='right')
        #     # ax.xticks(rotation=10)

        #Add acronym
        with open('./data/AntiAcronym_dict.json') as f:
            map_acr = json.load(f)

        each_anti_acr=map_acr[each_anti]
        ax.set_title(each_anti_acr, weight='bold',size=31,pad=10)
        # ax.set(xticklabels=[])
        # ax.set(xlabel=None)
        # ax.tick_params(axis='x',bottom=False)
        #--
        #connect dots representing the same tool+species combination
        for species in species_p:
            # print(species)
            # print(data_plot)
            df=change_layout(data_plot,fscore,species)
            # if species=='Mycobacterium tuberculosis':
            #     # print(df)

            jitter = 0.05
            df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=df.values.shape), columns=df.columns)
            df_x_jitter += np.arange(len(df.columns))
            df=df.set_index(df_x_jitter.index)

            # if species=='Mycobacterium tuberculosis':
            #     print(df)


            i_color=0
            for col_t in df:

                if species != 'Mycobacterium tuberculosis':
                    if i_color%3==0:
                        ax.plot(df_x_jitter[col_t], df[col_t], 'o',c=palette_tab10[species_list.index(species)], alpha=0.8, zorder=1, ms=8, mew=1, label="Random folds")
                    elif i_color%3==1:
                        ax.plot(df_x_jitter[col_t], df[col_t], 'v',c=palette_tab10[species_list.index(species)], alpha=0.8, zorder=1, ms=8, mew=1, label="Phylogeny-aware folds")
                    else:
                        ax.plot(df_x_jitter[col_t], df[col_t], 's', c=palette_tab10[species_list.index(species)], alpha=0.8, zorder=1, ms=8, mew=1, label="Homology-aware folds")

                else:
                    if i_color%3==0:
                        ax.plot(df_x_jitter[col_t], df[col_t], 'o',c=palette_tab10[species_list.index(species)], alpha=0.8, zorder=1, ms=8, mew=1,label="Random folds")
                    elif i_color%3==2:
                        # print('!!!')
                        ax.plot(df_x_jitter[col_t], df[col_t], 's',c=palette_tab10[species_list.index(species)],alpha=0.8, zorder=1, ms=8, mew=1, label="Homology-aware folds")



                i_color+=1
                if f_mean_std=='mean':
                    ax.set(ylim=(0, 1.0))
                else:
                    ax.set(ylim=(0, 0.5))
                if col==0:
                    ax.set_ylabel(fscore.replace("_", "-").capitalize(),size = 25)
                    ax.tick_params(axis='y', which='major', labelsize=25)
                else:
                    ax.set_yticks([])
                    ax.set(ylabel=None)
                    ax.tick_params(axis='y',bottom=False)

                if each_anti=='amoxicillin' and (i_color in [1,2,3]):



                    if f_mean_std=='mean':

                        leg=ax.legend(bbox_to_anchor=(0.8, 1.2),ncol=3,fontsize=30,frameon=False, markerscale=2)
                        # leg2=ax.legend(bbox_to_anchor=(0.8, 1.11),ncol=2,fontsize=20,frameon=False, markerscale=2)

                    else:
                        leg=ax.legend(bbox_to_anchor=(4, 1.38),ncol=3,fontsize=30,frameon=False, markerscale=2)

                    # leg.legendHandles[0].set_color('black')
                    ax.add_artist(leg)
                    leg = ax.get_legend()
                    # leg.legendHandles[0].set_color('black')
                    # leg.legendHandles[1].set_color('black')
                    # ax.add_artist(leg2)
                    for lh in leg.legendHandles:
                        lh._legmarker.set_color('black')
                        # lh._legmarker.set_alpha(1)


            # if row <8:
            ax.set(xticklabels=[])
            ax.set(xlabel=None)
            ax.tick_params(axis='x',bottom=False)




            if species !='Mycobacterium tuberculosis':
                for idx in df.index:
                    ax.plot(df_x_jitter.loc[idx,['Random folds','Phylogeny-aware folds']], df.loc[idx,['Random folds','Phylogeny-aware folds']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=1)
                    ax.plot(df_x_jitter.loc[idx,['Phylogeny-aware folds','Homology-aware folds']], df.loc[idx,['Phylogeny-aware folds','Homology-aware folds']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=1)
            else:
                for idx in df.index:
                    ax.plot(df_x_jitter.loc[idx,['Random folds','Homology-aware folds']], df.loc[idx,['Random folds','Homology-aware folds']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=1)

            ax.set(xlabel=None)
            ax.tick_params(axis='x',bottom=False)

    species_list_s=[(species[0] +". "+ species.split(' ')[1] ) for species in species_list]
    legend_dict=dict(zip(species_list_s, palette_tab10))

    patchList = []
    for key in legend_dict:
        data_key = mpatches.Patch(color=legend_dict[key], label=key)
        patchList.append(data_key)

    plt.legend(handles=patchList,bbox_to_anchor=(0.9, 12.2), ncol=6,fontsize=25,frameon=False,prop={'size': 25, 'style': 'italic'})

    fig.savefig(output_path+'Results/supplement_figures_tables/S2_byAnti_'+f_mean_std+'.png')
    fig.savefig(output_path+'Results/supplement_figures_tables/S2_byAnti_'+f_mean_std+'.pdf')


    ####summary table for antibiotic rankings.#################################################################
    print(Summary_table)
    Summary_table=Summary_table.sort_values('Random folds',ascending=False)
    print(Summary_table)
    anti_acro= [map_acr[x] for x in Summary_table['antibiotic'].tolist()]
    # print(anti_acro)
    Summary_table.insert(loc=1, column='Antibiotic', value=anti_acro)
    print(Summary_table)
    Summary_table.to_csv(output_path+ 'Results/other_figures_tables/antibiotic_ranking.csv', sep="\t")


    ###########################################################################################################
     ####summary table for antibiotic rankings.#################################################################
    print(Summary_table_md)
    Summary_table_md=Summary_table_md.sort_values('Random folds',ascending=False)
    print(Summary_table_md)
    anti_acro= [map_acr[x] for x in Summary_table_md['antibiotic'].tolist()]
    # print(anti_acro)
    Summary_table_md.insert(loc=1, column='Antibiotic', value=anti_acro)
    print(Summary_table)
    Summary_table_md.to_csv(output_path+ 'Results/other_figures_tables/antibiotic_MDranking.csv', sep="\t")
