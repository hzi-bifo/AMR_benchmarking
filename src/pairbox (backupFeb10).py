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

def combine_data(species_list,fscore,tool_list):

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
        for species in species_list:
            if species !='Mycobacterium tuberculosis' or f_phylotree==False:
                antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, 'loose')
                for anti in antibiotics:

                    for tool in tool_list:
                        # if tool=='Point-/ResFinder':
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


                        #[fscore, 'species', 'software','anti','folds']
                        df_plot_sub.loc['s'] = [score,species,tool,anti,fold]
                        df_plot = df_plot.append(df_plot_sub, sort=False)
    return df_plot
def change_layout(data_plot,fscore,species):
    if species !='Mycobacterium tuberculosis' :
        data=pd.DataFrame(columns=['random folds', 'phylo-tree-based folds','KMA-based folds'])
        data1=data_plot[(data_plot['folds'] == 'random folds')]
        data2=data_plot[(data_plot['folds'] == 'phylo-tree-based folds')]
        data3=data_plot[(data_plot['folds'] == 'KMA-based folds')]
        data['random folds']=data1[fscore]
        data['phylo-tree-based folds']=data2[fscore]
        data['KMA-based folds']=data3[fscore]
    else:
        data=pd.DataFrame(columns=['random folds', 'KMA-based folds'])
        data1=data_plot[(data_plot['folds'] == 'random folds')]
        # data2=data_plot[(data_plot['folds'] == 'phylo-tree-based folds')]
        data3=data_plot[(data_plot['folds'] == 'KMA-based folds')]
        data['random folds']=data1[fscore]
        # data['phylo-tree-based folds']=data2[fscore]
        data['KMA-based folds']=data3[fscore]
    return data
def change_layoutByTool(data_plot,fscore,tool):

    data=pd.DataFrame(columns=['species','anti','random folds', 'phylo-tree-based folds','KMA-based folds'])
    df = data_plot.reset_index()

    df=df.groupby(['species', 'anti']).size().reset_index().rename(columns={0:'count'})
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(df)

    data['species']=df['species']
    data['anti']=df['anti']
    for index, row in data.iterrows():
        s=row['species']
        a=row['anti']

        if s!='Mycobacterium tuberculosis':
                row['phylo-tree-based folds']=data_plot[(data_plot['species'] ==s)&(data_plot['anti'] ==a)&(data_plot['folds'] =='phylo-tree-based folds')].iloc[0][fscore]
        row['random folds']=data_plot[(data_plot['species'] ==s)&(data_plot['anti'] ==a)&(data_plot['folds'] =='random folds')].iloc[0][fscore]
        row['KMA-based folds']=data_plot[(data_plot['species'] ==s)&(data_plot['anti'] ==a)&(data_plot['folds'] =='KMA-based folds')].iloc[0][fscore]

    data_mt=data[(data['species'] == 'Mycobacterium tuberculosis')]
    data_else=data[(data['species'] != 'Mycobacterium tuberculosis')]
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(data)
    # exit()
    return data,data_else,data_mt

def ranking(data_plot,fscore,species):
    df = data_plot.reset_index()
    df['anti+folds']=df['anti']+'\n'+ df['folds']
    df=df.sort_values(['anti','folds'], ascending = (True,False ))
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    return df
def change_layoutWithinSpecies(data_plot,fscore,species):
    col=data_plot.groupby(by=["anti+folds"],sort=False).count().index.to_list()
    data=pd.DataFrame(columns=col)
    for c in col:
        data_sub=data_plot[(data_plot["anti+folds"] == c)]
        data[c]=data_sub[fscore].to_list()
    return data





def extract_info(level,s, fscore, cv_number, f_phylotree, f_kma,f_all,f_step):
    data = pd.read_csv('metadata/' + str(level) + '_Species_antibiotic_FineQuality.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    if f_all == False:#should not use it.
        data = data.loc[s, :]
    species_list=['Escherichia coli','Staphylococcus aureus','Salmonella enterica','Klebsiella pneumoniae','Pseudomonas aeruginosa','Acinetobacter baumannii','Streptococcus pneumoniae','Mycobacterium tuberculosis', 'Campylobacter jejuni','Enterococcus faecium','Neisseria gonorrhoeae']
    # data=data.loc[['Escherichia coli','Staphylococcus aureus','Salmonella enterica','Klebsiella pneumoniae','Pseudomonas aeruginosa','Acinetobacter baumannii','Streptococcus pneumoniae','Mycobacterium tuberculosis'],:]
    data=data.loc[species_list,:]
    # data=data.loc[['Escherichia coli','Staphylococcus aureus','Salmonella enterica','Klebsiella pneumoniae','Pseudomonas aeruginosa','Acinetobacter baumannii','Streptococcus pneumoniae', 'Campylobacter jejuni','Enterococcus faecium','Neisseria gonorrhoeae'],:]
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    # tool_list=['Point-/ResFinder', 'Neural networks', 'KmerC', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover']
    tool_list=[ 'Neural networks', 'KmerC', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover']
    # tool_list=['KmerC', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover']
    # --------------
    #1. Is it different for species?
    # --------------
    if f_step=='1':
        fig, axs = plt.subplots(3,4,figsize=(30, 25))
        plt.tight_layout(pad=4)
        fig.subplots_adjust(wspace=0.25, hspace=0.3, top=0.95, bottom=0.04)
        title='Performance change w.r.t. different folds ('+fscore+')'
        labels = tool_list
        # fig.suptitle(title,size=19, weight='bold')
        fig.suptitle('A', weight='bold',size=35,x=0.01)

        axs[2,3].axis('off')
        i=0
        for species, antibiotics_selected in zip(df_species, antibiotics):
            print(species)

            row = (i //4)
            col = i % 4
            i+=1
            species_p=[species]
            data_plot=combine_data(species_p,fscore,tool_list)
            #[fscore, 'species', 'software','anti','folds']
            print(data_plot)
            data_plot= data_plot.astype({fscore:float})
            ax = sns.violinplot(x="folds", y=fscore, ax=axs[row, col],data=data_plot,
                        inner=None, color="0.95")
            ax.set(ylim=(0, 1.0))
            ax.set_xticklabels(ax.get_xticks(), size = 17,rotation=10)
            # ax = sns.stripplot(x="folds", y=fscore,ax=axs[row, col],  data=data_plot)
            trans = ax.get_xaxis_transform()
            ax.set_xlabel('')
            ax.set_title(species,style='italic', weight='bold',size=22)

            #--
            #connect dots representing the same tool+anti combination
            df=change_layout(data_plot,fscore,species)


            jitter = 0.05
            df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=df.values.shape), columns=df.columns)
            df_x_jitter += np.arange(len(df.columns))
            df=df.set_index(df_x_jitter.index)
            # print(df)
            # print(df_x_jitter)

            j=0
            # colors=['b','g']
            for col in df:
                if species !='Mycobacterium tuberculosis':
                    ax.plot(df_x_jitter[col], df[col], 'o', alpha=.40, zorder=1, ms=8, mew=1)
                else:

                    if j==1:
                        ax.plot(df_x_jitter[col], df[col], 'go', alpha=.40, zorder=1, ms=8, mew=1 )
                    else:
                        ax.plot(df_x_jitter[col], df[col], 'o', alpha=.40, zorder=1, ms=8, mew=1)
                    j+=1
            ax.set_xticks(range(len(df.columns)))
            ax.set_xticklabels(df.columns)
            ax.set_xlim(-0.5,len(df.columns)-0.5)
            if species !='Mycobacterium tuberculosis':
                for idx in df.index:
                    ax.plot(df_x_jitter.loc[idx,['random folds','phylo-tree-based folds']], df.loc[idx,['random folds','phylo-tree-based folds']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
                    ax.plot(df_x_jitter.loc[idx,['phylo-tree-based folds','KMA-based folds']], df.loc[idx,['phylo-tree-based folds','KMA-based folds']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
            else:
                for idx in df.index:
                    ax.plot(df_x_jitter.loc[idx,['random folds','KMA-based folds']], df.loc[idx,['random folds','KMA-based folds']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)

        fig.savefig('log/results/FoldsChange_'+fscore+'.png')

    #2. Is it different for tools?
    if f_step=='2':
        fig, axs = plt.subplots(2,3,figsize=(25, 20))
        plt.tight_layout(pad=4)
        fig.subplots_adjust(wspace=0.25, hspace=0.3, top=0.92, bottom=0.05)
        title='Performance change w.r.t. different folds ('+fscore+')'
        labels = tool_list
        # fig.suptitle(title,size=17, weight='bold')
        fig.suptitle('B', weight='bold',size=35,x=0.01)
        axs[1,2].axis('off')
        i=0
        for tool in tool_list:


            row = (i //3)
            col = i % 3
            i+=1
            tool_p=[tool]
            data_plot=combine_data(df_species,fscore,tool_p)
            #[fscore, 'species', 'software','anti','folds']
            print(data_plot)
            data_plot= data_plot.astype({fscore:float})
            ax = sns.violinplot(x="folds", y=fscore, ax=axs[row, col],data=data_plot,
                        inner=None, color="0.95")
            # ax = sns.stripplot(x="folds", y=fscore,ax=axs[row, col],  data=data_plot)
            ax.set(ylim=(0, 1.0))
            ax.set_xticklabels(ax.get_xticks(), size = 17,rotation=10)
            ax.set_ylabel(fscore,size = 17)
            trans = ax.get_xaxis_transform()
            ax.set_xlabel('')
            ax.set_title(tool, weight='bold',size=22)
            #--
            #connect dots representing the same tool+anti combination
            df_whole,df_else,df_mt=change_layoutByTool(data_plot,fscore,tool)

            df=df_whole[['random folds', 'phylo-tree-based folds','KMA-based folds']]
            print(df_mt)

            jitter = 0.05
            df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=df.values.shape), columns=df.columns)
            df_x_jitter += np.arange(len(df.columns))
            df=df.set_index(df_x_jitter.index)


            for col in df:
                ax.plot(df_x_jitter[col], df[col], 'o', alpha=.40, zorder=1, ms=8, mew=1)

            ax.set_xticks(range(len(df.columns)))
            ax.set_xticklabels(df.columns)
            ax.set_xlim(-0.5,len(df.columns)-0.5)

            for idx in df_else.index:
                ax.plot(df_x_jitter.loc[idx,['random folds','phylo-tree-based folds']], df.loc[idx,['random folds','phylo-tree-based folds']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
                ax.plot(df_x_jitter.loc[idx,['phylo-tree-based folds','KMA-based folds']], df.loc[idx,['phylo-tree-based folds','KMA-based folds']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)

            for idx in df_mt.index:
                ax.plot(df_x_jitter.loc[idx,['random folds','KMA-based folds']], df.loc[idx,['random folds','KMA-based folds']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
                # ax.plot(df_x_jitter.loc[idx,['phylo-tree-based folds','KMA-based folds']], df.loc[idx,['phylo-tree-based folds','KMA-based folds']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)

        fig.savefig('log/results/FoldsChangeByTool_'+fscore+'.png')

    #2. Is it different for antis within species?
    if f_step=='3':
        # fig, axs = plt.subplots(11,1,figsize=(20, 30))
        # axs[3,2].axis('off')
        fig = plt.figure(figsize=(20, 30))
        # plt.tight_layout(pad=1,rect=(-0.5, -0.5, 1, 1))
        plt.tight_layout()
        fig.subplots_adjust(left=0.03,  right=0.98,wspace=0.1, hspace=0.3, top=0.93, bottom=0.03)
        title='Performance change w.r.t. different folds ('+fscore+')'

        fig.suptitle(title,size=17, weight='bold')
        i=0
        j=0
        for species, antibiotics_selected in zip(df_species, antibiotics):
            print(species)

            # row = (i //1)
            # col = i % 1
            species_p=[species]
            data_plot=combine_data(species_p,fscore,tool_list)
            #[fscore, 'species', 'software','anti','folds']
            # print(data_plot)
            data_plot= data_plot.astype({fscore:float})
            data_plot=ranking(data_plot,fscore,species)
            # data_plot=data_plot[[fscore,"anti+folds"]]

            # my_order = data_plot.groupby(by=["anti+folds"],sort=False).count().index.to_list()

            # print(row, col)
            # ax = sns.violinplot(x="anti+folds", y=fscore, ax=axs[row],data=data_plot,
            #             inner=None, color="0.95",order=my_order,width=2)
            # ax = sns.violinplot(x="anti+folds", y=fscore, ax=axs[row, col],data=data_plot,
            #             inner=None, color="0.95",order=my_order)
            # ax = sns.stripplot(x="anti+folds", y=fscore,ax=axs[row, col],  data=data_plot,order=my_order)
            # trans = ax.get_xaxis_transform()
            # ax.set_xticklabels(ax.get_xticklabels(), rotation=20, horizontalalignment='right')
            # ax.set_title(species,style='italic', weight='bold')
            # ax.set_xlabel('')
            #--
            #connect dots representing the same tool+anti combination
            df=change_layoutWithinSpecies(data_plot,fscore,species)
            # print(df)
            labels = df.columns.to_list()
            print(labels)
            jitter = 0.05
            df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=df.values.shape), columns=df.columns)
            df_x_jitter += np.arange(len(df.columns))
            df=df.set_index(df_x_jitter.index)
            print(df)
            # print(df_x_jitter)


            # colors=['b','g']

            if species not in ['Campylobacter jejuni','Enterococcus faecium','Neisseria gonorrhoeae']:
                i+=1
                number=910+i
                ax = plt.subplot(number)
            else:
                ax=fig.add_subplot(9,3,i*3+1+j)
                j+=1

            i_color=0
            for col in df:
                if species !='Mycobacterium tuberculosis':
                    if i_color%3==0:
                        ax.plot(df_x_jitter[col], df[col], 'o',c='tab:blue', alpha=.40, zorder=1, ms=8, mew=1, label="random folds")
                    elif i_color%3==1:
                        ax.plot(df_x_jitter[col], df[col], 'o',c='orange', alpha=.40, zorder=1, ms=8, mew=1, label="phylo-tree-based folds")
                    else:
                        ax.plot(df_x_jitter[col], df[col], 'go', alpha=.40, zorder=1, ms=8, mew=1, label="KMA-based folds")
                    # i_color+=1
                    # ax.set_title(species,style='italic', weight='bold')
                    # ax.set(ylim=(0, 1.0))
                else:
                    if i_color%2==0:
                        ax.plot(df_x_jitter[col], df[col], 'o',c='tab:blue', alpha=.40, zorder=1, ms=8, mew=1)
                    elif i_color%2==1:
                        ax.plot(df_x_jitter[col], df[col], 'go', alpha=.40, zorder=1, ms=8, mew=1)
                    # else:
                    #     ax.plot(df_x_jitter[col], df[col], 'go', alpha=.40, zorder=1, ms=8, mew=1)
                i_color+=1
                ax.set_title(species,style='italic', weight='bold')
                ax.set(ylim=(0, 1.0))
                ax.set_ylabel(fscore)
                if species=='Escherichia coli' and ( i_color in [1,2,3]):
                    ax.legend(bbox_to_anchor=(0.5, 1.09),ncol=3,fontsize=17,frameon=False)


            ax.set_xticks(range(len(df.columns)))
            ax.set_xticklabels(df.columns)
            ax.set_xlim(-0.5,len(df.columns)-0.5)
            labels_p = [item.get_text() for item in ax.get_xticklabels()]
            temp=0
            for i_label_p in labels_p:
                if temp%3==1:
                    labels_p[temp] = i_label_p.split('\n')[0]
                else:
                    labels_p[temp]=''
                temp+=1
            ax.set_xticklabels(labels_p)

            if species !='Mycobacterium tuberculosis':
                for idx in df.index:
                    for i_label in range(len(labels)):
                        if i_label%3==0:
                            ax.plot(df_x_jitter.loc[idx,labels[i_label:i_label+2]], df.loc[idx,labels[i_label:i_label+2]], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
                            ax.plot(df_x_jitter.loc[idx,labels[i_label+1:i_label+3]], df.loc[idx,labels[i_label+1:i_label+3]], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)

            else:
                for idx in df.index:
                    for i_label in range(len(labels)):
                        if i_label%2==0:
                            ax.plot(df_x_jitter.loc[idx,labels[i_label:i_label+2]], df.loc[idx,labels[i_label:i_label+2]], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)

        fig.savefig('log/results/FoldsChangeWithinSpecies_'+fscore+'.png')



