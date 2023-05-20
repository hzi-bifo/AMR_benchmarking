import sys
import os
sys.path.insert(0, os.path.abspath('../'))
from src.amr_utility import name_utility
import numpy as np
from src.benchmark_utility.lib.CombineResults import  combine_data_meanstd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def change_layout(data_plot,fscore,species):
    if species !='Mycobacterium tuberculosis' :
        data=pd.DataFrame(columns=['Random folds', 'Phylogeny-aware folds','Homology-aware folds'])
        data1=data_plot[(data_plot['folds'] == 'Random folds')]
        data2=data_plot[(data_plot['folds'] == 'Phylogeny-aware folds')]
        data3=data_plot[(data_plot['folds'] == 'Homology-aware folds')]
        data['Random folds']=data1[fscore]
        data['Phylogeny-aware folds']=data2[fscore]
        data['Homology-aware folds']=data3[fscore]
    else:
        data=pd.DataFrame(columns=['Random folds', 'Homology-aware folds'])
        data1=data_plot[(data_plot['folds'] == 'Random folds')]
        ### data2=data_plot[(data_plot['folds'] == 'Phylogeny-aware folds')]
        data3=data_plot[(data_plot['folds'] == 'Homology-aware folds')]
        data['Random folds']=data1[fscore]
        ### data['Phylogeny-aware folds']=data2[fscore]
        data['Homology-aware folds']=data3[fscore]
    return data
def change_layoutByTool(data_plot,fscore):
    '''connect dots representing the same species+anti combination'''
    data=pd.DataFrame(columns=['species','anti','Random folds', 'Phylogeny-aware folds','Homology-aware folds'])
    df = data_plot.reset_index()

    df=df.groupby(['species', 'antibiotics']).size().reset_index().rename(columns={0:'count'})

    data['species']=df['species']
    data['anti']=df['antibiotics']
    for index, row in data.iterrows():
        s=row['species']
        a=row['anti']

        if s!='Mycobacterium tuberculosis':
                row['Phylogeny-aware folds']=data_plot[(data_plot['species'] ==s)&(data_plot['antibiotics'] ==a)&(data_plot['folds'] =='Phylogeny-aware folds')].iloc[0][fscore]
        row['Random folds']=data_plot[(data_plot['species'] ==s)&(data_plot['antibiotics'] ==a)&(data_plot['folds'] =='Random folds')].iloc[0][fscore]
        row['Homology-aware folds']=data_plot[(data_plot['species'] ==s)&(data_plot['antibiotics'] ==a)&(data_plot['folds'] =='Homology-aware folds')].iloc[0][fscore]

    data_mt=data[(data['species'] == 'Mycobacterium tuberculosis')]
    data_else=data[(data['species'] != 'Mycobacterium tuberculosis')]
    return data,data_else,data_mt
def ranking(data_plot):
    df = data_plot.reset_index()
    df['anti+folds']=df['antibiotics']+'\n'+ df['folds']
    df=df.sort_values(['antibiotics','folds'], ascending = (True,False ))
    return df
def change_layoutWithinSpecies(data_plot,fscore):
    col=data_plot.groupby(by=["anti+folds"],sort=False).count().index.to_list()
    data=pd.DataFrame(columns=col)
    for c in col:
        data_sub=data_plot[(data_plot["anti+folds"] == c)]
        data[c]=data_sub[fscore].to_list()
    return data





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
    antibiotics = data['modelling antibiotics'].tolist()
    foldset=['Random folds', 'Phylogeny-aware folds','Homology-aware folds']
    if f_mean_std=='mean':
        flag='_PLOT'
    elif f_mean_std=='std':
        flag='_std'
    else:
        print('Wrong parameters set, please reset.')
        exit(1)
    np.random.seed(0)
    # --------------
    #1. Is it different for species?
    # --------------
    if f_step=='1':
        tool_list=[ 'Aytan-Aktug', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover']

        fig, axs = plt.subplots(4,4,figsize=(25, 25))
        plt.tight_layout(pad=4)
        fig.subplots_adjust(wspace=0.1, hspace=0.25, top=0.93, bottom=0.02,left=0.05)
        fig.suptitle('A', weight='bold',size=35,x=0.02,y=0.97 )

        i=0
        for species, antibiotics_selected in zip(df_species, antibiotics):
            print(species)

            row = (i //4)
            col = i % 4
            i+=1
            species_p=[species]
            data_plot=combine_data_meanstd(species_p,level,fscore,tool_list,foldset,output_path,flag)
            #[fscore, 'species', 'software','anti','folds']
            # print(data_plot)
            data_plot= data_plot.astype({fscore:float})
            # ax = sns.violinplot(x="folds", y=fscore, ax=axs[row, col],data=data_plot,
            #             inner='box', color="0.95")
            ax=sns.boxplot(data=data_plot, x="folds", ax=axs[row, col],y=fscore)
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

            species_title= (species[0] +". "+ species.split(' ')[1] )
            ax.set_title(species_title,style='italic', weight='bold',size=31,pad=10)
            ax.set(xticklabels=[])
            ax.set(xlabel=None)
            ax.tick_params(axis='x',bottom=False)
            #--
            #connect dots representing the same tool+anti combination
            df=change_layout(data_plot,fscore,species)


            jitter = 0.05
            df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=df.values.shape), columns=df.columns)
            df_x_jitter += np.arange(len(df.columns))
            df=df.set_index(df_x_jitter.index)

            i_color=0
            for col_t in df:
                if species !='Mycobacterium tuberculosis':
                    if i_color%3==0:
                        ax.plot(df_x_jitter[col_t], df[col_t], 'o',c='tab:blue', alpha=0.6, zorder=1, ms=8, mew=1, label="Random folds")
                    elif i_color%3==1:
                        ax.plot(df_x_jitter[col_t], df[col_t], 'o',c='orange', alpha=0.6, zorder=1, ms=8, mew=1, label="Phylogeny-aware folds")
                    else:
                        ax.plot(df_x_jitter[col_t], df[col_t], 'go', alpha=0.6, zorder=1, ms=8, mew=1, label="Homology-aware folds")

                else:
                    if i_color%2==0:
                        ax.plot(df_x_jitter[col_t], df[col_t], 'o',c='tab:blue', alpha=0.6, zorder=1, ms=8, mew=1)
                    elif i_color%2==1:
                        ax.plot(df_x_jitter[col_t], df[col_t], 'go', alpha=0.6, zorder=1, ms=8, mew=1)

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
                if species=='Escherichia coli' and ( i_color in [1,2,3]):
                    if f_mean_std=='mean':
                        leg=ax.legend(bbox_to_anchor=(0.8, 1.11),ncol=3,fontsize=30,frameon=False, markerscale=2)
                    else:
                        leg=ax.legend(bbox_to_anchor=(4, 1.38),ncol=3,fontsize=30,frameon=False, markerscale=2)
                    for lh in leg.legendHandles:
                        lh._legmarker.set_alpha(1)


            ax.set(xticklabels=[])
            ax.set(xlabel=None)
            ax.tick_params(axis='x',bottom=False)

            if species !='Mycobacterium tuberculosis':
                for idx in df.index:
                    ax.plot(df_x_jitter.loc[idx,['Random folds','Phylogeny-aware folds']], df.loc[idx,['Random folds','Phylogeny-aware folds']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
                    ax.plot(df_x_jitter.loc[idx,['Phylogeny-aware folds','Homology-aware folds']], df.loc[idx,['Phylogeny-aware folds','Homology-aware folds']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
            else:
                for idx in df.index:
                    ax.plot(df_x_jitter.loc[idx,['Random folds','Homology-aware folds']], df.loc[idx,['Random folds','Homology-aware folds']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)

            ax.set(xlabel=None)
            ax.tick_params(axis='x',bottom=False)



    # #2. Is it different for tools?
        ### tool_list=[ 'Aytan-Aktug', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover','ML Baseline (Majority)']
        summary_table_mean= pd.DataFrame(columns=['software','Random folds', 'Phylogeny-aware folds','Homology-aware folds']) ##not for plotting.
        summary_table_std= pd.DataFrame(columns=['software','Random folds', 'Phylogeny-aware folds','Homology-aware folds']) ##not for plotting.
        tool_list=[ 'Aytan-Aktug', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover','Point-/ResFinder']
        for tool in tool_list:
            print(tool)
            row = (i //4)
            col = i % 4
            i+=1
            tool_p=[tool]
            data_plot=combine_data_meanstd(df_species,level,fscore,tool_p,foldset,output_path,flag)
            ### [fscore, 'species', 'software','anti','folds']
            data_plot= data_plot.astype({fscore:float})
            # print(data_plot)
            average_table=data_plot.groupby(['folds']).mean()
            average_table=average_table.sort_values('folds',ascending=False)
            summary_table_mean_each=[tool]+average_table[fscore].tolist()
            summary_table_mean.loc[len(summary_table_mean)]=summary_table_mean_each
            print(average_table)
            average_table=data_plot.groupby(['folds']).std()
            average_table=average_table.sort_values('folds',ascending=False)
            print(average_table)
            summary_table_std_each=[tool]+average_table[fscore].tolist()
            summary_table_std.loc[len(summary_table_std)]=summary_table_std_each
            # summary_table_std=summary_table_std.append(summary_table_std_each, ignore_index=True)

            # ax = sns.violinplot(x="folds", y=fscore, ax=axs[row, col],data=data_plot,
            #             inner='box', color="0.95")
            #

            ax=sns.boxplot(data=data_plot, ax=axs[row, col],x="folds", y=fscore)
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
                ax.tick_params(axis='y', which='major', labelsize=25)
            else:
                ax.set_yticks([])
                ax.set(ylabel=None)
                ax.tick_params(axis='y',bottom=False)

            ax.set_title(tool, weight='bold',size=31)
            ax.set(xticklabels=[])
            ax.set(xlabel=None)
            ax.tick_params(axis='x',bottom=False)
            if i==12:
                ax.text(-0.7, 1.08, 'B', fontsize=35,weight='bold')

            #--
            #connect dots representing the same species+anti combination
            df_whole,df_else,df_mt=change_layoutByTool(data_plot,fscore)

            df=df_whole[['Random folds', 'Phylogeny-aware folds','Homology-aware folds']]

            jitter = 0.05

            df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=df.values.shape), columns=df.columns)
            df_x_jitter += np.arange(len(df.columns))
            df=df.set_index(df_x_jitter.index)


            for col in df:
                ax.plot(df_x_jitter[col], df[col], 'o', alpha=.40, zorder=1, ms=8, mew=1)


            ax.set(xticklabels=[])
            ax.set(xlabel=None)
            ax.tick_params(axis='x',bottom=False)
            for idx in df_else.index:
                ax.plot(df_x_jitter.loc[idx,['Random folds','Phylogeny-aware folds']], df.loc[idx,['Random folds','Phylogeny-aware folds']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
                ax.plot(df_x_jitter.loc[idx,['Phylogeny-aware folds','Homology-aware folds']], df.loc[idx,['Phylogeny-aware folds','Homology-aware folds']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)

            for idx in df_mt.index:
                ax.plot(df_x_jitter.loc[idx,['Random folds','Homology-aware folds']], df.loc[idx,['Random folds','Homology-aware folds']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)

            ax.set(xticklabels=[])
            ax.set(xlabel=None)
            ax.tick_params(axis='x',bottom=False)


        #set background color for B part
        fig.patches.extend([plt.Rectangle((0, 0),1,0.245,
                                  fill=True, color='grey', alpha=0.5, zorder=-1,
                                  transform=fig.transFigure, figure=fig)])
        fig.patches.extend([plt.Rectangle((0.743,0.245),0.26,0.24,
                                  fill=True, color='grey', alpha=0.5, zorder=-1,
                                  transform=fig.transFigure, figure=fig)])



        # -----------------------------
        if f_mean_std=='mean':
            fig.savefig(output_path+'Results/final_figures_tables/F6_RobustAnalysis_mean.png')
            fig.savefig(output_path+'Results/final_figures_tables/F6_RobustAnalysis_mean.pdf')
            summary_table_mean.to_csv(output_path+'Results/final_figures_tables/F6_tool_mean.csv', sep="\t")
            summary_table_std.to_csv(output_path+'Results/final_figures_tables/F6_tool_std.csv', sep="\t")
        elif f_mean_std=='std':
            fig.savefig(output_path+'Results/supplement_figures_tables/S2_RobustAnalysis_std.png')
            fig.savefig(output_path+'Results/supplement_figures_tables/S2_RobustAnalysis_std.pdf')
        else:
            print('Wrong parameters set, please check.')
            exit(1)

    #2. Is it different for antis within species?
    if f_step=='3':
        tool_list=[ 'Aytan-Aktug', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover']
        fig = plt.figure(figsize=(20, 30))
        plt.tight_layout()
        fig.subplots_adjust(left=0.04,  right=0.98,wspace=0.1, hspace=0.6, top=0.96, bottom=0.03)


        i=0
        j=0
        for species, antibiotics_selected in zip(df_species, antibiotics):

            species_p=[species]
            data_plot=combine_data_meanstd(species_p,level,fscore,tool_list,foldset,output_path,flag)
            ### [fscore, 'species', 'software','anti','folds']
            data_plot= data_plot.astype({fscore:float})
            data_plot=ranking(data_plot)

            #--
            #connect dots representing the same tool+anti combination
            df=change_layoutWithinSpecies(data_plot,fscore)
            labels = df.columns.to_list()
            jitter = 0.05
            df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=df.values.shape), columns=df.columns)
            df_x_jitter += np.arange(len(df.columns))
            df=df.set_index(df_x_jitter.index)

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
                        ax.plot(df_x_jitter[col], df[col], 'o',c='tab:blue', alpha=.80, zorder=1, ms=8, mew=1, label="Random folds")
                    elif i_color%3==1:
                        ax.plot(df_x_jitter[col], df[col], 'o',c='orange', alpha=.80, zorder=1, ms=8, mew=1, label="Phylogeny-aware folds")
                    else:
                        ax.plot(df_x_jitter[col], df[col], 'go', alpha=.80, zorder=1, ms=8, mew=1, label="Homology-aware folds")

                else:
                    if i_color%2==0:
                        ax.plot(df_x_jitter[col], df[col], 'o',c='tab:blue', alpha=.80, zorder=1, ms=8, mew=1)
                    elif i_color%2==1:
                        ax.plot(df_x_jitter[col], df[col], 'go', alpha=.80, zorder=1, ms=8, mew=1)

                i_color+=1
                species_title= (species[0] +". "+ species.split(' ')[1] )
                ax.set_title(species_title,style='italic', weight='bold',size=25)
                if f_mean_std=='mean':
                    ax.set(ylim=(0, 1.0))
                else:
                    ax.set(ylim=(0, 0.5))
                ax.set_ylabel(fscore.replace("_", "-").capitalize(),size=22)
                if species=='Escherichia coli' and ( i_color in [1,2,3]):
                    ax.legend(bbox_to_anchor=(1, 1.65),ncol=3,fontsize=26,frameon=False,markerscale=2,labels=['Random folds', 'Phylogeny-aware folds','Homology-aware folds'])


            ax.set_xticks(range(len(df.columns)))
            ax.set_xticklabels(df.columns)
            ax.set_xlim(-0.5,len(df.columns)-0.5)
            labels_p = [item.get_text() for item in ax.get_xticklabels()]
            for i_anti in labels_p:
                if '/' in i_anti:
                    posi=i_anti.find('/')
                    _i_anti=i_anti[:(posi+1)] + '\n' + i_anti[(posi+1):]
                    labels_p=[_i_anti if x==i_anti else x for x in labels_p]
            temp=0
            for i_label_p in labels_p:
                if temp%3==1:
                    labels_p[temp] = i_label_p.rsplit('\n',1)[0]

                else:
                    labels_p[temp]=''
                temp+=1
            ax.set_xticklabels(labels_p,size=20,rotation=10)

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


        fig.savefig(output_path+'Results/supplement_figures_tables/S2_RobustAnalysis_C'+f_mean_std+'.png')
        fig.savefig(output_path+'Results/supplement_figures_tables/S2_RobustAnalysis_C'+f_mean_std+'.pdf')




