import sys
import os
sys.path.insert(0, os.path.abspath('../'))
from src.amr_utility import name_utility,load_data,file_utility
import numpy as np
from src.benchmark_utility.lib.CombineResults import combine_data_meanstd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.benchmark_utility.lib.pairbox import change_layoutByTool




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
    elif f_mean_std=='standard deviation':
        flag='_std'
    else:
        print('Wrong parameters set, please reset.')
        exit(1)
    np.random.seed(0)
    # --------------
    #1-2. How is robustness for ML baseline majority?
    # --------------
    if f_step=='1':

        fig, axs = plt.subplots(1,1,figsize=(10, 10))
        plt.tight_layout(pad=7)
        tool_list=['ML Baseline (Majority)']

        for tool in tool_list:

            tool_p=[tool]
            data_plot=combine_data_meanstd(df_species,level,fscore,tool_p,foldset,output_path,flag)
            data_plot= data_plot.astype({fscore:float})

            ax = sns.violinplot(x="folds", y=fscore,data=data_plot,
                        inner=None, color="0.95")
            if f_mean_std=='mean':
                ax.set(ylim=(0, 1.0))
            else:
                ax.set(ylim=(0, 0.5))
            ax.set_ylabel(fscore.replace("_", "-").capitalize(),size = 25)
            ax.tick_params(axis='y', which='major', labelsize=25)
            ax.set_title(tool+'\n '+f_mean_std, weight='bold',size=31)
            ax.set(xticklabels=[])
            ax.set(xlabel=None)
            ax.tick_params(axis='x',bottom=False)
            #--
            #connect dots representing the same tool+anti combination
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


        fig.savefig(output_path+'Results/supplement_figures_tables/S2_RobustAnalysis_MLbaseline_'+f_mean_std+'.pdf')
        fig.savefig(output_path+'Results/supplement_figures_tables/S2_RobustAnalysis_MLbaseline_'+f_mean_std+'.png')



