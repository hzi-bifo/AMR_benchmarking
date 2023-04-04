
import argparse
import pandas as pd
import json
from src.amr_utility import name_utility,load_data,file_utility
import matplotlib.pyplot as plt
import seaborn as sns
from src.benchmark_utility.lib.CombineResults import  combine_data_meanstd
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.legend_handler import HandlerPatch

'''April 2023. one box plot to replace Fig. 8.'''

def combinedata(species,level,df_anti,merge_name,fscore,learning,epochs,f_fixed_threshold,f_nn_base,f_optimize_score,temp_path,output_path):
    #1. SSSA

    f_kma=True
    f_phylotree=False

    save_name_score_final=name_utility.GETname_AAresult('AytanAktug',merge_name,0.0, 0,f_fixed_threshold,f_nn_base
                                                        ,'f1_macro',f_kma,f_phylotree,'MSMA_discrete',output_path)
    single_results=pd.read_csv(save_name_score_final+'_SSSAmapping_'+fscore+'.txt', index_col=0,sep="\t")

    #2.-------------------MSMA_concat_LOO

    merge_name_test = species.replace(" ", "_")
    concat_s_score=name_utility.GETname_AAresult('AytanAktug',merge_name,learning, epochs,f_fixed_threshold,\
                                     f_nn_base,f_optimize_score,f_kma,f_phylotree,'MSMA_concatLOO',output_path)

    concat_results=pd.read_csv(concat_s_score + '_'+merge_name_test+'_SummaryBenchmarking.txt', sep="\t", header=0, index_col=0)

    # 3 . -------------------MSMA_concat_mixedS
    save_name_score_final = name_utility.GETname_AAresult('AytanAktug',merge_name,learning, epochs,f_fixed_threshold,\
                                         f_nn_base,f_optimize_score,f_kma,f_phylotree,'MSMA_concat_mixedS',output_path)

    concatM_results=pd.read_csv(save_name_score_final+'_split_discrete_model_'+str(fscore)+'.txt', index_col=0,sep="\t")


    # 4. ---------MSMA_discrete
    save_name_score_final = name_utility.GETname_AAresult('AytanAktug',merge_name,learning, epochs,f_fixed_threshold,\
                                         f_nn_base,f_optimize_score,f_kma,f_phylotree,'MSMA_discrete',output_path)
    dis_results=pd.read_csv(save_name_score_final+'_split_discrete_model_'+str(fscore)+'.txt',  index_col=0,sep="\t")



    #-------------------------------
    #Prepare dataframe for plotting.
    #-------------------------------

    antibiotics = df_anti[species].split(';')

    summary_plot = pd.DataFrame(columns=[fscore, 'antibiotic', 'model','species'])

    for each_anti in antibiotics:

        # print(single_results)
        # summary_plot_single=single_results.loc[single_results['antibiotic']==each_anti]
        # summary_plot_single=summary_plot_single[[fscore,'antibiotic']]
        # summary_plot_single['model']='single-species-antibiotic model'
        # summary_plot = summary_plot.append(summary_plot_single, ignore_index=True)
        summary_plot_single=pd.DataFrame(columns=[fscore, 'antibiotic', 'model','species'])
        summary_plot_single.loc['e'] = [single_results.loc[species,each_anti ], each_anti, 'single-model',species]
        summary_plot = summary_plot.append(summary_plot_single, ignore_index=True)
        #-------discrete
        #
        summary_plot_dis = pd.DataFrame(columns=[fscore, 'antibiotic', 'model','species'])
        summary_plot_dis.loc['e'] = [dis_results.loc[species,each_anti ], each_anti, 'control multi-species model',species]
        summary_plot = summary_plot.append(summary_plot_dis, ignore_index=True)

        #------------------------------------------
        #concat M

        summary_plot_concatM = pd.DataFrame(columns=[fscore, 'antibiotic', 'model','species'])
        summary_plot_concatM.loc['e'] = [concatM_results.loc[species,each_anti ],each_anti, 'type 2 multi-species model evaluation 1',species]
        summary_plot = summary_plot.append(summary_plot_concatM, ignore_index=True)


        #-----------concat leave-one-out
        # summary_plot_sub.loc[species, each_anti] = data_score.loc[each_anti, each_score]
        summary_plot_multi = pd.DataFrame(columns=[fscore, 'antibiotic', 'model','species'])
        summary_plot_multi.loc['e'] = [concat_results.loc[each_anti,fscore], each_anti, 'type 2 multi-species model evaluation 2',species]
        summary_plot = summary_plot.append(summary_plot_multi, ignore_index=True)


    return summary_plot

def change_layout(data_plot,fscore,species):

    data=pd.DataFrame(columns=['single-model', 'control multi-species model','type 2 multi-species model evaluation 1','type 2 multi-species model evaluation 2'])

    data_plot=data_plot[(data_plot['species'] == species)] #newly added
    data1=data_plot[(data_plot['model'] == 'single-model')]
    data2=data_plot[(data_plot['model'] == 'control multi-species model')]
    data3=data_plot[(data_plot['model'] == 'type 2 multi-species model evaluation 1')]
    data4=data_plot[(data_plot['model'] == 'type 2 multi-species model evaluation 2')]

    data['single-model']=data1[fscore].tolist()
    data['control multi-species model']=data2[fscore].tolist()
    data['type 2 multi-species model evaluation 1']=data3[fscore].tolist()
    data['type 2 multi-species model evaluation 2']=data4[fscore].tolist()

    return data



def extract_info(fscore,level,f_all,learning,epochs,f_optimize_score,f_fixed_threshold,
                 f_nn_base,temp_path,output_path):

    # tool_list=['Single-species-antibiotic Aytan-Aktug','Discrete databases multi-species model',\
    #             'Concatenated databases mixed multi-species model', 'Concatenated databases leave-one-out multi-species model']
    # foldset=['Homology-aware folds']
    data = pd.read_csv('./data/PATRIC/meta/'+str(level)+'_multi-species_summary.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")

    merge_name = []
    if f_all:
        list_species = data.index.tolist()[:-1]
        data = data.loc[list_species, :]

    else:
        pass
    for n in list_species:
        merge_name.append(n[0] + n.split(' ')[1][0])
    merge_name = '_'.join(merge_name)  # e.g.Se_Kp_Pa
    # rearrange species order:
    list_species=['Mycobacterium tuberculosis','Campylobacter jejuni','Salmonella enterica','Escherichia coli','Streptococcus pneumoniae',\
                  'Klebsiella pneumoniae','Staphylococcus aureus','Acinetobacter baumannii','Pseudomonas aeruginosa']
    data=data.reindex(list_species)
    df_anti = data.dot(data.columns + ';').str.rstrip(';')

    print(df_anti)
    # fig = plt.figure(figsize=(9, 9))
    fig, axs = plt.subplots(1,1,figsize=(9, 9))
    # plt.tight_layout(pad=4)
    fig.subplots_adjust(top=0.98, bottom=0.35,left=0.2)

    color_selection=['#a6611a','#dfc27d','#80cdc1','#018571']
    palette = iter(color_selection)
    data_plot = pd.DataFrame(columns=[fscore, 'antibiotic', 'model','species'])
    for species in list_species:
        print(species)
        summary_plot=combinedata(species,level,df_anti,merge_name,fscore, learning,epochs,f_fixed_threshold,f_nn_base,f_optimize_score,temp_path,output_path)
        data_plot=data_plot.append(summary_plot, ignore_index=True)


    print(data_plot)

    # ax = sns.boxplot(x="day", y="total_bill", data=data_plot, color='white', width=.5, fliersize=0)
    ax=sns.boxplot(data=data_plot, x="model", y=fscore,palette=palette)


    ax.set_xticklabels(ax.get_xticklabels(),rotation=20,fontsize=20,ha='right')
    ax.set(xlabel=None)
    # ax.xticks(rotation=10)
    ax.tick_params(axis='y', which='major', labelsize=25)
    ax.set_ylabel(fscore.replace("_", "-").capitalize(),size = 25)
    # iterate over boxes
    for i,box in enumerate(ax.artists):
        box.set_edgecolor('black')
        box.set_facecolor('white')
        # iterate over whiskers and median lines
        for j in range(6*i,6*(i+1)):
             ax.lines[j].set_color('black')





    #--
    # dots  colored by species
    palette_tab10 = sns.color_palette("tab10",9)
    for species in list_species:
        print(species)

        df=change_layout(data_plot,fscore,species)
        print(df)

        jitter = 0.05
        df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=df.values.shape), columns=df.columns)
        df_x_jitter += np.arange(len(df.columns))
        df=df.set_index(df_x_jitter.index)



        for col_t in df:
            # print(list_species.index(species))
            ax.plot(df_x_jitter[col_t], df[col_t], 'o',c=palette_tab10[list_species.index(species)], alpha=0.8, zorder=1, ms=8, mew=1 )

    species_list_s=[(species[0] +". "+ species.split(' ')[1] ) for species in list_species]
    legend_dict=dict(zip(species_list_s, palette_tab10))
    patchList = []



    # for key in legend_dict:
    #     data_key = mpatches.Patch(color=legend_dict[key], label=key)
    #     patchList.append(data_key)
    # plt.legend(handles=patchList,bbox_to_anchor=(1.15, -0.25), ncol=2,fontsize=10,frameon=False,prop={'size': 15, 'style': 'italic'})
    c = [ mpatches.Circle((0.5, 0.5), radius = 0.25, facecolor=palette_tab10[i], edgecolor="none" ) for i in range(len(species_list_s))]
    plt.legend(c,species_list_s,bbox_to_anchor=(1.15, -0.25),  ncol=2,prop={'size': 15, 'style': 'italic'}, handler_map={mpatches.Circle: HandlerEllipse()})


    file_utility.make_dir(output_path+'Results/final_figures_tables')
    fig.savefig(output_path+'Results/final_figures_tables/F8_Compare_MSMA_box.pdf')
    fig.savefig(output_path+'Results/final_figures_tables/F8_Compare_MSMA_box.png')


class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=height + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--level', default='loose', type=str, required=False,
                        help='Quality control: strict or loose')
    parser.add_argument('-fscore', '--fscore', default='f1_macro', type=str, required=False,
                        help='the score used to choose the best classifier for each antibiotic.')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument('-f_fixed_threshold', '--f_fixed_threshold', dest='f_fixed_threshold', action='store_true',
                        help='set a fixed threshod:0.5.')
    parser.add_argument('-f_optimize_score', '--f_optimize_score', default='f1_macro', type=str,
                        help='the optimizing-score for choosing the best estimator in inner loop. Choose: auc or f1_macro.')
    parser.add_argument("-e", "--epochs", default=0, type=int,
                        help='epochs')
    parser.add_argument("-learning", "--learning", default=0.0, type=float,
                        help='learning rate')
    parser.add_argument('-f_nn_base', '--f_nn_base', dest='f_nn_base', action='store_true',
                        help='benchmarking baseline.')
    parser.add_argument('-o', '--output_path', default='./', type=str, required=False,
                        help='Directory to store CV scores.')

    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                        help='Directory to store temporary files.')
    parsedArgs = parser.parse_args()

    extract_info(parsedArgs.fscore,parsedArgs.level,parsedArgs.f_all,
                 parsedArgs.learning,parsedArgs.epochs,parsedArgs.f_optimize_score,parsedArgs.f_fixed_threshold,parsedArgs.f_nn_base,
                 parsedArgs.temp_path,parsedArgs.output_path)
