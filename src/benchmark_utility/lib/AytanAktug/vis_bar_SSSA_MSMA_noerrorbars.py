
import argparse
import pandas as pd
import pickle,json
from src.amr_utility import name_utility,load_data,file_utility
from src.analysis_utility.lib import extract_score,make_table,math_utility
import matplotlib.pyplot as plt
import seaborn as sns


'''Bar plot of  SSSA,MSMA_discrete,MSMA_concat_mixedS , MSMA_concat_LOO'''

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

    summary_plot = pd.DataFrame(columns=[fscore, 'antibiotic', 'model'])

    for each_anti in antibiotics:

        # print(single_results)
        # summary_plot_single=single_results.loc[single_results['antibiotic']==each_anti]
        # summary_plot_single=summary_plot_single[[fscore,'antibiotic']]
        # summary_plot_single['model']='single-species-antibiotic model'
        # summary_plot = summary_plot.append(summary_plot_single, ignore_index=True)
        summary_plot_single=pd.DataFrame(columns=[fscore, 'antibiotic', 'model'])
        summary_plot_single.loc['e'] = [single_results.loc[species,each_anti ], each_anti, 'single-species-antibiotic model']
        summary_plot = summary_plot.append(summary_plot_single, ignore_index=True)
        #-------discrete
        #
        summary_plot_dis = pd.DataFrame(columns=[fscore, 'antibiotic', 'model'])
        summary_plot_dis.loc['e'] = [dis_results.loc[species,each_anti ], each_anti, 'Discrete databases multi-species model']
        summary_plot = summary_plot.append(summary_plot_dis, ignore_index=True)

        #------------------------------------------
        #concat M

        summary_plot_concatM = pd.DataFrame(columns=[fscore, 'antibiotic', 'model'])
        summary_plot_concatM.loc['e'] = [concatM_results.loc[species,each_anti ],each_anti, 'Concatenated databases mixed multi-species model']
        summary_plot = summary_plot.append(summary_plot_concatM, ignore_index=True)


        #-----------concat leave-one-out
        # summary_plot_sub.loc[species, each_anti] = data_score.loc[each_anti, each_score]
        summary_plot_multi = pd.DataFrame(columns=[fscore, 'antibiotic', 'model'])
        summary_plot_multi.loc['e'] = [concat_results.loc[each_anti,fscore], each_anti, 'Concatenated databases leave-one-out multi-species model']
        summary_plot = summary_plot.append(summary_plot_multi, ignore_index=True)
    return summary_plot


def extract_info(fscore,level,f_all,learning,epochs,f_optimize_score,f_fixed_threshold,
                 f_nn_base,temp_path,output_path):
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


    fig, axs = plt.subplots(2, 5,figsize=(30,20), gridspec_kw={'width_ratios': [1.2,1, 2,2,1.5]})#
    plt.tight_layout()
    fig.subplots_adjust(left=0.04,  right=0.98,wspace=0.1, hspace=0.3, top=0.8, bottom=0.08)
    gs = axs[1, 0].get_gridspec()
    # remove the underlying axes
    for ax in axs[1, :2]:
        ax.remove()
    axbig = fig.add_subplot(gs[1, :2])



    n = 0
    for species in list_species:

        summary_plot=combinedata(species,level,df_anti,merge_name,fscore, learning,epochs,f_fixed_threshold,f_nn_base,f_optimize_score,temp_path,output_path)


        with open('./data/AntiAcronym_dict.json') as f:
            map_acr = json.load(f)

        summary_plot['antibiotic_acr']=summary_plot['antibiotic'].apply(lambda x: map_acr[x])
        color_selection=['#a6611a','#dfc27d','#80cdc1','#018571']
        palette = iter(color_selection)
        row = (n //5)
        col = n % 5+1
        species_title=(species[0] +". "+ species.split(' ')[1] )
        if species in ['Mycobacterium tuberculosis']:

            ax_ = plt.subplot(251)
            g = sns.barplot(x="antibiotic_acr", y=fscore, hue='model',
                        data=summary_plot, dodge=True, ax=ax_,palette=palette)

            

            n+=1
            g.set_ylabel(fscore, fontsize=25)

            g.set_title(species_title,fontsize=31,style='italic', weight='bold',pad=10)
            g.tick_params(axis='y', which='major', labelsize=25)
        elif species in ['Campylobacter jejuni']:

            ax_ = plt.subplot(252)
            g = sns.barplot(x="antibiotic_acr", y=fscore, hue='model',
                        data=summary_plot, dodge=True, ax=ax_,palette=palette)

            g.set_yticklabels([])
            g.set_yticks([])
            g.set_title(species_title,fontsize=31,style='italic', weight='bold',pad=10 )
            g.set(ylabel=None)
            n+=1
        elif species in ['Salmonella enterica','Streptococcus pneumoniae','Escherichia coli']:
            n+=1
            num=250+n
            ax_= plt.subplot(num)
            g = sns.barplot(x="antibiotic_acr", y=fscore, hue='model',
                        data=summary_plot, dodge=True, ax=ax_,palette=palette)#ax=axs[row, col]
            g.set_yticklabels([])
            g.set_yticks([])
            g.set_title(species_title,fontsize=31,style='italic', weight='bold',pad=10 )
            g.set(ylabel=None)
        elif species =='Klebsiella pneumoniae':
            g = sns.barplot(x="antibiotic_acr", y=fscore, hue='model',
                        data=summary_plot, dodge=True, ax=axbig,palette=palette)#ax=axs[row, col]
            n+=1
            g.set_ylabel(fscore, fontsize=25)
            g.set_title(species_title,fontsize=31,style='italic', weight='bold' ,pad=10)
            g.tick_params(axis='y', which='major', labelsize=25)
        else:
            g = sns.barplot(x="antibiotic_acr", y=fscore, hue='model',
                    data=summary_plot, dodge=True, ax=axs[row, col],palette=palette)#ax=axs[row, col]
            n+=1
            g.set_yticklabels([])
            g.set_yticks([])
            g.set_title(species_title,fontsize=29,style='italic', weight='bold',pad=10 )
            g.set(ylabel=None)
        g.set(ylim=(0, 1.0))

        labels_p = [item.get_text() for item in g.get_xticklabels()]
        for i_anti in labels_p:
            if '/' in i_anti:
                posi=i_anti.find('/')
                _i_anti=i_anti[:(posi+1)] + '\n' + i_anti[(posi+1):]
                labels_p=[_i_anti if x==i_anti else x for x in labels_p]

        g.set_xticklabels(labels_p, size=32, rotation=40, horizontalalignment='right')
        g.set_xlabel('')
        if n!=1:
            g.get_legend().remove()

        else:
            handles, labels = g.get_legend_handles_labels()
            g.legend(bbox_to_anchor=(5,1.7), fontsize=35, ncol=1,frameon=False)


    file_utility.make_dir(output_path+'Results/final_figures_tables')
    fig.savefig(output_path+'Results/final_figures_tables/F8_Compare_MSMA_bar.pdf')
    fig.savefig(output_path+'Results/final_figures_tables/F8_Compare_MSMA_bar.png')





if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--level', default='loose', type=str, required=False,
                        help='Quality control: strict or loose')
    parser.add_argument('-fscore', '--fscore', default='f1_macro', type=str, required=False,
                        help='the score used to choose the best classifier for each antibiotic.')
    # parser.add_argument('-t_p', '--threshold_point', default=0.8, type=float,
    #                     help='Threshold for identity of Pointfinder. ')
    # parser.add_argument('-l_p', '--min_cov_point', default=0.6, type=float,
    #                     help=' Minimum (breadth-of) coverage of Pointfinder. ')
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
    # parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
    #                     help=' phylo-tree based cv folders.')
    # parser.add_argument("-cv", "--cv_number", default=10, type=int,
    #                     help='CV splits number')
    parser.add_argument('-o', '--output_path', default='./', type=str, required=False,
                        help='Directory to store CV scores.')

    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                        help='Directory to store temporary files.')
    parsedArgs = parser.parse_args()
    # parser.print_help()
    # print(parsedArgs)
    extract_info(parsedArgs.fscore,parsedArgs.level,parsedArgs.f_all,
                 parsedArgs.learning,parsedArgs.epochs,parsedArgs.f_optimize_score,parsedArgs.f_fixed_threshold,parsedArgs.f_nn_base,
                 parsedArgs.temp_path,parsedArgs.output_path)


