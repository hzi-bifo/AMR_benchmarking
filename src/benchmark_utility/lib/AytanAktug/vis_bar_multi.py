
import argparse
import pandas as pd
import json
from src.amr_utility import name_utility,file_utility
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

'''Bar plot of  SSSA,MSMA_discrete,MSMA_concat_mixedS , MSMA_concat_LOO.
April 2023. Kover multi LOSO added.
May 1,, 2023. PhenotypeSeeker multi-species LOSO added'''

def combinedata(species,df_anti,merge_name,fscore,learning,epochs,f_fixed_threshold,f_nn_base,f_optimize_score,output_path):
    #1. SSSA

    f_kma=True
    f_phylotree=False

    save_name_score_final=name_utility.GETname_AAresult('AytanAktug',merge_name,0.0, 0,f_fixed_threshold,f_nn_base
                                                        ,'f1_macro',f_kma,f_phylotree,'MSMA_discrete',output_path)
    single_results=pd.read_csv(save_name_score_final+'_SSSAmapping_'+fscore+'.txt', index_col=0,sep="\t")
    single_results_std=pd.read_csv(save_name_score_final+'_SSSAmapping_'+fscore+'_std.txt', index_col=0,sep="\t")
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


    # 5. ---------Kover single -model
    results_file1, _= name_utility.GETname_result('kover', merge_name_test,fscore,f_kma,f_phylotree,'scm',output_path)
    k_s1_results=pd.read_csv(results_file1 +'_PLOT.txt', header=0, index_col=0,sep="\t")
    k_s1_results_std=pd.read_csv(results_file1 +'_std.txt', header=0, index_col=0,sep="\t")
    results_file2, _= name_utility.GETname_result('kover', merge_name_test,fscore,f_kma,f_phylotree,'tree',output_path)
    k_s2_results=pd.read_csv(results_file2 +'_PLOT.txt', header=0, index_col=0,sep="\t")
    k_s2_results_std=pd.read_csv(results_file2 +'_std.txt', header=0, index_col=0,sep="\t")

    # 6. ---------Kover multi LOSO
    k_m1=name_utility.GETname_result2('kover',merge_name_test,fscore,'scm',output_path)
    k_m1_results=pd.read_csv(k_m1 + '.txt', sep="\t", header=0, index_col=0)
    k_m2=name_utility.GETname_result2('kover',merge_name_test,fscore,'tree',output_path)
    k_m2_results=pd.read_csv(k_m2 +'.txt', sep="\t", header=0, index_col=0)

    # 6. ---------phenotypeseeker single -model
    results_pts1, _= name_utility.GETname_result('phenotypeseeker', merge_name_test,fscore,f_kma,f_phylotree,'lr',output_path)
    pts_s1_results=pd.read_csv(results_pts1 +'_PLOT.txt', header=0, index_col=0,sep="\t")
    pts_s1_results_std=pd.read_csv(results_pts1 +'_std.txt', header=0, index_col=0,sep="\t")


    # 7. ---------phenotypeseeker multi LOSO
    pts_m1=name_utility.GETname_result2('phenotypeseeker',merge_name_test,fscore,'lr',output_path)
    pts_m1_results=pd.read_csv(pts_m1 + '.txt', sep="\t", header=0, index_col=0)




    #-------------------------------
    #Prepare dataframe for plotting.
    #-------------------------------

    antibiotics = df_anti[species].split(';')

    summary_plot = pd.DataFrame(columns=[fscore, 'antibiotic', 'model','std'])

    for each_anti in antibiotics:

        summary_plot_single=pd.DataFrame(columns=[fscore, 'antibiotic', 'model','std'])
        summary_plot_single.loc['e'] = [single_results.loc[species,each_anti ], each_anti, 'single-species model, homology nested CV',single_results_std.loc[species,each_anti]]
        summary_plot = summary_plot.append(summary_plot_single, ignore_index=True)
        #-------discrete
        #
        summary_plot_dis = pd.DataFrame(columns=[fscore, 'antibiotic', 'model','std'])
        summary_plot_dis.loc['e'] = [dis_results.loc[species,each_anti ], each_anti, 'control multi-species model, homology CV',np.nan]
        summary_plot = summary_plot.append(summary_plot_dis, ignore_index=True)

        #------------------------------------------
        #concat M

        summary_plot_concatM = pd.DataFrame(columns=[fscore, 'antibiotic', 'model','std'])
        summary_plot_concatM.loc['e'] = [concatM_results.loc[species,each_anti ],each_anti,'cross-species model, homology CV',np.nan]
        summary_plot = summary_plot.append(summary_plot_concatM, ignore_index=True)


        #-----------concat leave-one-out
        # summary_plot_sub.loc[species, each_anti] = data_score.loc[each_anti, each_score]
        summary_plot_multi = pd.DataFrame(columns=[fscore, 'antibiotic', 'model','std'])
        summary_plot_multi.loc['e'] = [concat_results.loc[each_anti,fscore], each_anti, 'cross-species model, LOSO',np.nan]
        summary_plot = summary_plot.append(summary_plot_multi, ignore_index=True)


        #-----------Kover single
        summary_plot_k_s1=pd.DataFrame(columns=[fscore, 'antibiotic', 'model','std'])
        summary_plot_k_s1.loc['e'] = [k_s1_results.loc[each_anti,'weighted-'+fscore ], each_anti, 'SCM single-species model, homology CV',k_s1_results_std.loc[each_anti,'weighted-'+fscore]]
        summary_plot = summary_plot.append(summary_plot_k_s1, ignore_index=True)

        summary_plot_k_s2=pd.DataFrame(columns=[fscore, 'antibiotic', 'model','std'])
        summary_plot_k_s2.loc['e'] = [k_s2_results.loc[each_anti,'weighted-'+fscore ], each_anti, 'CART single-species model, homology CV',k_s2_results_std.loc[each_anti,'weighted-'+fscore]]
        summary_plot = summary_plot.append(summary_plot_k_s2, ignore_index=True)

        #-----------Kover LOSO
        summary_plot_k_m1 = pd.DataFrame(columns=[fscore, 'antibiotic', 'model','std'])
        summary_plot_k_m1.loc['e'] = [k_m1_results.loc[each_anti,fscore], each_anti, 'SCM cross-species model, LOSO',np.nan]
        summary_plot = summary_plot.append(summary_plot_k_m1, ignore_index=True)

        summary_plot_k_m2 = pd.DataFrame(columns=[fscore, 'antibiotic', 'model','std'])
        summary_plot_k_m2.loc['e'] = [k_m2_results.loc[each_anti,fscore], each_anti, 'CART cross-species model, LOSO',np.nan]
        summary_plot = summary_plot.append(summary_plot_k_m2, ignore_index=True)


        #-----------phenotypeseeker single
        summary_plot_pts_s1=pd.DataFrame(columns=[fscore, 'antibiotic', 'model','std'])
        summary_plot_pts_s1.loc['e'] = [pts_s1_results.loc[each_anti,'weighted-'+fscore ], each_anti, 'LR single-species model, homology CV',pts_s1_results_std.loc[each_anti,'weighted-'+fscore]]
        summary_plot = summary_plot.append(summary_plot_pts_s1, ignore_index=True)


        #-----------phenotypeseeker LOSO
        summary_plot_pts_m1 = pd.DataFrame(columns=[fscore, 'antibiotic', 'model','std'])
        summary_plot_pts_m1.loc['e'] = [pts_m1_results.loc[each_anti,fscore], each_anti, 'LR cross-species model, LOSO',np.nan]
        summary_plot = summary_plot.append(summary_plot_pts_m1, ignore_index=True)




    return summary_plot


def extract_info(fscore,level,f_all,learning,epochs,f_optimize_score,f_fixed_threshold,
                 f_nn_base,output_path):
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


    fig, axs = plt.subplots(4, 3,figsize=(38,30), gridspec_kw={'width_ratios': [1.2,1, 2]})#
    plt.tight_layout()
    fig.subplots_adjust(left=0.04,  right=0.98,wspace=0.1, hspace=0.35, top=0.8, bottom=0.08)
    gs0 = axs[1, 0].get_gridspec()
    # remove the underlying axes
    for ax in axs[1, :2]:
        ax.remove()
    axbig0 = fig.add_subplot(gs0[1, :2])

    gs1 = axs[2, 0].get_gridspec()
    # remove the underlying axes
    for ax in axs[2, :2]:
        ax.remove()
    axbig1 = fig.add_subplot(gs1[2, :2])

    gs2 = axs[3, 0].get_gridspec()
    # remove the underlying axes
    for ax in axs[3, :2]:
        ax.remove()
    axbig2 = fig.add_subplot(gs2[3, :2])

    n = 0
    for species in list_species:

        summary_plot=combinedata(species,df_anti,merge_name,fscore, learning,epochs,f_fixed_threshold,f_nn_base,f_optimize_score,output_path)


        with open('./data/AntiAcronym_dict.json') as f:
            map_acr = json.load(f)

        summary_plot['antibiotic_acr']=summary_plot['antibiotic'].apply(lambda x: map_acr[x])
        # color_selection=['#a6611a','#dfc27d','#018571','#80cdc1']
        color_selection=['#543005','#8c510a','#bf812d','#dfc27d','#00441b','#006d2c','#238b45','#66c2a4', '#0571b0', '#92c5de'] # for future PhenotypeSeeker #0571b0 #92c5de





        palette = iter(color_selection)
        row = (n //3)
        col = n % 3
        species_title=(species[0] +". "+ species.split(' ')[1] )

        ### make the model come in an order
        custom_dict = {'single-species model, homology nested CV':0, \
                               'control multi-species model, homology CV':1,\
                               'cross-species model, homology CV':2,\
                               'cross-species model, LOSO':3,\
                                'SCM single-species model, homology CV':4,\
                               'CART single-species model, homology CV':5,\
                               'SCM cross-species model, LOSO':6,\
                               'CART cross-species model, LOSO':7,\
                        'LR single-species model, homology CV':8,\
                       'LR multi-species model, LOSO':9 }
        summary_plot['rank'] = summary_plot['model'].map(custom_dict)
        summary_plot = summary_plot.sort_values(['rank','antibiotic'],ascending=[True, True])

        print(species)

        if species in ['Mycobacterium tuberculosis']:

            ax_ = plt.subplot(431)
            g = sns.barplot(x="antibiotic_acr", y=fscore, hue='model',
                        data=summary_plot, dodge=True, ax=ax_,palette=palette)

            x_coords = [p.get_x() + 0.5 * p.get_width() for p in g.patches]
            y_coords = [p.get_height() for p in g.patches]
            ax_.errorbar(x=x_coords, y=y_coords, yerr=summary_plot["std"].to_list(), elinewidth=4,capsize=6,capthick=4,fmt="none",ecolor='black', c="k")

            n+=1
            g.set_ylabel(fscore.replace("_", "-").capitalize(), fontsize=25)
            g.set_title(species_title,fontsize=31,style='italic', weight='bold',pad=10)
            g.tick_params(axis='y', which='major', labelsize=25)
        elif species in ['Campylobacter jejuni']:

            ax_ = plt.subplot(432)
            g = sns.barplot(x="antibiotic_acr", y=fscore, hue='model',
                        data=summary_plot, dodge=True, ax=ax_,palette=palette)
            x_coords = [p.get_x() + 0.5 * p.get_width() for p in g.patches]
            y_coords = [p.get_height() for p in g.patches]
            ax_.errorbar(x=x_coords, y=y_coords, yerr=summary_plot["std"],elinewidth=4,capsize=6,capthick=4,fmt="none",ecolor='black', c="k")

            g.set_yticklabels([])
            g.set_yticks([])
            g.set_title(species_title,fontsize=31,style='italic', weight='bold',pad=10 )
            g.set(ylabel=None)
            n+=1
        elif species in ['Salmonella enterica']:
            n+=1
            num=433
            ax_= plt.subplot(num)
            g = sns.barplot(x="antibiotic_acr", y=fscore, hue='model',
                        data=summary_plot, dodge=True, ax=ax_,palette=palette)#ax=axs[row, col]
            x_coords = [p.get_x() + 0.5 * p.get_width() for p in g.patches]
            y_coords = [p.get_height() for p in g.patches]
            ax_.errorbar(x=x_coords, y=y_coords, yerr=summary_plot["std"], elinewidth=4,capsize=6,capthick=4,fmt="none",ecolor='black', c="k")

            g.set_yticklabels([])
            g.set_yticks([])
            g.set_title(species_title,fontsize=31,style='italic', weight='bold',pad=10 )
            g.set(ylabel=None)

        elif species in ['Escherichia coli']:
            g = sns.barplot(x="antibiotic_acr", y=fscore, hue='model',
                                    data=summary_plot, dodge=True, ax=axbig0,palette=palette)#ax=axs[row, col]

            x_coords = [p.get_x() + 0.5 * p.get_width() for p in g.patches]
            y_coords = [p.get_height() for p in g.patches]
            axbig0.errorbar(x=x_coords, y=y_coords, yerr=summary_plot["std"], elinewidth=4,capsize=3,capthick=4,fmt="none",ecolor='black', c="k")

            n+=2
            g.set_ylabel(fscore.replace("_", "-").capitalize(), fontsize=25)
            g.set_title(species_title,fontsize=31,style='italic', weight='bold' ,pad=10)
            g.tick_params(axis='y', which='major', labelsize=25)


        elif species in ['Streptococcus pneumoniae']:
            num=436
            ax_= plt.subplot(num)
            g = sns.barplot(x="antibiotic_acr", y=fscore, hue='model',
                        data=summary_plot, dodge=True, ax=ax_,palette=palette)#ax=axs[row, col]
            x_coords = [p.get_x() + 0.5 * p.get_width() for p in g.patches]
            y_coords = [p.get_height() for p in g.patches]
            ax_.errorbar(x=x_coords, y=y_coords, yerr=summary_plot["std"], elinewidth=4,capsize=6,capthick=4,fmt="none",ecolor='black', c="k")

            g.set_yticklabels([])
            g.set_yticks([])
            g.set_title(species_title,fontsize=31,style='italic', weight='bold',pad=10 )
            g.set(ylabel=None)
            n+=1

        elif species =='Klebsiella pneumoniae':
            g = sns.barplot(x="antibiotic_acr", y=fscore, hue='model',
                        data=summary_plot, dodge=True, ax=axbig1,palette=palette)#ax=axs[row, col]

            x_coords = [p.get_x() + 0.5 * p.get_width() for p in g.patches]
            y_coords = [p.get_height() for p in g.patches]
            axbig1.errorbar(x=x_coords, y=y_coords, yerr=summary_plot["std"], elinewidth=4,capsize=3,capthick=4,fmt="none",ecolor='black', c="k")

            n+=2
            g.set_ylabel(fscore.replace("_", "-").capitalize(), fontsize=25)
            g.set_title(species_title,fontsize=31,style='italic', weight='bold' ,pad=10)
            g.tick_params(axis='y', which='major', labelsize=25)

        elif species =='Acinetobacter baumannii':
            g = sns.barplot(x="antibiotic_acr", y=fscore, hue='model',
                        data=summary_plot, dodge=True, ax=axbig2,palette=palette)#ax=axs[row, col]

            x_coords = [p.get_x() + 0.5 * p.get_width() for p in g.patches]
            y_coords = [p.get_height() for p in g.patches]
            axbig2.errorbar(x=x_coords, y=y_coords, yerr=summary_plot["std"], elinewidth=4,capsize=3,capthick=4,fmt="none",ecolor='black', c="k")

            n+=2
            g.set_ylabel(fscore.replace("_", "-").capitalize(), fontsize=25)
            g.set_title(species_title,fontsize=31,style='italic', weight='bold' ,pad=10)
            g.tick_params(axis='y', which='major', labelsize=25)

        else:

            g = sns.barplot(x="antibiotic_acr", y=fscore, hue='model',
                    data=summary_plot, dodge=True, ax=axs[row, col],palette=palette)#ax=axs[row, col]
            x_coords = [p.get_x() + 0.5 * p.get_width() for p in g.patches]
            y_coords = [p.get_height() for p in g.patches]
            axs[row, col].errorbar(x=x_coords, y=y_coords, yerr=summary_plot["std"],  elinewidth=4,capsize=6,capthick=4,fmt="none",ecolor='black', c="k")
            n+=1
            # g.set_yticklabels([])
            # g.set_yticks([])
            g.set_title(species_title,fontsize=29,style='italic', weight='bold',pad=10 )
            g.set(ylabel=None)


        g.set(ylim=(0, 1.05))

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
            # g.legend(bbox_to_anchor=(4,2), fontsize=35, ncol=3,frameon=False)
            (lines, labels) = g.get_legend_handles_labels()
            leg1 = plt.legend(lines[:8], labels[:8], bbox_to_anchor=(2.5,2), frameon=False,fontsize=35,  ncol=2)
            leg2 = plt.legend(lines[8:], labels[8:], bbox_to_anchor=(3.8,2), frameon=False, fontsize=35, ncol=1,)
            g.add_artist(leg1)
            g.add_artist(leg2)

            g.text(-0.3, 2.2, 'AytanAktug', fontsize=38,weight='bold')
            g.text(2.3, 2.2, 'Kover', fontsize=38,weight='bold')
            g.text(5, 2.2, 'PhenotypeSeeker', fontsize=38,weight='bold')



    file_utility.make_dir(output_path+'Results/final_figures_tables')
    fig.savefig(output_path+'Results/final_figures_tables/F8_Compare_multi_bar.pdf')
    fig.savefig(output_path+'Results/final_figures_tables/F8_Compare_multi_bar.png')





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

    parsedArgs = parser.parse_args()

    extract_info(parsedArgs.fscore,parsedArgs.level,parsedArgs.f_all,parsedArgs.learning,parsedArgs.epochs,
                 parsedArgs.f_optimize_score,parsedArgs.f_fixed_threshold,parsedArgs.f_nn_base,parsedArgs.output_path)


