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


def prepare_data(df,fscore):
    # This function changes data to the dafaframe that can be used directly by seaborn for plotting.
    df_plot = pd.DataFrame(columns=[fscore, 'antibiotics', 'software'])
    anti_list=list(df.columns)
    print(df)

    columns_name=df_plot.columns.tolist()
    df_plot_sub = pd.DataFrame(columns=columns_name)
    for anti in anti_list:
        for tool in list(df.index):
            df_plot_sub.loc['s'] = [df.loc[tool,anti], anti,tool]
            df_plot = df_plot.append(df_plot_sub, sort=False)
    return df_plot



def combine_data(species,antibiotics,fscore, f_phylotree, f_kma,tool_list):
    # This function makes a matrix of all tools' results.
    # 'Point-/ResFinder', 'Neural networks', 'KmerC', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover'
    # todo need to change names before final release.
    data = []
    for tool in tool_list:
        if tool=='Point-/ResFinder':
            results_file='./benchmarking2_kma/resfinder/Results/summary/loose/'+str(species.replace(" ", "_"))+'.csv'
            results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
            score=results.loc[:,fscore].to_list()
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
                score=results.loc[:,fscore].to_list()
            elif f_kma:
                results_file=results_file+'_score_final_PLOT.txt'
                fscore_="weighted-"+fscore
                results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
                score=results.loc[:,fscore_].to_list()
            else:
                results_file=results_file+ '_score_final_Random_PLOT.txt'
                results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
                score=results.loc[:,fscore].to_list()

        if tool=='KmerC':
            if species !='Mycobacterium tuberculosis'  :#no MT information.
                _, results_file = amr_utility.name_utility.GETsave_name_final(fscore,species, f_kma, f_phylotree, '')
                results_file='./patric_2022/'+results_file
                results=pd.read_csv(results_file + '_SummeryBenchmarking_PLOT.txt', header=0, index_col=0,sep="\t")
                score=results.loc[:,fscore].to_list()
            else:
                score=np.empty((len(antibiotics)))
                score[:] = np.NaN
                score=score.tolist()


        if tool=='Seq2Geno2Pheno':
            if species !='Mycobacterium tuberculosis':#no MT information.
                _, results_file = amr_utility.name_utility.GETsave_name_final(fscore,species, f_kma, f_phylotree, '')
                results_file='./seq2geno/'+results_file
                results=pd.read_csv(results_file + '_SummeryBenchmarking_PLOT.txt', header=0, index_col=0,sep="\t")
                score=results.loc[:,fscore].to_list()
            else:
                score=np.empty((len(antibiotics)))
                score[:] = np.NaN
                score=score.tolist()
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
                score=results.loc[:,fscore].to_list()
            else:
                score=np.empty((len(antibiotics)))
                score[:] = np.NaN
                score=score.tolist()

        if tool=='Kover':

            _, results_file = amr_utility.name_utility.GETsave_name_final(fscore,species, f_kma, f_phylotree, '')
            if f_kma:
                results_file='./kover/'+results_file
            elif f_phylotree:
                results_file='./kover_tree/'+results_file
            else:
                results_file='./kover_random/'+results_file
            results=pd.read_csv(results_file + '_SummeryBenchmarking_PLOT.txt', header=0, index_col=0,sep="\t")
            score=results.loc[:,fscore].to_list()

        if tool=='Baseline (Majority)':

            results_file,_ = amr_utility.name_utility.GETsave_name_final(fscore,species, f_kma, f_phylotree, 'majority')
            if f_kma:
                results_file='./majority/'+results_file
            elif f_phylotree:
                results_file='./majority/'+results_file
            else:
                results_file='./majority/'+results_file

            results=pd.read_csv(results_file + '_PLOT.txt', header=0, index_col=0,sep="\t")

            score=results.loc[:,fscore].to_list()



        data.append(score)
    return data


def ComBySpecies(level,s, fscore, cv_number, f_phylotree, f_kma,f_all):
    if fscore=='f1_macro' or fscore=='accuracy':
        tool_list=['Point-/ResFinder', 'Neural networks', 'KmerC', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover','Baseline (Majority)']
    else:
        tool_list=['Point-/ResFinder', 'Neural networks', 'KmerC', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover']
    data = pd.read_csv('metadata/' + str(level) + '_Species_antibiotic_FineQuality.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    if f_all == False:#should not use it.
        data = data.loc[s, :]

    s_radar=['Escherichia coli','Staphylococcus aureus','Salmonella enterica','Klebsiella pneumoniae','Pseudomonas aeruginosa',
                'Acinetobacter baumannii','Streptococcus pneumoniae','Mycobacterium tuberculosis']
    s_bar=['Campylobacter jejuni','Enterococcus faecium','Neisseria gonorrhoeae']

    data_radar=data.loc[s_radar, :]
    data_bar=data.loc[s_bar, :]
    if f_phylotree:
        data_radar=data_radar.loc[['Escherichia coli','Staphylococcus aureus','Salmonella enterica','Klebsiella pneumoniae','Pseudomonas aeruginosa','Acinetobacter baumannii','Streptococcus pneumoniae'],:]


    df_species_radar = data_radar.index.tolist()
    antibiotics_radar = data_radar['modelling antibiotics'].tolist()
    df_species_bar = data_bar.index.tolist()
    antibiotics_bar = data_radar['modelling antibiotics'].tolist()

    amr_utility.file_utility.make_dir('log/results/')


    # theta = radar_factory(9, frame='circle')
    # fig, axs = plt.subplots(figsize=(20, 25), nrows=4, ncols=3)
    fig, axs = plt.subplots(4,3,figsize=(20, 25))
    # fig.subplots_adjust(top=0.88)
    plt.tight_layout(pad=4.5)
    fig.subplots_adjust(wspace=0.25, hspace=0.4, top=0.96, bottom=0.03)
    if f_phylotree:
        title='Performance w.r.t. phylo-tree-based folds ('+fscore+')'
    elif f_kma:
        title='Performance w.r.t. KMA-based folds ('+fscore+')'
    else:
        title='Performance w.r.t. random folds ('+fscore+')'

    # add legend relative to top-left plot
    labels = tool_list
    # fig.suptitle(title,size=20, weight='bold')
    [axi.set_axis_off() for axi in axs.ravel()[0:9]]
    blue=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
    orange=(1.0, 0.4980392156862745, 0.054901960784313725)
    green= (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
    purp=(0.5803921568627451, 0.403921568627451, 0.7411764705882353)
    red=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392)
    red='#ef3b2c'
    brown=(0.5490196078431373, 0.33725490196078434, 0.29411764705882354)
    if fscore=='f1_macro' or fscore=='accuracy':
        colors = [blue,brown, orange,purp , green , red, "black"]# #ffd343
    else:
        colors = [blue,brown, orange,purp , green , red]
    # colors=sns.color_palette(None,7)
    # for ax in axs[0, :]:
    #     ax.remove()




    # -----------------------------------------------------------------------------------------------
    # 1. ploting radar graphs
    # ------------------------------------------------------------------------------------------------
    i=1
    for species, antibiotics_selected in zip(df_species_radar, antibiotics_radar):

        antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
        # N = len(antibiotics)
        # theta = radar_factory(N, frame='circle')
        # fig_, axs_=axs[i//3,i%3].plot(subplot_kw=dict(projection='radar'))
        # plt.style.use('ggplot')
        axs_ = fig.add_subplot(4,3,i,polar= 'spine')#
        data = combine_data(species,antibiotics,fscore, f_phylotree, f_kma,tool_list)
        # print(data)
        spoke_labels = antibiotics#antibiotics
        #if too long label, make it 2 lines.
        for i_anti in spoke_labels:
            if '/' in i_anti:
                posi=i_anti.find('/')
                _i_anti=i_anti[:(posi+1)] + '\n' + i_anti[(posi+1):]
                spoke_labels=[_i_anti if x==i_anti else x for x in spoke_labels]

        theta = np.linspace(0, 2*np.pi, len(antibiotics), endpoint=False)
        # Plot the four cases from the example data on separate axes
        # for ax, (title, case_data) in zip(axs.flat, data):
        axs_.set_rgrids([0, 0.2, 0.4, 0.6, 0.8,1])
        axs_.set_title(species, weight='bold',style='italic', size=20, position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        p_radar=[]
        for d, color in zip(data, colors):
            # axs_.plot(theta, d, marker='o', markersize=4,color=color,dashes=[6, 2])
            d=np.concatenate((d,[d[0]]))
            theta_=np.concatenate((theta,[theta[0]]))
            p_,=axs_.plot(theta_, d,  'o-', markersize=4,color=color,dashes=[6, 2])
            p_radar.append(p_)
            # ax.fill(theta, d, facecolor=color, alpha=0.25)
        # Circle((0.5, 0.5), 0.5)
        axs_.set(ylim=(0.2, 1.0))
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8,0.1])

        axs_._gen_axes_spines()
        axs_.set_thetagrids(np.degrees(theta), spoke_labels)
        # axs_.set_yticks([])
        # axs_.yaxis.grid(False)
        # axs_.xaxis.grid(False)
        # if species=="Streptococcus pneumoniae":
        #     axs_.tick_params(axis='both', which='major', pad=20,labelrotation=15)
        # else:
        #     axs_.tick_params(axis='both', which='major', pad=20,labelrotation=15)

        axs_.tick_params(axis='x', which='major', pad=25,labelsize=12)
        axs_.spines["start"].set_color("gray")
        axs_.spines["polar"].set_color("gray")
        i+=1





    # fig.text(0.5, 0.965, title,
    #          horizontalalignment='center', color='black', weight='bold',
    #          fontsize=18)
    # legend graph
    # legend = axs_.legend(labels, loc=(1.8,0.1),labelspacing=1, fontsize=10)
    axs_ = fig.add_subplot(4,3,9)
    axs_.set_axis_off()
    if fscore=='f1_macro' or fscore=='accuracy':
        legend_elements = p_radar+[
                           Patch(facecolor=colors[0],edgecolor= colors[0]),Patch(facecolor=colors[1],edgecolor= colors[1]),
                           Patch(facecolor=colors[2],edgecolor= colors[2]),Patch(facecolor=colors[3],edgecolor= colors[3]),
                           Patch(facecolor=colors[4],edgecolor= colors[4]),Patch(facecolor=colors[5],edgecolor= colors[5]),
                           Patch(facecolor=colors[6],edgecolor= colors[6])]
        axs_.legend(handles=legend_elements,  labels=['', '','', '', '','', '']+labels,ncol=2, handlelength=3, edgecolor='grey',
          borderpad=0.7, handletextpad=1.5, columnspacing=0,labelspacing=1,loc='lower right',title='Software',fontsize=16,title_fontsize=16)
    else:
        legend_elements = p_radar+[
                               Patch(facecolor=colors[0],edgecolor= colors[0]),Patch(facecolor=colors[1],edgecolor= colors[1]),
                               Patch(facecolor=colors[2],edgecolor= colors[2]),Patch(facecolor=colors[3],edgecolor= colors[3]),
                               Patch(facecolor=colors[4],edgecolor= colors[4]),Patch(facecolor=colors[5],edgecolor= colors[5])]
        axs_.legend(handles=legend_elements,  labels=['', '','', '', '','']+labels,ncol=2, handlelength=3, edgecolor='grey',
          borderpad=0.7, handletextpad=1.5, columnspacing=0,labelspacing=1,loc='lower right',title='Software',fontsize=16,title_fontsize=16)
    # fig, ax = plt.subplots()

    # axs_ .legend(handles=legend_elements, loc='center', ncol=2, handlelength=3)
    # plt.legend(p,labels, loc=(1.8,0.1),labelspacing=1, fontsize=10)

    # -----------------------------------------------------------------------------------------------
    # 2. ploting bar graphs
    # ------------------------------------------------------------------------------------------------
    i=9

    for species, antibiotics_selected in zip(df_species_bar, antibiotics_bar):
        antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
        # plt.style.use('ggplot')
        # axs_ = fig.add_subplot(4,3,i)#Campylobacter jejuni

        data=combine_data(species,antibiotics,fscore, f_phylotree, f_kma,tool_list)

        df=pd.DataFrame(data, index=labels, columns=antibiotics)
        df=prepare_data(df,fscore)

        print(df)
        # print(df1)
        row = (i // 3)
        col = i % 3
        i+=1

        # g = df.plot(ax=axs[row, col],kind="bar",color=colors, x='software',y=antibiotics)
        g = sns.barplot(x="antibiotics", y=fscore, hue='software',
                        data=df, dodge=True, ax=axs[row, col],palette=colors)
        g.set_title(species,style='italic', weight='bold',size=20)

        g.set(ylim=(0.2, 1.0))
        g.set_ylabel(fscore,size = 16)
        g.set_xlabel('')
        # g.set_xticklabels(g.get_xticks(), size=16)
        # g.set_xticklabels([str(i) for i in g.get_xticks()], fontsize = 16)
        g.set_xticklabels(g.get_xticklabels(),fontsize = 16 )

        # if species =='Neisseria gonorrhoeae':
        #     # plt.legend(loc='upper left',bbox_to_anchor=(1,0.5))
        #     # sns.move_legend(g, "upper left", bbox_to_anchor=(.55, .45), title='Species')
        #     # sns.move_legend(g, "upper left", bbox_to_anchor=(.55, .45))
        # else:
        g.legend_.remove()
        # sns.boxplot(x="antibiotic", y=Tscore, hue="selection Method",
        #                          data=summary_plot, dodge=True, width=0.4)
        # df.plot(ax=axs[i//3,i%3],kind="bar",color=color)
        # df.plot(legend=False)
        # axbig = fig.add_subplot(4,3,11)#Enterococcus faecium
        #
        # axbig = fig.add_subplot(4,3,12)#Neisseria gonorrhoeae

    fig.savefig('log/results/'+'kma_'+str(f_kma)+'_tree_'+str(f_phylotree)+'_'+fscore+'.png')
