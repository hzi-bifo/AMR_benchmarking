import os
import numpy as np
import ast
import amr_utility.name_utility
import amr_utility.graph_utility
import amr_utility.file_utility
import argparse
import amr_utility.load_data
import analysis_results.extract_score
import analysis_results.make_table
import analysis_results.math_utility
import pandas as pd
import pickle
import statistics
from scipy.stats import ttest_rel
import math
from collections import Counter
from pandas.plotting import table
import matplotlib.pyplot as plt
import seaborn as sns
from mycolorpy import colorlist as mcp

#old. 11 folds versions
#plot a comparative graph of single-s model and discrete & concatenated multiple-s model.
# python main_nn_analysis_hyper.py -f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -f_multi -f_all
# python main_nn_analysis_hyper.py -f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -f_concat -f_all




def combine_data(list_species,anti_list,fscore,level,learning,epochs,f_fixed_threshold,f_nn_base,f_optimize_score,merge_name):



    save_name_score_final = amr_utility.name_utility.GETname_multi_bench_save_name_final(fscore,merge_name,
                                                                                                 'all_possible_anti', level,
                                                                                                 learning,
                                                                                                 epochs, f_fixed_threshold,
                                                                                                 f_nn_base,
                                                                                                 f_optimize_score)




    save_name_score_final_concat=amr_utility.name_utility.GETname_multi_bench_save_name_concatM_final(merge_name,merge_name,
                                                                                                        level,learning,epochs,
                                                                                                        f_fixed_threshold,
                                                                                                        f_nn_base,
                                                                                                        f_optimize_score,
                                                                                                        0.8,
                                                                                                        0.6)
    print('check!',save_name_score_final_concat)
    # 1. single-s model
    pd_score_s=pd.read_csv(os.path.dirname(save_name_score_final)+ '/single_species_f1_macro.txt', index_col=0,sep="\t")
    # 2. Multi-species discrete-databases model
    pd_score_m=pd.read_csv(save_name_score_final + '_score_final.txt', index_col=0,sep="\t")
    # 3. concated multi-s mixed species model
    pd_score_c=pd.read_csv(save_name_score_final_concat + '_score_final.txt', index_col=0,sep="\t")

    # print(pd_score_s)
    # print(pd_score_m)
    pd_score_s=pd_score_s.T
    pd_score_m=pd_score_m[fscore]
    pd_score_c=pd_score_c[fscore]
    result_temp = pd.concat([pd_score_s, pd_score_m], axis=1, join="inner")
    result = pd.concat([result_temp, pd_score_c], axis=1, join="inner")
    result=result.T
    data=result.to_numpy()
    # print(data)
    # exit()
    return data

def adjust_lable(axs_,antibiotics,font_size):
        XTICKS = axs_.xaxis.get_major_ticks()
        # X_VERTICAL_TICK_PADDING = 40
        # X_HORIZONTAL_TICK_PADDING = 25
        n_lable=len(antibiotics)
        # for tick in XTICKS[math.floor(3*n_lable/4):-1]:
        #     tick.set_pad(X_HORIZONTAL_TICK_PADDING)
        angles = np.linspace(0,2*np.pi,len(axs_.get_xticklabels())+1)
        # angles[np.cos(angles) < 0] = angles[np.cos(angles) < 0] + np.pi
        angles = np.rad2deg(angles)
        # if n_lable>7:
        #     for tick in XTICKS[0::2]:
        #         tick.set_pad(10)
        #     for tick in XTICKS[1::2]:
        #         tick.set_pad(40)
        labels = []
        i=0

        for label, angle in zip(axs_.get_xticklabels(), angles):
            # label.set_rotation(90-angle*(365/n_lable))
            # label.set_rotation_mode("anchor")
            if i>n_lable/2:
                angle=angle-85
            else:
                angle=angle-85
            x,y = label.get_position()
            # print(label.get_text())
            # print('---')
            lab = axs_.text(x,y, label.get_text(), transform=label.get_transform(),
                          ha=label.get_ha(), va=label.get_va(),size=font_size)


            lab.set_rotation(angle)
            i+=1
            lab.set_rotation_mode("anchor")
            labels.append(lab)
        axs_.set_xticklabels([])
def extract_info(fscore,level,f_all,threshold_point,min_cov_point,learning,epochs,f_optimize_score,f_fixed_threshold,
                 f_nn_base,f_phylotree,cv):
    data = pd.read_csv('metadata/' +str(level)+'_multi-species_summary.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    # --------------------------------------------------------
    print(data)
    # print(data.index)
    merge_name = []
    if f_all:
        list_species = data.index.tolist()[:-1]
        data = data.loc[list_species, :]

    else:
        pass
    for n in list_species:
        merge_name.append(n[0] + n.split(' ')[1][0])
    merge_name = '_'.join(merge_name)  # e.g.Se_Kp_Pa

    # data=data.loc[list_species, :]
    df_anti = data.dot(data.columns + ';').str.rstrip(';')


    fig, axs = plt.subplots(1,1,figsize=(35, 20))
    axs.set_axis_off()
    axs_ = fig.add_subplot(1,1,1,polar= 'spine')
    # plt.tight_layout(pad=5)
    # fig.subplots_adjust(wspace=0, hspace=0.2, top=0, bottom=0)
    plt.subplots_adjust(left=0.08, right=0.6, top=1.02, bottom=-0.05)
    anti_list=[]
    for species in list_species:
        anti = df_anti[species].split(';')
        anti_list=anti_list+anti
    anti_list=list(set(anti_list))
    anti_list.sort()
    print(anti_list)


    data_plot = combine_data(list_species,anti_list,fscore,level,learning,epochs,f_fixed_threshold,f_nn_base,f_optimize_score,merge_name)
    # print(data_plot)
    with open('../src/AntiAcronym_dict.pkl', 'rb') as f:
            map_acr = pickle.load(f)
    spoke_labels= [map_acr[x] for x in anti_list]
    list_species=[x[0] +". "+ x.split(' ')[1] for x in list_species]
    labels = list_species+['Discrete databases\nmulti-species model'] +['Concatenated databases\nmulti-species model']
    #if too long label, make it 2 lines.
    for i_anti in spoke_labels:
        if '/' in i_anti:
            posi=i_anti.find('/')
            _i_anti=i_anti[:(posi+1)] + '\n' + i_anti[(posi+1):]
            spoke_labels=[_i_anti if x==i_anti else x for x in spoke_labels]
    colors=mcp.gen_color(cmap="tab10",n=len(labels)-1)

    color_multi=colors[0]
    colors.pop(0)
    colors.append(color_multi)
    colors.append('#004529')



    theta = np.linspace(0, 2*np.pi, len(spoke_labels), endpoint=False)
    # Plot the four cases from the example data on separate axes
    # for ax, (title, case_data) in zip(axs.flat, data):

    # axs_.set_title(species, weight='bold',style='italic', size=22, position=(0.5, 1.1),
    #              horizontalalignment='center', verticalalignment='center',pad=60)
    p_radar=[]
    i=0
    for d, color in zip(data_plot, colors):
        # axs_.plot(theta, d, marker='o', markersize=4,color=color,dashes=[6, 2])
        d=np.concatenate((d,[d[0]]))
        theta_=np.concatenate((theta,[theta[0]]))
        if i in [9,10]:
            p_,=axs_.plot(theta_, d,  's-', markersize=20,color=color,dashes=[6, 2],linewidth=10,zorder=1, alpha=0.8)
        else:
            p_,=axs_.plot(theta_, d,  'o', markersize=20,color=color,zorder=3)
        # p_,=axs_.plot(theta_, d,  'o-', markersize=4,color=color)
        p_radar.append(p_)
        i+=1
        # ax.fill(theta, d, facecolor=color, alpha=0.25)
    # Circle((0.5, 0.5), 0.5)
    # axs_.set_rgrids([0, 0.2, 0.4, 0.6, 0.8,1])
    axs_.yaxis.grid(False)
    axs_.xaxis.grid(False)
    axs_.tick_params(width=10)
    axs_._gen_axes_spines()
    axs_.set_thetagrids(np.degrees(theta), spoke_labels)

    print(labels)
    # labels=[x.replace(' ','\n') if x == 'Discrete databases multi-species model' else x for x in labels]

    leg=axs_.legend(spoke_labels,labels= labels, ncol=1, fontsize=38, loc=(1.2,0.1),markerscale=1,labelspacing=1)#
    # for line in leg.get_lines():
    #     line.set_linewidth(5.0)
    axs_.set_xticklabels(axs_.get_xticklabels(),fontsize = 40)

    axs_.tick_params(axis='x', which='major', color='grey',grid_linestyle='dashed', pad=10,labelsize=60)
    # axs_.spines["start"].set_color("grey")
    # axs_.spines["polar"].set_color("grey")
    axs_.set(ylim=(-0.05, 1.0))
    plt.yticks([0,  0.5, 1])

    axs_.tick_params(axis='y', which='major', labelsize=50)
    axs_.set_ylabel(fscore+'\n\n',size =40)
    plt.grid(axis='y',color='silver', dashes=[3,9], linewidth=3)
    plt.grid(axis='x',color='silver', dashes=[3,9], linewidth=3)
    axs_.tick_params(axis='x', which='major', pad=60,labelsize=60)
    axs_.set(ylabel=None)
    # adjust_lable(axs_,spoke_labels,60)
    fig.savefig('log/results/' + str(level) + '/Compare_single_discrete.pdf')








if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--level', default='loose', type=str, required=False,
                        help='Quality control: strict or loose')
    parser.add_argument('-fscore', '--fscore', default='f1_macro', type=str, required=False,
                        help='the score used to choose the best classifier for each antibiotic. Can be f1_pos'
                             'f1_neg, accuracy.')
    parser.add_argument('-t_p', '--threshold_point', default=0.8, type=float,
                        help='Threshold for identity of Pointfinder. ')
    parser.add_argument('-l_p', '--min_cov_point', default=0.6, type=float,
                        help=' Minimum (breadth-of) coverage of Pointfinder. ')
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
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    parser.add_argument("-cv", "--cv_number", default=10, type=int,
                        help='CV splits number')
    parsedArgs = parser.parse_args()
    # parser.print_help()
    # print(parsedArgs)
    extract_info(parsedArgs.fscore,parsedArgs.level,parsedArgs.f_all,parsedArgs.threshold_point,parsedArgs.min_cov_point,
                 parsedArgs.learning,parsedArgs.epochs,parsedArgs.f_optimize_score,parsedArgs.f_fixed_threshold,parsedArgs.f_nn_base,
                 parsedArgs.f_phylotree,parsedArgs.cv_number)



