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
from pylab import *
#2 single-s models
#plot a comparative graph of single-a model and  multiple-a model.


def adjust_lable(axs_,antibiotics,font_size):
        XTICKS = axs_.xaxis.get_major_ticks()
        X_VERTICAL_TICK_PADDING = 40
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
        for tick in XTICKS:
            tick.set_pad(250)
        labels = []
        i=0

        for label, angle in zip(axs_.get_xticklabels(), angles):
            # label.set_rotation(90-angle*(365/n_lable))
            # label.set_rotation_mode("anchor")
            if i>n_lable/2:
                angle=angle
            else:
                angle=angle
            x,y = label.get_position()
            # print(label.get_text())
            # print('---')
            lab = axs_.text(x,y, label.get_text(), transform=label.get_transform(),
                          ha=label.get_ha(), va=label.get_va(),size=font_size)

            # if i>n_lable/2:
            #     lab.set_rotation( 90+i*(365/n_lable))
            # elif i==0:
            #     pass
            # else:
            #     lab.set_rotation(-90+i*(365/n_lable))
            #

            # colors = iter([plt.cm.pastel1(i) for i in range(len(anti_share))])


            # colors=mcp.gen_color(cmap="Set3",n=len(anti_share))

            # colors=['#77dd77','#836953','#89cff0','#99c5c4','#9adedb','#aa9499','#aaf0d1','#b2fba5','#b39eb5','#bdb0d0','']
            # lab.set_bbox(dict(facecolor=chose_color, alpha=0.5, edgecolor=chose_color))
            # if label.get_text() in anti_share:
            #     index_color=anti_share.index(label.get_text())
            #     lab.set_bbox(dict(facecolor=colors[index_color], alpha=0.5, edgecolor='white'))
            # else:
            #     lab.set_bbox(dict(facecolor='silver', alpha=0.5, edgecolor='white'))
            lab.set_rotation(angle)
            i+=1
            lab.set_rotation_mode("anchor")
            labels.append(lab)
        axs_.set_xticklabels([])
        # plt.setp(axs_.get_xticklabels(), backgroundcolor="limegreen")

# def adjust_lable(axs_,antibiotics,font_size):
#         XTICKS = axs_.xaxis.get_major_ticks()
#         # X_VERTICAL_TICK_PADDING = 40
#         # X_HORIZONTAL_TICK_PADDING = 25
#         n_lable=len(antibiotics)
#         for tick in XTICKS[math.floor(3*n_lable/4):-1]:
#             tick.set_pad(X_HORIZONTAL_TICK_PADDING)
#         angles = np.linspace(0,2*np.pi,len(axs_.get_xticklabels())+1)
#         # angles[np.cos(angles) < 0] = angles[np.cos(angles) < 0] + np.pi
#         angles = np.rad2deg(angles)
#
#         labels = []
#         i=0
#
#         target=axs_.get_xticklabels()
#         target.append(target.pop(0))
#         angles=angles.tolist()
#         angles.append(angles.pop(0))
#         for label, angle in zip(target, angles):
#
#             angle=angle+10
#             x,y = label.get_position()
#             lab = axs_.text(x,y, label.get_text(), transform=label.get_transform(),
#                           ha=label.get_ha(), va=label.get_va(),size=font_size)
#
#             lab.set_rotation(angle)
#             i+=1
#             lab.set_rotation_mode("anchor")
#             labels.append(lab)
#             axs_.set_xticklabels([])


def extract_info(fscore,level,f_all,threshold_point,min_cov_point,learning,epochs,f_optimize_score,f_fixed_threshold,
                 f_nn_base,f_phylotree,cv):



    data = pd.read_csv('../log/results/cv_results_multi.csv' ,sep="\t" )
    anti_list=data['antibiotics'].to_list()
    species_list=data['species'].to_list()
    data_plot=data[[ 'Single-species multi-antibiotics Aytan-Aktug', 'Single-species-antibiotic Aytan-Aktug',\
                     'Single-species multi-antibiotics Aytan-Aktug_std', 'Single-species-antibiotic Aytan-Aktug_std' ]].T.values
    # print(data_plot)

    # print(data_plot)
    with open('../src/AntiAcronym_dict.pkl', 'rb') as f:
            map_acr = pickle.load(f)
    anti_labels= [map_acr[x] for x in anti_list]
    species_lable=['$\mathbf{'+x.split(' ')[0][0]+'.' +x.split(' ')[0]+'}$' + '| ' for x in species_list]

    # list_species=[x[0] +". "+ x.split(' ')[1] for x in list_species]
    labels = ['Single-species multi-antibiotics Mean','Single-species-antibiotic Aytan-Aktug Mean','Single-species multi-antibiotics Standard deviation','Single-species-antibiotic Aytan-Aktug Standard deviation' ]
    #if too long label, make it 2 lines.


    rcParams['mathtext.fontset'] = 'custom'
    rcParams['mathtext.it'] = 'stixsans:italic'
    rcParams['mathtext.bf'] = 'stixsans:italic:bold'
    # antibiotics_com=['$\mathbf{C. jejuni}$ TE','$\mathbf{E. faecium}$\nVA ','$\mathbf{N. gonorrhoeae}$ AZI','$\mathbf{N. gonorrhoeae}$\nFIX']


    # print(species_lable)
    # print(anti_labels)
    spoke_labels=[m+n for m,n in zip(species_lable,anti_labels)]
    # print(spoke_labels)
    colors=mcp.gen_color(cmap="tab20",n=4)

    color_temp=colors[0]
    colors.pop(0)
    # colors.append(color_temp)
    colors.insert( 2, color_temp)

    fig, axs = plt.subplots(1,1,figsize=(30, 35))
    axs.set_axis_off()
    axs_ = fig.add_subplot(1,1,1,polar= 'spine')
    # plt.tight_layout(pad=40)
    # fig.subplots_adjust(wspace=0, hspace=0.2, top=0, bottom=0)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.65, bottom=0.22)

    theta = np.linspace(0, 2*np.pi, len(spoke_labels), endpoint=False)
    p_radar=[]
    i=0
    for d, color in zip(data_plot, colors):
        # axs_.plot(theta, d, marker='o', markersize=4,color=color,dashes=[6, 2])

        d=np.concatenate((d,[d[0]]))
        theta_=np.concatenate((theta,[theta[0]]))
        if i==3:
            p_,=axs_.plot(theta_, d,  's-', markersize=15,color=color,linewidth=10,zorder=2, alpha=0.8)
        else:
            p_,=axs_.plot(theta_, d,  's-', markersize=15,color=color,linewidth=10,zorder=3, alpha=0.8)

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

    leg=axs_.legend(spoke_labels,labels= labels, ncol=1, fontsize=38, loc=(0.05,1.5),markerscale=1,labelspacing=1)#
    # for line in leg.get_lines():
    #     line.set_linewidth(5.0)
    axs_.set_xticklabels(axs_.get_xticklabels(),fontsize = 40)

    axs_.tick_params(axis='x', which='major', color='grey',grid_linestyle='dashed', pad=10,labelsize=60)
    # axs_.spines["start"].set_color("grey")
    # axs_.spines["polar"].set_color("grey")
    axs_.set(ylim=(-0.05, 1.0))
    plt.yticks([0,  0.5, 1])

    axs_.tick_params(axis='y', which='major', labelsize=40)
    axs_.set_ylabel(fscore+'\n\n',size =40)
    plt.grid(axis='y',color='silver', dashes=[10,30], linewidth=5)
    # plt.grid(axis='x',color='silver', dashes=[10,30], linewidth=3)
    # axs_.tick_params(axis='x', which='major', pad=60,labelsize=30)
    axs_.set(ylabel=None)
    adjust_lable(axs_,spoke_labels,40)
    fig.savefig('log/results/' + str(level) + '/Compare_multiA.pdf')








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



