#!/usr/bin/python
import sys,os
# sys.path.append('../')
sys.path.insert(0, os.getcwd())
from src.amr_utility import name_utility,file_utility
import numpy as np
import argparse
import pandas as pd
import json
import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp



def combine_data(fscore,learning,epochs,f_fixed_threshold,f_nn_base,f_optimize_score,merge_name,output_path):

    f_kma=True
    f_phylotree=False


    save_name_score_final=name_utility.GETname_AAresult('AytanAktug',merge_name,learning, epochs,f_fixed_threshold,f_nn_base
                                                        ,'f1_macro',f_kma,f_phylotree,'MSMA_discrete',output_path)



    save_name_score_final_concatM=  name_utility.GETname_AAresult('AytanAktug',merge_name,learning, epochs,f_fixed_threshold,\
                                         f_nn_base,f_optimize_score,f_kma,f_phylotree,'MSMA_concat_mixedS',output_path)

    # 1. single-s model
    pd_score_s=pd.read_csv(save_name_score_final+'_SSSAmapping_'+fscore+'.txt', index_col=0,sep="\t")
    # 2. Multi-species discrete-databases model
    pd_score_m=pd.read_csv(save_name_score_final + '_SummaryBenchmarking.txt', index_col=0,sep="\t")
    # 3. concated multi-s mixed species model
    pd_score_c=pd.read_csv(save_name_score_final_concatM + '_SummaryBenchmarking.txt', index_col=0,sep="\t")


    pd_score_s=pd_score_s.T
    pd_score_m=pd_score_m[fscore]
    pd_score_c=pd_score_c[fscore]
    result_temp = pd.concat([pd_score_s, pd_score_m], axis=1, join="inner")
    result = pd.concat([result_temp, pd_score_c], axis=1, join="inner")
    result=result.T
    data=result.to_numpy()

    return data

def adjust_lable(axs_,antibiotics,font_size):
        n_lable=len(antibiotics)
        angles = np.linspace(0,2*np.pi,len(axs_.get_xticklabels())+1)
        angles = np.rad2deg(angles)
        labels = []
        i=0

        for label, angle in zip(axs_.get_xticklabels(), angles):
            if i>n_lable/2:
                angle=angle-85
            else:
                angle=angle-85
            x,y = label.get_position()
            lab = axs_.text(x,y, label.get_text(), transform=label.get_transform(),
                          ha=label.get_ha(), va=label.get_va(),size=font_size)
            lab.set_rotation(angle)
            i+=1
            lab.set_rotation_mode("anchor")
            labels.append(lab)
        axs_.set_xticklabels([])
def extract_info(fscore,level,f_all,learning,epochs,f_optimize_score,f_fixed_threshold,
                 f_nn_base,output_path):

    data = pd.read_csv('./data/PATRIC/meta/'+str(level)+'_multi-species_summary.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    merge_name = []
    if f_all:
        list_species = data.index.tolist()[:-1]
        data = data.loc[list_species, :]
    else:
        print('Please set -f_all.')
        exit(0)
    for n in list_species:
        merge_name.append(n[0] + n.split(' ')[1][0])
    merge_name = '_'.join(merge_name)  # e.g.Se_Kp_Pa

    df_anti = data.dot(data.columns + ';').str.rstrip(';')
    fig, axs = plt.subplots(1,1,figsize=(31, 23))
    axs.set_axis_off()
    axs_ = fig.add_subplot(1,1,1,polar= 'spine')
    plt.subplots_adjust(left=0.08, right=0.6, top=1.02, bottom=-0.05)
    anti_list=[]
    for species in list_species:
        anti = df_anti[species].split(';')
        anti_list=anti_list+anti
    anti_list=list(set(anti_list))
    anti_list.sort()



    data_plot = combine_data(fscore,learning,epochs,f_fixed_threshold,f_nn_base,f_optimize_score,merge_name,output_path)
    # print(data_plot)

    with open('./data/AntiAcronym_dict.json') as f:
        map_acr = json.load(f)

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

        p_radar.append(p_)
        i+=1

    axs_.yaxis.grid(False)
    axs_.xaxis.grid(False)
    axs_.tick_params(width=10)
    axs_._gen_axes_spines()
    axs_.set_thetagrids(np.degrees(theta), spoke_labels)
    leg=axs_.legend(spoke_labels,labels= labels, ncol=1, fontsize=38, loc=(1.2,0.1),markerscale=1,labelspacing=1)#
    axs_.set_xticklabels(axs_.get_xticklabels(),fontsize = 40)

    axs_.tick_params(axis='x', which='major', color='grey',grid_linestyle='dashed', pad=10,labelsize=60)
    axs_.set(ylim=(-0.05, 1.0))
    plt.yticks([0,  0.5, 1])

    axs_.tick_params(axis='y', which='major', labelsize=50)
    axs_.set_ylabel(fscore+'\n\n',size =40)
    plt.grid(axis='y',color='silver', dashes=[3,9], linewidth=3)
    plt.grid(axis='x',color='silver', dashes=[3,9], linewidth=3)
    axs_.tick_params(axis='x', which='major', pad=60,labelsize=60)
    axs_.set(ylabel=None)

    file_utility.make_dir(output_path+'Results/final_figures_tables')
    fig.savefig(output_path+'Results/supplement_figures_tables/S2_Compare_MSMA_oldradar.pdf')
    fig.savefig(output_path+'Results/supplement_figures_tables/S2_Compare_MSMA_oldradar.png')#only for DOC work.







if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--level', default='loose', type=str, required=False,
                        help='Quality control: strict or loose')
    parser.add_argument('-fscore', '--fscore', default='f1_macro', type=str, required=False,
                        help='the score used to choose the best classifier for each antibiotic. Can be f1_pos'
                             'f1_neg, accuracy.')
    parser.add_argument('-o', '--output_path', default='./', type=str, required=False,
                        help='Directory to store CV scores.')
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

    parsedArgs = parser.parse_args()

    extract_info(parsedArgs.fscore,parsedArgs.level,parsedArgs.f_all, parsedArgs.learning,parsedArgs.epochs,
                 parsedArgs.f_optimize_score,parsedArgs.f_fixed_threshold,parsedArgs.f_nn_base,parsedArgs.output_path)



