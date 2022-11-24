import sys,os
# sys.path.append('../')
sys.path.insert(0, os.getcwd())
import argparse
import pandas as pd
import json
from mycolorpy import colorlist as mcp
from pylab import *

#plot a comparative radar graph of SSSA and SSMA model.


def adjust_lable(axs_,antibiotics,font_size):
        XTICKS = axs_.xaxis.get_major_ticks()
        n_lable=len(antibiotics)
        angles = np.linspace(0,2*np.pi,len(axs_.get_xticklabels())+1)
        angles = np.rad2deg(angles)
        for tick in XTICKS:
            tick.set_pad(250)
        labels = []
        i=0

        for label, angle in zip(axs_.get_xticklabels(), angles):

            if i>n_lable/2:
                angle=angle
            else:
                angle=angle
            x,y = label.get_position()
            lab = axs_.text(x,y, label.get_text(), transform=label.get_transform(),
                          ha=label.get_ha(), va=label.get_va(),size=font_size)
            lab.set_rotation(angle)
            i+=1
            lab.set_rotation_mode("anchor")
            labels.append(lab)
        axs_.set_xticklabels([])


def extract_info(fscore,output_path):


    temp_results=output_path+ 'Results/other_figures_tables/S8_Aytan-Aktug_SSMA'
    data = pd.read_csv(temp_results+'_multi.csv' ,sep="\t" )
    anti_list=data['antibiotics'].to_list()
    species_list=data['species'].to_list()
    data_plot=data[[ 'Single-species multi-antibiotics Aytan-Aktug', 'Single-species-antibiotic Aytan-Aktug',\
                     'Single-species multi-antibiotics Aytan-Aktug_std', 'Single-species-antibiotic Aytan-Aktug_std' ]].T.values

    with open('./data/AntiAcronym_dict.json') as f:
        map_acr = json.load(f)
    anti_labels= [map_acr[x] for x in anti_list]
    species_lable=['$\mathbf{'+x.split(' ')[0][0]+'.' +x.split(' ')[0]+'}$' + '| ' for x in species_list]


    labels = ['Single-species multi-antibiotics F1-macro mean','Single-species-antibiotic F1-macro mean','Single-species multi-antibiotics F1-macro standard deviation','Single-species-antibiotic F1-macro standard deviation' ]
    #if too long label, make it 2 lines.

    rcParams['mathtext.fontset'] = 'custom'
    rcParams['mathtext.it'] = 'stixsans:italic'
    rcParams['mathtext.bf'] = 'stixsans:italic:bold'


    spoke_labels=[m+n for m,n in zip(species_lable,anti_labels)]
    colors=mcp.gen_color(cmap="tab20",n=4)

    color_temp=colors[0]
    colors.pop(0)
    colors.insert( 2, color_temp)

    fig, axs = plt.subplots(1,1,figsize=(30, 35))
    axs.set_axis_off()
    axs_ = fig.add_subplot(1,1,1,polar= 'spine')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.65, bottom=0.22)

    theta = np.linspace(0, 2*np.pi, len(spoke_labels), endpoint=False)
    p_radar=[]
    i=0
    for d, color in zip(data_plot, colors):

        d=np.concatenate((d,[d[0]]))
        theta_=np.concatenate((theta,[theta[0]]))
        if i==3:
            p_,=axs_.plot(theta_, d,  's-', markersize=15,color=color,linewidth=10,zorder=2, alpha=0.8)
        else:
            p_,=axs_.plot(theta_, d,  's-', markersize=15,color=color,linewidth=10,zorder=3, alpha=0.8)

        p_radar.append(p_)
        i+=1

    axs_.yaxis.grid(False)
    axs_.xaxis.grid(False)
    axs_.tick_params(width=10)
    axs_._gen_axes_spines()
    axs_.set_thetagrids(np.degrees(theta), spoke_labels)
    leg=axs_.legend(spoke_labels,labels= labels, ncol=1, fontsize=38, loc=(0.05,1.5),markerscale=1,labelspacing=1)#
    axs_.set_xticklabels(axs_.get_xticklabels(),fontsize = 40)
    axs_.tick_params(axis='x', which='major', color='grey',grid_linestyle='dashed', pad=10,labelsize=60)
    axs_.set(ylim=(-0.05, 1.0))
    plt.yticks([0,  0.5, 1])

    axs_.tick_params(axis='y', which='major', labelsize=40)
    axs_.set_ylabel(fscore+'\n\n',size =40)
    plt.grid(axis='y',color='silver', dashes=[10,30], linewidth=5)
    axs_.set(ylabel=None)
    adjust_lable(axs_,spoke_labels,40)

    fig.savefig(output_path+'Results/final_figures_tables/F7_Compare_SSMA.pdf')
    fig.savefig(output_path+'Results/final_figures_tables/F7_Compare_SSMA.png')#only for DOC work.







if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fscore', '--fscore', default='f1_macro', type=str, required=False,
                        help='the score used to choose the best classifier for each antibiotic. Can be f1_pos'
                             'f1_neg, accuracy.')
    parser.add_argument('-o', '--output_path', default='./', type=str, required=False,
                    help='Results folder.')
    parsedArgs = parser.parse_args()
    extract_info(parsedArgs.fscore,parsedArgs.output_path)



