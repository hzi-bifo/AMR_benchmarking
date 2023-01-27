import sys
import os
# sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.getcwd())
from src.amr_utility import name_utility,file_utility,load_data
from pylab import *
import json
import pandas as pd
from src.benchmark_utility.lib.Com_BySpecies import combine_data_mean, combine_data_std,adjust_lable_bar,radar_factory,prepare_data



class colorStyle:
    I='\x1B[3m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def ComBySpecies(tool_list,level,s, fscore, f_phylotree, f_kma,fig,i,output_path):

    ##tool_list=['Point-/ResFinder', 'Aytan-Aktug', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover','ML Baseline (Majority)']
    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    data_radar=data.loc[s, :]
    df_species_radar = data_radar.index.tolist()
    antibiotics_radar = data_radar['modelling antibiotics'].tolist()

    labels = tool_list

    blue=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
    # orange=(1.0, 0.4980392156862745, 0.054901960784313725)
    green= (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
    purp=(0.5803921568627451, 0.403921568627451, 0.7411764705882353)
    # red=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392)
    red='#ef3b2c'
    # brown=(0.5490196078431373, 0.33725490196078434, 0.29411764705882354)

    colors = [blue,"orange",  purp , green , red, '#653700']# #ffd343brown
    font_size=60
    line_width=12

    # -----------------------------------------------------------------------------------------------
    # 1. ploting radar graphs
    # ------------------------------------------------------------------------------------------------

    for species, antibiotics_selected in zip(df_species_radar, antibiotics_radar):
        print(species)
        antibiotics, _, _ = load_data.extract_info(species, False, level)

        #---------------------------------------------------
        # -------------------Std of scores--------------

        theta = radar_factory(len(antibiotics),'radar'+species, frame='polygon')
        axs_std = plt.subplot(1,1,i, projection='radar'+species)
        data = combine_data_std(species,antibiotics,fscore, f_phylotree, f_kma,tool_list,output_path)
        spoke_labels = antibiotics#antibiotics
        #Add acronym
        with open('./data/AntiAcronym_dict.json') as f:
            map_acr = json.load(f)
        spoke_labels= [map_acr[x] for x in spoke_labels]



        species_title=(species[0] +". "+ species.split(' ')[1] )
        axs_std.set_title(species_title, weight='bold',style='italic', size=60, position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center',pad=80)
        p_radar=[]

        for d, color in zip(data, colors):
            p_ =axs_std.plot(theta, d,  'o-', markersize=25,color=color,dashes=[5,2],linewidth=line_width )
            p_radar.append(p_)

        if species=='Klebsiella pneumoniae' and f_kma==True and fscore=='f1_negative':
            axs_std.set_rgrids([-1,-0.5,0, 0.2,  0.4])
            axs_std.set(ylim=(-1, 0.41))
            plt.yticks([-1,-0.5,0, 0.2,  0.4],size=40)
        elif f_kma==True or f_phylotree==True:
            axs_std.set_rgrids([-1,-0.5,0, 0.2, 0.4])
            axs_std.set(ylim=(-1,  0.41))
            plt.yticks([-1,-0.5,0, 0.2,  0.4],size=40)
        else:
            axs_std.set_rgrids([-1,-0.5,0, 0.2,  0.4])
            axs_std.set(ylim=(-1,  0.41))
            plt.yticks([-1,-0.5,0, 0.2, 0.4],size=40)
        plt.grid(color='white', linestyle='-', linewidth=3)

        axs_std._gen_axes_spines()
        axs_std.set_thetagrids(np.degrees(theta), spoke_labels)
        axs_std.set_facecolor('#d9d9d9')
        # if species=='Streptococcus pneumoniae':
        #     axs_std.tick_params(axis='x', which='major', color='grey',grid_linestyle='dashed', pad=10,labelsize=16,zorder=3)
        # else:
        axs_std.tick_params(axis='x', which='major', color='grey',grid_linestyle='dashed', pad=5,zorder=3)
        #----------------------
        # -----legend---------
        # ------------------
        if i==1:
            leg=axs_std.legend(antibiotics,  labels= labels, ncol=2, loc=(-0.1,1.2),fontsize=50, markerscale=1,frameon=False)
            for line in leg.get_lines():
                line.set_linewidth(10)

        i+=1
        # Adjust tick label positions ------------------------------------
        axs_std.tick_params(axis='x', which='major', pad=50,labelsize=font_size)

        pos1=axs_std.get_position()
        if species=='Klebsiella pneumoniae' and f_kma==False and f_phylotree==False:
            axs_ = fig.add_axes([pos1.x0+0.03575,pos1.y0+0.0128,pos1.width / 1.44,pos1.height / 1.44], projection= 'radar'+species)
        elif f_kma:
            axs_ = fig.add_axes([pos1.x0+0.03575,pos1.y0+0.0128,pos1.width / 1.44,pos1.height / 1.44], projection= 'radar'+species)

        else:
            axs_ = fig.add_axes([pos1.x0+0.121,pos1.y0+0.107,pos1.width / 1.44,pos1.height / 1.44], projection= 'radar'+species)

        #---------------------------------------------------
        # -------------------Mean of scores--------------
        data =combine_data_mean(species,antibiotics,fscore, f_phylotree, f_kma,tool_list,output_path)

        temp_zorder=0
        for d, color in zip(data, colors):
            if temp_zorder==0:
                p_ =axs_.plot(theta, d,  'o-', markersize=25,color=color,linewidth=line_width,zorder=3)#,dashes=[6, 2],
            else:
                p_ =axs_.plot(theta, d,  'o-', markersize=25,color=color,linewidth=line_width)#,dashes=[6, 2],
            temp_zorder+=1

        axs_.set_ylim(ymin=-0.05)
        axs_.set_rgrids([0, 0.5, 1 ],size=40)
        axs_.tick_params(axis='y', which='major', pad=15)
        axs_.yaxis.grid(False)
        axs_.xaxis.grid(False)
        plt.grid(axis='y',color='gray', dashes=[3,3], linewidth=10)
        axs_.set(xticklabels=[])
        axs_.set(xlabel=None)
        axs_.tick_params(axis='x',bottom=False)


def draw(tool_list,level,species, fscore,f_phylotree,f_kma,output_path,save_file_name):

    fig, axs = plt.subplots(1,1,figsize=(20, 23))
    lim_pad=15
    lim_w,lim_h,lim_t,lim_b=0.4,0.3,0.92,-0.15
    axs.set_axis_off()
    plt.tight_layout(pad=lim_pad)
    fig.subplots_adjust(wspace=lim_w, hspace=lim_h, top=lim_t, bottom=lim_b)
    ComBySpecies(tool_list,level,species, fscore, f_phylotree, f_kma,fig,1,output_path)
    fig.savefig(save_file_name+str(species[0].replace(" ", "_")) +'.pdf')
    fig.savefig(save_file_name+str(species[0].replace(" ", "_")) +'.png')
