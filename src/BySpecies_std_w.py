import sys
import os
sys.path.insert(0, os.path.abspath('../'))
import amr_utility.name_utility
import amr_utility.load_data
import amr_utility.file_utility
import src.BySpecies
import src.ByAnti
import ast
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
import matplotlib.projections
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from ast import literal_eval
from mycolorpy import colorlist as mcp
from matplotlib.patches import Patch
from pylab import *
import pickle
import pandas as pd
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

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
            if species !='Mycobacterium tuberculosis':#no MT information.
                _, results_file = amr_utility.name_utility.GETsave_name_final(fscore,species, f_kma, f_phylotree, '')
                results_file='./patric_2022/'+results_file
                results=pd.read_csv(results_file + '_SummeryBenchmarking_PLOT.txt', header=0, index_col=0,sep="\t")
                score=results.loc[:,fscore].to_list()
            else:#'Mycobacterium tuberculosis'
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
            # if species !='Mycobacterium tuberculosis':
            _, results_file = amr_utility.name_utility.GETsave_name_final(fscore,species, f_kma, f_phylotree, '')
            if f_kma:
                results_file='./PhenotypeSeeker_Nov08/'+results_file
            elif f_phylotree:
                results_file='./PhenotypeSeeker_tree/'+results_file
            else:
                results_file='./PhenotypeSeeker_random/'+results_file
            results=pd.read_csv(results_file + '_SummeryBenchmarking_PLOT.txt', header=0, index_col=0,sep="\t")
            score=results.loc[:,fscore].to_list()
            # else:
            #
            #     score=np.empty((len(antibiotics)))
            #     score[:] = np.NaN
            #     score=score.tolist()

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

def combine_data_std(species,antibiotics,fscore, f_phylotree, f_kma,tool_list):
    # This function makes a matrix of all tools' results.
    # 'Point-/ResFinder', 'Neural networks', 'KmerC', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover'
    # todo need to change names before final release.
    data = []
    for tool in tool_list:
        if tool=='Point-/ResFinder':
            # results_file='./benchmarking2_kma/resfinder/Results/summary/loose/'+str(species.replace(" ", "_"))+'.csv'
            # results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
            # score=results.loc[:,fscore].to_list()
            score=np.empty((len(antibiotics)))
            score[:] = np.NaN
            score=score.tolist()
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
                results_file=results_file+'_score_final_Tree_std.txt'
                results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
                score=results.loc[:,fscore].to_list()
            elif f_kma:
                results_file=results_file+'_score_final_std.txt'
                fscore_="weighted-"+fscore
                results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
                score=results.loc[:,fscore_].to_list()
            else:
                results_file=results_file+ '_score_final_Random_std.txt'
                results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
                score=results.loc[:,fscore].to_list()

        # if tool=='KmerC':
        #     if species !='Mycobacterium tuberculosis':#no MT information.
        #         _, results_file = amr_utility.name_utility.GETsave_name_final(fscore,species, f_kma, f_phylotree, '')
        #         results_file='./patric_2022/'+results_file
        #         results=pd.read_csv(results_file + '_SummeryBenchmarking_std.txt', header=0, index_col=0,sep="\t")
        #         score=results.loc[:,fscore].to_list()
        #     else:#'Mycobacterium tuberculosis'
        #         score=np.empty((len(antibiotics)))
        #         score[:] = np.NaN
        #         score=score.tolist()


        if tool=='Seq2Geno2Pheno':
            if species !='Mycobacterium tuberculosis':#no MT information.
                _, results_file = amr_utility.name_utility.GETsave_name_final(fscore,species, f_kma, f_phylotree, '')
                results_file='./seq2geno/'+results_file
                results=pd.read_csv(results_file + '_SummeryBenchmarking_std.txt', header=0, index_col=0,sep="\t")
                score=results.loc[:,fscore].to_list()
            else:
                score=np.empty((len(antibiotics)))
                score[:] = np.NaN
                score=score.tolist()
        if tool=='PhenotypeSeeker':
            # if species !='Mycobacterium tuberculosis':
            _, results_file = amr_utility.name_utility.GETsave_name_final(fscore,species, f_kma, f_phylotree, '')
            if f_kma:
                results_file='./PhenotypeSeeker_Nov08/'+results_file
            elif f_phylotree:
                results_file='./PhenotypeSeeker_tree/'+results_file
            else:
                results_file='./PhenotypeSeeker_random/'+results_file
            results=pd.read_csv(results_file + '_SummeryBenchmarking_std.txt', header=0, index_col=0,sep="\t")
            score=results.loc[:,fscore].to_list()
            # else:
            #
            #     score=np.empty((len(antibiotics)))
            #     score[:] = np.NaN
            #     score=score.tolist()

        if tool=='Kover':

            _, results_file = amr_utility.name_utility.GETsave_name_final(fscore,species, f_kma, f_phylotree, '')
            if f_kma:
                results_file='./kover/'+results_file
            elif f_phylotree:
                results_file='./kover_tree/'+results_file
            else:
                results_file='./kover_random/'+results_file
            results=pd.read_csv(results_file + '_SummeryBenchmarking_std.txt', header=0, index_col=0,sep="\t")
            score=results.loc[:,fscore].to_list()

        if tool=='Baseline (Majority)':

            results_file,_ = amr_utility.name_utility.GETsave_name_final(fscore,species, f_kma, f_phylotree, 'majority')
            if f_kma:
                results_file='./majority/'+results_file
            elif f_phylotree:
                results_file='./majority/'+results_file
            else:
                results_file='./majority/'+results_file

            results=pd.read_csv(results_file + '_std.txt', header=0, index_col=0,sep="\t")

            score=results.loc[:,fscore].to_list()



        data.append(score)
    return data

def extract_multi_model_summary():
    '''Get antibiotic list that is shared by mutiple species.'''
    data = pd.read_csv('metadata/loose_Species_antibiotic_FineQuality.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    #gather all the possible anti
    data_sub=data['modelling antibiotics'].apply(literal_eval)
    All_anti=np.concatenate(data_sub)
    All_anti=list(set(All_anti))
    All_anti.sort()
    summary=pd.DataFrame(index=data.index, columns=All_anti)  # initialize for visualization
    # print(summary)
    for i in All_anti:
        summary[i] =data_sub.apply(lambda x: 1 if i in x else 0)
    # print(summary.columns,summary.columns.size)
    summary = summary.loc[:, (summary.sum() >1)]
    summary = summary[(summary.T != 0).any()]#drops rows(bacteria) where all zero
    return summary

def adjust_lable(axs_,antibiotics,anti_share,colors,species):
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
        if species not in ['Klebsiella pneumoniae']:
            for label, angle in zip(axs_.get_xticklabels(), angles):
                # label.set_rotation(90-angle*(365/n_lable))
                # label.set_rotation_mode("anchor")
                if species in ['Klebsiella pneumoniae']:
                    angle=angle+22
                elif species in ['Escherichia coli','Salmonella enterica' ]:
                    angle=angle+20
                else:
                    angle=angle+10
                # if i>n_lable/2:
                #     angle=angle-10
                # else:
                #     angle=angle-10
                x,y = label.get_position()
                # print(label.get_text())
                # print('---')
                lab = axs_.text(x,y, label.get_text(), transform=label.get_transform(),
                              ha=label.get_ha(), va=label.get_va(),size=20)

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
                # print( label.get_text())
                # print('---')
                if label.get_text() in anti_share:
                    index_color=anti_share.index(label.get_text())
                    # if label.get_text() in ['trimethoprim/\nsulfamethoxazole']:
                    #     lab.set_bbox(dict(facecolor=colors[index_color], alpha=0.6, edgecolor='white',zorder=1))
                    # # elif ['amoxicillin/\nclavulanic acid']:
                    # #     lab.set_bbox(dict(facecolor=colors[index_color], alpha=1, edgecolor='white'))
                    # else:
                    lab.set_bbox(dict(facecolor=colors[index_color], alpha=0.6, edgecolor='white'))
                else:

                    lab.set_bbox(dict(facecolor='silver', alpha=0.6, edgecolor='white'))
                lab.set_rotation(angle)
                i+=1
                lab.set_rotation_mode("anchor")
                labels.append(lab)
            axs_.set_xticklabels([])
        else:
            target=axs_.get_xticklabels()
            target.append(target.pop(0))
            angles=angles.tolist()
            angles.append(angles.pop(0))
            for label, angle in zip(target, angles):

                if species in ['Klebsiella pneumoniae']:
                    angle=angle+22
                elif species in ['Escherichia coli','Salmonella enterica' ]:
                    angle=angle+20
                else:
                    angle=angle+10
                # if i>n_lable/2:
                #     angle=angle-10
                # else:
                #     angle=angle-10
                x,y = label.get_position()
                # print(label.get_text())
                # print('---')
                lab = axs_.text(x,y, label.get_text(), transform=label.get_transform(),
                              ha=label.get_ha(), va=label.get_va(),size=20)

                if label.get_text() in anti_share:
                    index_color=anti_share.index(label.get_text())
                    # if label.get_text() in ['trimethoprim/\nsulfamethoxazole']:
                    #     lab.set_bbox(dict(facecolor=colors[index_color], alpha=0.6, edgecolor='white',zorder=1))
                    # # elif ['amoxicillin/\nclavulanic acid']:
                    # #     lab.set_bbox(dict(facecolor=colors[index_color], alpha=1, edgecolor='white'))
                    # else:
                    lab.set_bbox(dict(facecolor=colors[index_color], alpha=0.6, edgecolor='white'))
                else:

                    lab.set_bbox(dict(facecolor='silver', alpha=0.6, edgecolor='white'))
                lab.set_rotation(angle)
                i+=1
                lab.set_rotation_mode("anchor")
                labels.append(lab)
            axs_.set_xticklabels([])
        # plt.setp(axs_.get_xticklabels(), backgroundcolor="limegreen")
def adjust_lable_bar(axs_,antibiotics,anti_share,colors):
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
            angle=angle-10
        else:
            angle=angle-10
        x,y = label.get_position()
        # print(label.get_text())
        # print('---')
        lab = axs_.text(x,y, label.get_text(), transform=label.get_transform(),
                      ha=label.get_ha(), va=label.get_va(),size=20)

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
        if label.get_text().split('\n')[1] in anti_share:

            index_color=anti_share.index(label.get_text().split('\n')[1])
            lab.set_bbox(dict(facecolor=colors[index_color], alpha=0.6, edgecolor='white'))
        else:
            lab.set_bbox(dict(facecolor='silver', alpha=0.6, edgecolor='white'))
        lab.set_rotation(angle)
        i+=1
        lab.set_rotation_mode("anchor")
        labels.append(lab)
    axs_.set_xticklabels([])

def radar_factory(num_vars,name_sub, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = name_sub

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)


        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)


                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def draw(level,s, fscore, cv_number, f_all):
    fig, axs = plt.subplots(9,3,figsize=(25, 25*2.8))
    # fig.subplots_adjust(top=0.88)
    lim_pad=4.5
    lim_w,lim_h,lim_t,lim_b=0.4,0.41,0.9608,0.01
    [axi.set_axis_off() for axi in axs.ravel()[0:26]]
    plt.tight_layout(pad=lim_pad)

    fig.text(0.001, 0.972, 'A', fontsize=42,weight='bold')
    fig.text(0.001, 0.65, 'B', fontsize=42,weight='bold')
    fig.text(0.001, 0.32, 'C', fontsize=42,weight='bold')

    fig.subplots_adjust(wspace=lim_w, hspace=lim_h, top=lim_t, bottom=lim_b)
    i=1
    f_phylotree=False
    f_kma=False
    ComBySpecies(level,s, fscore, cv_number, f_phylotree, f_kma,f_all,axs,fig,i)
    i=10
    f_phylotree=True
    f_kma=False
    ComBySpecies(level,s, fscore, cv_number, f_phylotree, f_kma,f_all,axs,fig,i)
    i=19
    f_phylotree=False
    f_kma=True
    ComBySpecies(level,s, fscore, cv_number, f_phylotree, f_kma,f_all,axs,fig,i)


    im = plt.imread('src/legend.png')
    newax = fig.add_axes([0.84,0.84,0.16,0.16], anchor='NE', zorder=-1)
    newax.imshow(im)
    newax.axis('off')
    fig.savefig('log/results/result_STD.png')

def ComBySpecies(level,s, fscore, cv_number, f_phylotree, f_kma,f_all,axs,fig,i):
    # if fscore=='f1_macro' or fscore=='accuracy':
    #     tool_list=['Point-/ResFinder', 'Neural networks', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover','Baseline (Majority)']
    #
    # else:
    tool_list=['Point-/ResFinder', 'Neural networks', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover','Baseline (Majority)']
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

    '''
    # theta = radar_factory(9, frame='circle')
    # fig, axs = plt.subplots(figsize=(20, 25), nrows=4, ncols=3)
    fig, axs = plt.subplots(3,3,figsize=(20, 25))
    # fig.subplots_adjust(top=0.88)
    lim_pad=4.5
    lim_w,lim_h,lim_t,lim_b=0.25,0,0.94,-0.03

    plt.tight_layout(pad=lim_pad)

    fig.subplots_adjust(wspace=lim_w, hspace=lim_h, top=lim_t, bottom=lim_b)
    '''
    labels = tool_list
    # fig.suptitle(title,size=20, weight='bold')
    # [axi.set_axis_off() for axi in axs.ravel()[0:9]] #remove outer frame
    blue=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
    orange=(1.0, 0.4980392156862745, 0.054901960784313725)
    green= (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
    purp=(0.5803921568627451, 0.403921568627451, 0.7411764705882353)
    red=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392)
    red='#ef3b2c'
    brown=(0.5490196078431373, 0.33725490196078434, 0.29411764705882354)
    # if fscore=='f1_macro' or fscore=='accuracy':
    colors = [blue,"orange",  purp , green , red, '#653700']# #ffd343brown
        # colors_std=[ brown,  purp , green , red, "black"]
    # else:
    #     colors = [blue,brown, purp , green , red]
    #     colors_std=[brown, purp , green , red]
    colors_anti=['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#bc80bd','#ccebc5',\
                    '#ffed6f','#836953','#ff9899','#ff694f','#1f78b4','#33a02c','#ff7f00','#a6cee3','#77dd77','#f6e8c3']##b39eb5
    # colors=sns.color_palette(None,7)
    # for ax in axs[0, :]:
    #     ax.remove()
    # summerize antibiotis of all species
    anti_summary=extract_multi_model_summary()
    anti_share=anti_summary.columns.to_list()
    # print(anti_share)

    for i_anti in anti_share:
        if '/' in i_anti:
            posi=i_anti.find('/')
            _i_anti=i_anti[:(posi+1)] + '\n' + i_anti[(posi+1):]
            anti_share=[_i_anti if x==i_anti else x for x in anti_share]

    # -----------------------------------------------------------------------------------------------
    # 1. ploting radar graphs
    # ------------------------------------------------------------------------------------------------
    # i=0
    # i=1
    for species, antibiotics_selected in zip(df_species_radar, antibiotics_radar):

        antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)

        #---------------------------------------------------
        # -------------------Std of scores--------------
        # axs_std = fig.add_subplot(3,3,i,polar= 'spine')#, frame_on=False
        theta = radar_factory(len(antibiotics),'radar'+species, frame='polygon')
        axs_std = plt.subplot(9,3,i, projection='radar'+species)

        data = combine_data_std(species,antibiotics,fscore, f_phylotree, f_kma,tool_list)
        # print(data)
        spoke_labels = antibiotics#antibiotics
        #if too long label, make it 2 lines.
        for i_anti in spoke_labels:
            if '/' in i_anti:
                posi=i_anti.find('/')
                _i_anti=i_anti[:(posi+1)] + '\n' + i_anti[(posi+1):]
                spoke_labels=[_i_anti if x==i_anti else x for x in spoke_labels]

        species_title=(species[0] +". "+ species.split(' ')[1] )
        axs_std.set_title(species_title, weight='bold',style='italic', size=28, position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center',pad=60)
        p_radar=[]

        for d, color in zip(data, colors):
            # if color=="black" and fscore=='f1_negative':
            #     print('??')
            #     p_ =axs_std.plot(theta, d,  'o-', markersize=4,color=color,dashes=[2,2],linewidth=2,alpha=.3)
            # else:
            p_ =axs_std.plot(theta, d,  'o-', markersize=4,color=color,dashes=[2,2],linewidth=2 )
            p_radar.append(p_)
            # ax.fill(theta, d, facecolor=color, alpha=0.25)
        # Circle((0.5, 0.5), 0.5)
        if species=='Klebsiella pneumoniae' and f_kma==True and fscore=='f1_negative':
            axs_std.set_rgrids([-1,-0.5,0, 0.2,  0.4])
            axs_std.set(ylim=(-1, 0.4))
            plt.yticks([-1,-0.5,0, 0.2,  0.4],size=16)
        elif f_kma==True or f_phylotree==True:
            axs_std.set_rgrids([-1,-0.5,0, 0.2, 0.4])
            axs_std.set(ylim=(-1,  0.4))
            plt.yticks([-1,-0.5,0, 0.2,  0.4],size=16)
        else:
            axs_std.set_rgrids([-1,-0.5,0, 0.2,  0.4])
            axs_std.set(ylim=(-1,  0.4))
            plt.yticks([-1,-0.5,0, 0.2, 0.4],size=16)
        plt.grid(color='white', linestyle='-', linewidth=0.7)
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
            leg=axs_std.legend(antibiotics,  labels= labels, ncol=3, loc=(0.2,1.27),fontsize=28, markerscale=4)
            for line in leg.get_lines():
                line.set_linewidth(5.0)
        # axs_std.spines["start"].set_color("white")
        # axs_std.spines["polar"].set_color("white")
        i+=1
        # Adjust tick label positions ------------------------------------
        adjust_lable(axs_std,antibiotics,anti_share,colors_anti,species)


        pos1=axs_std.get_position()
        if species=='Klebsiella pneumoniae' and f_kma==False and f_phylotree==False:
            axs_ = fig.add_axes([pos1.x0+0.0315,pos1.y0+0.0111,pos1.width / 1.41,pos1.height / 1.41], projection= 'radar'+species)
        elif f_kma:
            axs_ = fig.add_axes([pos1.x0+0.0315,pos1.y0+0.0111,pos1.width / 1.41,pos1.height / 1.41], projection= 'radar'+species)
            # axs_ = fig.add_axes([pos1.x0+0.032,pos1.y0+0.027,pos1.width / 1.34,pos1.height / 1.34], projection= 'radar'+species)
        else:
            axs_ = fig.add_axes([pos1.x0+0.0315,pos1.y0+0.0111,pos1.width / 1.41,pos1.height / 1.41], projection= 'radar'+species)
            # axs_ = fig.add_axes([pos1.x0+0.029,pos1.y0+0.024,pos1.width / 1.29,pos1.height / 1.29], projection= 'radar'+species)
        #---------------------------------------------------
        # -------------------Mean of scores--------------
        # axs_ = fig.add_subplot(4,3,i,polar= 'spine')#
        data = combine_data(species,antibiotics,fscore, f_phylotree, f_kma,tool_list)
        # print(data)
        spoke_labels = antibiotics#antibiotics
        #if too long label, make it 2 lines.
        for i_anti in spoke_labels:
            if '/' in i_anti:
                posi=i_anti.find('/')
                _i_anti=i_anti[:(posi+1)] + '\n' + i_anti[(posi+1):]
                spoke_labels=[_i_anti if x==i_anti else x for x in spoke_labels]

        # theta = np.linspace(0, 2*np.pi, len(antibiotics), endpoint=False)

        # Plot the four cases from the example data on separate axes
        # for ax, (title, case_data) in zip(axs.flat, data):
        axs_.set_rgrids([0,0.2,  0.4, 0.6, 0.8],size=18)
        # axs_.set_title(species, weight='bold',style='italic', size=22, position=(0.5, 1.1),
        #              horizontalalignment='center', verticalalignment='center',pad=60)
        # p_radar=[]
        for d, color in zip(data, colors):
            p_ =axs_.plot(theta, d,  'o-', markersize=4,color=color,dashes=[6, 2],linewidth=2)

        # Circle((0.5, 0.5), 0.5)
        # axs_.set(ylim=(0.4, 1.0))
        axs_.set_ylim(ymin=0)
        # plt.yticks([ 0,0.2,0.4, 0.6, 0.8,0.1],size=18)
        plt.grid(color='grey', linestyle='--', linewidth=0.7)

        axs_.set(xticklabels=[])
        axs_.set(xlabel=None)
        axs_.tick_params(axis='x',bottom=False)
        # fig.savefig('log/results/'+'kma_'+str(f_kma)+'_tree_'+str(f_phylotree)+'_'+fscore+'_STD.png')


    # ----------------------------------------
    # =======================================
    #a combination of the rest 3 species
    # data_combine=[]
    # Data=[ ]
    data_com=pd.DataFrame(np.zeros(len(tool_list)))
    for species, antibiotics_selected in zip(df_species_bar, antibiotics_bar):
        antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
        # plt.style.use('ggplot')
        # axs_ = fig.add_subplot(4,3,i)#Campylobacter jejuni
        data=combine_data_std(species,antibiotics,fscore, f_phylotree, f_kma,tool_list)
        data_df=pd.DataFrame(data)
        data_com=pd.concat([data_com, data_df], axis=1, ignore_index=True)

    data_com=data_com.drop(columns=[0])
    # print(data_com)
    data_com=data_com.values
    # print(data_com)
    # antibiotics_com=['Campylobacter jejuni\ntetracycline','Enterococcus faecium\nvancomycin','Neisseria gonorrhoeae\nazithromycin',	'Neisseria gonorrhoeae\ncefixime']
    antibiotics_com=['C. jejuni\ntetracycline','E. faecium\nvancomycin','N. gonorrhoeae\nazithromycin','N. gonorrhoeae\ncefixime']

    theta = radar_factory(4,'radar_com', frame='polygon')
    axs_std = plt.subplot(9,3,i, projection='radar_com')
    for d, color in zip(data_com, colors):
        # print(d)
        p_ =axs_std.plot(theta, d,  'o-', markersize=4,color=color,dashes=[2,2],linewidth=2)
        # p_radar.append(p_)
    axs_std.set_rgrids([-1,-0.5,0, 0.2, 0.4])
    axs_std.set(ylim=(-1, 0.4))
    plt.yticks([-1,-0.5,0, 0.2, 0.4],size=16)
    plt.grid(color='white', linestyle='-', linewidth=0.7)
    axs_std._gen_axes_spines()
    axs_std.set_thetagrids(np.degrees(theta), antibiotics_com)
    axs_std.set_facecolor('#d9d9d9')
    axs_std.tick_params(axis='x', which='major', color='grey',grid_linestyle='dashed', pad=0,zorder=3)#labelsize=16,
    # Adjust tick label positions ------------------------------------
    adjust_lable_bar(axs_std,antibiotics_com,anti_share,colors_anti)


    pos1=axs_std.get_position()
    axs_ = fig.add_axes([pos1.x0+0.0315,pos1.y0+0.0111,pos1.width / 1.41,pos1.height / 1.41], projection= 'radar_com')

    #---------------------------------------------------
    # -------------------Mean of scores--------------
    # axs_ = fig.add_subplot(4,3,i,polar= 'spine')#
    data_com=pd.DataFrame(np.zeros(len(tool_list)))
    for species, antibiotics_selected in zip(df_species_bar, antibiotics_bar):
        antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
        # plt.style.use('ggplot')
        # axs_ = fig.add_subplot(4,3,i)#Campylobacter jejuni
        data=combine_data(species,antibiotics,fscore, f_phylotree, f_kma,tool_list)
        data_df=pd.DataFrame(data)
        data_com=pd.concat([data_com, data_df], axis=1, ignore_index=True)

    data_com=data_com.drop(columns=[0])
    data_com=data_com.values

    axs_.set_rgrids([0,0.2,  0.4, 0.6, 0.8],size=18)
    # axs_.set_title(species, weight='bold',style='italic', size=22, position=(0.5, 1.1),
    #              horizontalalignment='center', verticalalignment='center',pad=60)
    # p_radar=[]
    for d, color in zip(data_com, colors):
        p_ =axs_.plot(theta, d,  'o-', markersize=4,color=color,dashes=[6, 2],linewidth=2)

    axs_.set_ylim(ymin=0)
    plt.grid(color='grey', linestyle='--', linewidth=0.7)
    axs_.set(xticklabels=[])
    axs_.set(xlabel=None)
    axs_.tick_params(axis='x',bottom=False)
    axs_std.set_title('C. jejuni\nE. faecium\nN. gonorrhoeae', weight='bold',style='italic', size=22, position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center',pad=75)


    # im = plt.imread('src/legend.png')
    # newax = fig.add_axes([0.84,0.84,0.16,0.16], anchor='NE', zorder=-1)
    # newax.imshow(im)
    # newax.axis('off')

    # fig.savefig('log/results/'+'kma_'+str(f_kma)+'_tree_'+str(f_phylotree)+'_'+fscore+'_STD.png')
