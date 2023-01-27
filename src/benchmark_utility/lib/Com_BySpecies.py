import sys
import os
# sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.getcwd())
from src.amr_utility import name_utility,file_utility,load_data
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from ast import literal_eval
from pylab import *
import json
import pandas as pd


def prepare_data(df,fscore):
    # This function changes data to the dafaframe that can be used directly by seaborn for plotting.
    df_plot = pd.DataFrame(columns=[fscore, 'antibiotics', 'software'])
    anti_list=list(df.columns)
    columns_name=df_plot.columns.tolist()
    df_plot_sub = pd.DataFrame(columns=columns_name)
    for anti in anti_list:
        for tool in list(df.index):
            df_plot_sub.loc['s'] = [df.loc[tool,anti], anti,tool]
            df_plot = df_plot.append(df_plot_sub, sort=False)
    return df_plot

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

def combine_data_get_score(species,tool,antibiotics,f_phylotree,f_kma,fscore,fscore_format,output_path,flag):


    if flag=='_PLOT':
        if tool=='KMA-based Point-/ResFinder' : #without folds
            if species not in ['Neisseria gonorrhoeae']:
                results_file=name_utility.GETname_ResfinderResults(species,'resfinder_k',output_path)
                results = pd.read_csv(results_file + '.csv',index_col=0,header=0 ,sep="\t")
                score=results.loc[:,fscore]
            else:
                score=np.nan
        if tool=='Blastn-based Point-/ResFinder': #without folds
            results_file=name_utility.GETname_ResfinderResults(species,'resfinder_b',output_path)
            results = pd.read_csv(results_file + '.csv',index_col=0 ,header=0,sep="\t")
            score=results.loc[:,fscore]
        if tool=='Point-/ResFinder not valuated by folds' : #without folds, only for radar plot Supplemental File 2 Fig. S4
            if species not in ['Neisseria gonorrhoeae']:
                results_file=name_utility.GETname_ResfinderResults(species,'resfinder_k',output_path)
                results = pd.read_csv(results_file + '.csv',index_col=0,header=0 ,sep="\t")
                score=results.loc[:,fscore]
            else:
                results_file=name_utility.GETname_ResfinderResults(species,'resfinder_b',output_path)
                results = pd.read_csv(results_file + '.csv',index_col=0 ,header=0,sep="\t")
                score=results.loc[:,fscore]
    else:
        score=np.empty((len(antibiotics)))
        score[:] = np.NaN



    if tool in ['Point-/ResFinder','Point-/ResFinder evaluated by folds']:#folds version.
        _, results_file= name_utility.GETname_result('resfinder_folds', species, '',f_kma,f_phylotree,'',output_path)
        results=pd.read_csv(results_file + '_SummaryBenchmarking'+flag+'.txt', header=0, index_col=0,sep="\t")
        score=results.loc[:,fscore]

    if tool=='Aytan-Aktug':
        learning, epochs,f_fixed_threshold,f_nn_base,f_optimize_score=0.0,0,True,False,'f1_macro'
        results_file =  name_utility.GETname_AAresult('AytanAktug',species,learning, epochs,\
                      f_fixed_threshold,f_nn_base,f_optimize_score,f_kma,f_phylotree,'SSSA',output_path)
        results_file=results_file+'_SummaryBenchmarking'+flag+'.txt'
        results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
        score=results.loc[:,fscore_format]

    if tool=='Seq2Geno2Pheno':
        if species !='Mycobacterium tuberculosis':#no MT information.
            _, results_file= name_utility.GETname_result('seq2geno', species, fscore,f_kma,f_phylotree,'',output_path)
            results=pd.read_csv(results_file + '_SummaryBenchmarking'+flag+'.txt', header=0, index_col=0,sep="\t")
            score=results.loc[:,fscore]
        else:
            score=np.empty((len(antibiotics)))
            score[:] = np.NaN

    if tool=='PhenotypeSeeker':

        _, results_file= name_utility.GETname_result('phenotypeseeker', species, fscore,f_kma,f_phylotree,'',output_path)
        results=pd.read_csv(results_file + '_SummaryBenchmarking'+flag+'.txt', header=0, index_col=0,sep="\t")
        score=results.loc[:,fscore]

    if tool=='Kover':

        _, results_file= name_utility.GETname_result('kover', species,fscore,f_kma,f_phylotree,'',output_path)
        results=pd.read_csv(results_file + '_SummaryBenchmarking'+flag+'.txt', header=0, index_col=0,sep="\t")
        score=results.loc[:,fscore]

    if tool=='ML Baseline (Majority)':
        _, results_file= name_utility.GETname_result('majority', species, '',f_kma,f_phylotree,'',output_path)
        results=pd.read_csv(results_file + '_SummaryBenchmarking'+flag+'.txt', header=0, index_col=0,sep="\t")
        score=results.loc[:,fscore]

    return score

def combine_data_mean(species,antibiotics,fscore, f_phylotree, f_kma,tool_list,output_path):
    if (f_phylotree==False) and (f_kma==True):
        fscore_format="weighted-"+fscore
    else:
        fscore_format= fscore
    data = []
    for tool in tool_list:

        score=combine_data_get_score(species,tool,antibiotics,f_phylotree,f_kma,fscore,fscore_format,output_path,'_PLOT')
        score=score.tolist()
        data.append(score)
    return data

def combine_data_std(species,antibiotics,fscore, f_phylotree, f_kma,tool_list,output_path):
    if (f_phylotree==False) and (f_kma==True):
        fscore_format="weighted-"+fscore
    else:
        fscore_format= fscore
    data = []
    for tool in tool_list:
        score=combine_data_get_score(species,tool,antibiotics,f_phylotree,f_kma,fscore,fscore_format,output_path,'_std')
        score=score.tolist()
        data.append(score)
    return data

def extract_multi_model_summary():
    '''Get antibiotic list that is shared by mutiple species.'''
    main_meta,_=name_utility.GETname_main_meta('loose')
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")

    #gather all the possible anti
    data_sub=data['modelling antibiotics'].apply(literal_eval)
    All_anti=np.concatenate(data_sub)
    All_anti=list(set(All_anti))
    All_anti.sort()
    summary=pd.DataFrame(index=data.index, columns=All_anti)  # initialize for visualization

    for i in All_anti:
        summary[i] =data_sub.apply(lambda x: 1 if i in x else 0)

    summary = summary.loc[:, (summary.sum() >1)]
    summary = summary[(summary.T != 0).any()]#drops rows(bacteria) where all zero
    return summary


def adjust_lable_bar(axs_,antibiotics,anti_share,colors,font_size):
    XTICKS = axs_.xaxis.get_major_ticks()
    n_lable=len(antibiotics)
    angles = np.linspace(0,2*np.pi,len(axs_.get_xticklabels())+1)
    angles = np.rad2deg(angles)
    XTICKS[1].set_pad(25)
    XTICKS[3].set_pad(20)
    XTICKS[2].set_pad(-10)
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
        if i==3:
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



def ComBySpecies(tool_list,level,s, fscore, f_phylotree, f_kma,f_all,fig,i,transparent,output_path):

    #### tool_list=['Point-/ResFinder', 'Aytan-Aktug', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover','ML Baseline (Majority)']
    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
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

    file_utility.make_dir('log/results/')


    labels = tool_list
    blue=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
    # orange=(1.0, 0.4980392156862745, 0.054901960784313725)
    green= (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
    purp=(0.5803921568627451, 0.403921568627451, 0.7411764705882353)
    # red=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392)
    red='#ef3b2c'
    # brown=(0.5490196078431373, 0.33725490196078434, 0.29411764705882354)

    colors = [blue,"orange",  purp , green , red, '#653700']# #ffd343brown
    colors_anti=['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#bc80bd','#ccebc5',\
                    '#ffed6f','#836953','#ff9899','#ff694f','#1f78b4','#33a02c','#ff7f00','#a6cee3','#77dd77','#f6e8c3']##b39eb5

    # summerize antibiotis of all species
    anti_summary=extract_multi_model_summary() #antis shared by multiple species
    anti_share=anti_summary.columns.to_list()

    font_size=30
    line_width=5

    # -----------------------------------------------------------------------------------------------
    # 1. ploting radar graphs
    # ------------------------------------------------------------------------------------------------

    for species, antibiotics_selected in zip(df_species_radar, antibiotics_radar):
        print(species)
        antibiotics, ID, Y =load_data.extract_info(species, False, level)

        #---------------------------------------------------
        # -------------------Std of scores--------------

        theta = radar_factory(len(antibiotics),'radar'+species, frame='polygon')
        axs_std = plt.subplot(9,3,i, projection='radar'+species)
        data = combine_data_std(species,antibiotics,fscore, f_phylotree, f_kma,tool_list,output_path)
        # print(data)

        spoke_labels = antibiotics#antibiotics
        #Add acronym
        with open('./data/AntiAcronym_dict.json') as f:
            map_acr = json.load(f)
        spoke_labels= [map_acr[x] for x in spoke_labels]



        species_title=(species[0] +". "+ species.split(' ')[1] )
        axs_std.set_title(species_title, weight='bold',style='italic', size=30, position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center',pad=30)
        p_radar=[]

        for d, color in zip(data, colors):
            p_ =axs_std.plot(theta, d,  'o-', markersize=6,color=color,dashes=[5,2],linewidth=line_width )
            p_radar.append(p_)

        if species=='Klebsiella pneumoniae' and f_kma==True and fscore=='f1_negative':
            axs_std.set_rgrids([-1,-0.5,0, 0.2,  0.4])
            axs_std.set(ylim=(-1, 0.41))
            plt.yticks([-1,-0.5,0, 0.2,  0.4],size=16)
        elif f_kma==True or f_phylotree==True:
            axs_std.set_rgrids([-1,-0.5,0, 0.2, 0.4])
            axs_std.set(ylim=(-1,  0.41))
            plt.yticks([-1,-0.5,0, 0.2,  0.4],size=16)
        else:
            axs_std.set_rgrids([-1,-0.5,0, 0.2,  0.4])
            axs_std.set(ylim=(-1,  0.41))
            plt.yticks([-1,-0.5,0, 0.2, 0.4],size=16)
        plt.grid(color='white', linestyle='-', linewidth=1)

        axs_std._gen_axes_spines()
        axs_std.set_thetagrids(np.degrees(theta), spoke_labels)
        axs_std.set_facecolor('#d9d9d9')
        axs_std.tick_params(axis='x', which='major', color='grey',grid_linestyle='dashed', pad=5,zorder=3)
        #----------------------
        # -----legend---------
        # ------------------
        if i==1:
            leg=axs_std.legend(antibiotics,  labels= labels, ncol=3, loc=(0.4,1.24),fontsize=28, markerscale=4)
            for line in leg.get_lines():
                line.set_linewidth(5.0)
        i+=1
        # Adjust tick label positions ------------------------------------
        axs_std.tick_params(axis='x', which='major', pad=18,labelsize=font_size)
        pos1=axs_std.get_position()
        if species=='Klebsiella pneumoniae' and f_kma==False and f_phylotree==False:
            axs_ = fig.add_axes([pos1.x0+0.03575,pos1.y0+0.0128,pos1.width / 1.44,pos1.height / 1.44], projection= 'radar'+species)
        elif f_kma:
            axs_ = fig.add_axes([pos1.x0+0.03575,pos1.y0+0.0128,pos1.width / 1.44,pos1.height / 1.44], projection= 'radar'+species)

        else:
            axs_ = fig.add_axes([pos1.x0+0.03575,pos1.y0+0.0128,pos1.width / 1.44,pos1.height / 1.44], projection= 'radar'+species)

        #---------------------------------------------------
        # -------------------Mean of scores--------------

        data = combine_data_mean(species,antibiotics,fscore, f_phylotree, f_kma,tool_list,output_path)
        # print(data)


        temp_zorder=0
        for d, color in zip(data, colors):
            if temp_zorder==0:
                p_ =axs_.plot(theta, d,  'o-', markersize=9,color=color,linewidth=line_width,zorder=3,alpha= transparent)#,dashes=[6, 2],
            else:
                p_ =axs_.plot(theta, d,  'o-', markersize=9,color=color,linewidth=line_width, alpha=transparent)#,dashes=[6, 2],
            temp_zorder+=1

        axs_.set_ylim(ymin=-0.05)
        axs_.set_rgrids([0, 0.5, 1 ],size=18)
        axs_.tick_params(axis='y', which='major', pad=15)
        axs_.yaxis.grid(False)
        axs_.xaxis.grid(False)
        plt.grid(axis='y',color='gray', dashes=[3,3], linewidth=2)
        axs_.set(xticklabels=[])
        axs_.set(xlabel=None)
        axs_.tick_params(axis='x',bottom=False)

    # ----------------------------------------
    # =======================================
    #a combination of the rest 3 species

    data_com=pd.DataFrame(np.zeros(len(tool_list)))
    for species, antibiotics_selected in zip(df_species_bar, antibiotics_bar):
        antibiotics, _, _ = load_data.extract_info(species, False, level)

        data=combine_data_std(species,antibiotics,fscore, f_phylotree, f_kma,tool_list,output_path)
        data_df=pd.DataFrame(data)
        data_com=pd.concat([data_com, data_df], axis=1, ignore_index=True)

    data_com=data_com.drop(columns=[0])
    data_com=data_com.values

    rcParams['mathtext.fontset'] = 'custom'
    rcParams['mathtext.it'] = 'stixsans:italic'
    rcParams['mathtext.bf'] = 'stixsans:italic:bold'
    antibiotics_com=['$\mathbf{C. jejuni}$ TE','$\mathbf{E. faecium}$\nVA ','$\mathbf{N. gonorrhoeae}$ AZI','$\mathbf{N. gonorrhoeae}$\nFIX']
    theta = radar_factory(4,'radar_com', frame='polygon')
    axs_std = plt.subplot(9,3,i, projection='radar_com')
    for d, color in zip(data_com, colors):
        p_ =axs_std.plot(theta, d,  'o-', markersize=6,color=color,dashes=[5,2],linewidth=line_width)

    axs_std.set_rgrids([-1,-0.5,0, 0.2, 0.4])
    axs_std.set(ylim=(-1, 0.41))
    plt.yticks([-1,-0.5,0, 0.2, 0.4],size=16)
    plt.grid(color='white', linestyle='-', linewidth=2)
    axs_std._gen_axes_spines()
    axs_std.set_thetagrids(np.degrees(theta), antibiotics_com)
    axs_std.set_facecolor('#d9d9d9')
    axs_std.tick_params(axis='x', which='major', color='grey',grid_linestyle='dashed', pad=5,zorder=3)#labelsize=16,
    # Adjust tick label positions ------------------------------------
    adjust_lable_bar(axs_std,antibiotics_com,anti_share,colors_anti,font_size)

    pos1=axs_std.get_position()
    axs_ = fig.add_axes([pos1.x0+0.03575,pos1.y0+0.0128,pos1.width / 1.44,pos1.height / 1.44], projection= 'radar_com')

    #---------------------------------------------------
    # -------------------Mean of scores--------------

    data_com=pd.DataFrame(np.zeros(len(tool_list)))
    for species, antibiotics_selected in zip(df_species_bar, antibiotics_bar):
        antibiotics, _, _ = load_data.extract_info(species, False, level)
        data=combine_data_mean(species,antibiotics,fscore, f_phylotree, f_kma,tool_list,output_path)
        data_df=pd.DataFrame(data)
        data_com=pd.concat([data_com, data_df], axis=1, ignore_index=True)

    data_com=data_com.drop(columns=[0])
    data_com=data_com.values
    axs_.set_rgrids([0,0.5,1],size=18)

    for d, color in zip(data_com, colors):
        if temp_zorder==0:
            p_ =axs_.plot(theta, d,  'o-', markersize=9,color=color,linewidth=line_width,zorder=3)#,dashes=[6, 2],
        else:
            p_ =axs_.plot(theta, d,  'o-', markersize=9,color=color,linewidth=line_width)#,dashes=[6, 2],



    axs_.set(xticklabels=[])
    axs_.set(xlabel=None)
    axs_.tick_params(axis='x',bottom=False)
    axs_.set_ylim(ymin=-0.05)
    axs_.yaxis.grid(False)
    axs_.xaxis.grid(False)
    plt.grid(axis='y',color='gray', dashes=[3,3], linewidth=2)



def draw(tool_list,level,species, fscore, f_all,transparent,output_path,save_file_name):
    fig, axs = plt.subplots(9,3,figsize=(25, 25*2.8))
    lim_pad=4.5
    lim_w,lim_h,lim_t,lim_b=0.4,0.3,0.9608,0.01
    [axi.set_axis_off() for axi in axs.ravel()[0:27]]
    plt.tight_layout(pad=lim_pad)

    fig.text(0.001, 0.972, 'A', fontsize=42,weight='bold')
    fig.text(0.001, 0.65, 'B', fontsize=42,weight='bold')
    fig.text(0.001, 0.32, 'C', fontsize=42,weight='bold')
    fig.text(0.7, 0.42, 'D', fontsize=42,weight='bold')

    fig.subplots_adjust(wspace=lim_w, hspace=lim_h, top=lim_t, bottom=lim_b)
    i=1
    f_phylotree=False
    f_kma=False
    ComBySpecies(tool_list,level,species, fscore, f_phylotree, f_kma,f_all,fig,i,transparent,output_path)
    i=10
    f_phylotree=True
    f_kma=False
    ComBySpecies(tool_list,level,species, fscore, f_phylotree, f_kma,f_all,fig,i,transparent,output_path)



    i=19
    f_phylotree=False
    f_kma=True
    ComBySpecies(tool_list,level,species, fscore, f_phylotree, f_kma,f_all,fig,i,transparent,output_path)



    im = plt.imread(output_path+'src/benchmark_utility/lib/legend.png')
    newax = fig.add_axes([0.69,0.155,0.25,0.25], anchor='NE', zorder=-1)
    newax.imshow(im)
    newax.axis('off')
    fig.savefig(save_file_name+'.pdf')


