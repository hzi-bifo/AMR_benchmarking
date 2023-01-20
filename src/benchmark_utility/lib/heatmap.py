import sys,os,json
# sys.path.append('../')
sys.path.insert(0, os.getcwd())
from src.amr_utility import name_utility,file_utility
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from src.benchmark_utility.lib.CombineResults import combine_data

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def heatmap(f_folds,data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    # if f_folds:
    #     cbar = ax.figure.colorbar(im, ax=ax, location='top',shrink=0.6,**cbar_kw)
    #     cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    if f_folds:
        # Show all ticks and label them with the respective list entries.
        ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)
        # ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    else:
        ax.get_yaxis().set_ticks([])
        # ax.get_xaxis().set_ticks([])

    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    # ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(axis='both',which="minor", bottom=False, left=False, top=False, right=False)
    ax.tick_params(axis='x', which='major', labelsize=20)
    ax.tick_params(axis='y', which='major', labelsize=15)

    ## return im, cbar
    return im

def annotate_heatmap(data_anno,im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    # if threshold is not None:
    #     threshold = im.norm(threshold)
    # else:
    #     threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    # if isinstance(valfmt, str):
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)


    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):

            kw.update(color='white')
            if data_anno[i,j]==1:
                text = im.axes.text(j, i, data[i,j], horizontalalignment='center',verticalalignment='center',color='black',fontweight='extra bold')
            else:
                text = im.axes.text(j, i, data[i,j], horizontalalignment='center',verticalalignment='center',color='white')
            texts.append(text)

    return texts



def extract_info(level, fscore,foldset,tool_list,output_path,save_file_name):
    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    species_list=['Escherichia coli','Staphylococcus aureus','Salmonella enterica','Klebsiella pneumoniae','Pseudomonas aeruginosa',
                  'Acinetobacter baumannii','Streptococcus pneumoniae','Campylobacter jejuni',
                  'Enterococcus faecium','Neisseria gonorrhoeae','Mycobacterium tuberculosis']
    data=data.loc[species_list,:]
    df_species = data.index.tolist()
    antibiotics= data['modelling antibiotics'].tolist()

    plt.tight_layout()
    fig, ax= plt.subplots(1,3, figsize=(20, 30))
    (ax1, ax2 , ax3)=ax

    # fig.subplots_adjust(left=0.06,  right=0.9,wspace=-0.6, hspace=0.1, top=0.9, bottom=0.03)
    fig.subplots_adjust(left=0,  right=1.1,wspace=-0.6, top=0.88, bottom=0.02)
    # cmap = plt.get_cmap('viridis')
    cmap = plt.get_cmap('plasma')
    mymap=truncate_colormap(cmap, minval=0.0, maxval=0.9, n=300)
    x_axis_lable=tool_list#species+anti


    j=0
    for eachfold in foldset:

        i=0
        for each_tool in tool_list:
            print(each_tool)
            df_final=pd.DataFrame(columns=['species', 'antibiotics', each_tool])
            for species, antibiotics_selected in zip(df_species, antibiotics):
                species_sub=[species]
                df_macro=combine_data(species_sub,level,fscore,[each_tool],[eachfold],output_path)
                df_macro=df_macro.reset_index()
                df_macro=df_macro.drop(columns=['index'])
                df_macro[each_tool]=df_macro[fscore]
                df_macro=df_macro[['species', 'antibiotics',each_tool]]
                df_final= pd.concat([df_final,df_macro])

            if i==0:
                df_complete=df_final
            else:
                df_complete=pd.merge(df_complete, df_final, how="left", on=['species', 'antibiotics'])
            i+=1


        with open('./data/AntiAcronym_dict.json') as f:
            map_acr = json.load(f)
        anti_acro= [map_acr[x] for x in df_complete['antibiotics'].tolist()]

        df_complete['anti']=anti_acro
        df_complete=df_complete[['species', 'anti']+tool_list]
        df_complete['temp']=df_complete["species"] +  ' ' +df_complete["anti"]
        df_complete['s+a']=df_complete['temp'].apply(lambda x:x[0] +". "+ x.split(' ')[1] +'  '+  x.split(' ')[2])
        df_complete=df_complete[['s+a']+tool_list]
        df_complete=df_complete.set_index('s+a')
        # print(df_complete)

        #------------------------
        i=0
        for each_tool in tool_list:

            df_final=pd.DataFrame(columns=['species', 'antibiotics', each_tool])
            for species, antibiotics_selected in zip(df_species, antibiotics):

                species_sub=[species]
                df_macro=combine_data(species_sub,level,fscore,[each_tool],[eachfold],output_path)
                df_macro=df_macro.reset_index()
                df_macro=df_macro.drop(columns=['index'])

                df_macro[fscore] = df_macro[fscore].astype(str)
                df_macro[each_tool]=df_macro[fscore].apply(lambda x:x.split('±')[0])
                df_macro[each_tool] = df_macro[each_tool] .astype(float)
                df_macro=df_macro[['species', 'antibiotics',each_tool]]
                df_final= pd.concat([df_final,df_macro])

            if i==0:
                df_mean=df_final
            else:
                df_mean=pd.merge(df_mean, df_final, how="left", on=['species', 'antibiotics'])
            i+=1
        # df_mean=df_mean.round(2)
        df_mean=df_mean[tool_list]
        if j==1:
            df_mean=df_mean.reindex(list(range(0,79))).reset_index(drop=True)

        data_mean=df_mean.to_numpy()

        data_mean=np.clip(data_mean, 0.7, 1)
        data_complete=df_complete.to_numpy()

        #--------------------
        # ##annotate the best cell in each row by bold font
        df_winner=pd.read_csv(output_path+ 'Results/other_figures_tables/software_winner_'+fscore+'_'+eachfold+'.csv'
                            , index_col=0, dtype={'genome_id': object}, sep="\t")
        with open('./data/AntiAcronym_dict.json') as f:
            map_acr = json.load(f)
        anti_acro= [map_acr[x] for x in df_winner['antibiotics'].tolist()]
        df_winner['anti']=anti_acro
        df_winner['temp']=df_winner["species"] +  ' ' +df_winner["anti"]
        df_winner['s+a']=df_winner['temp'].apply(lambda x:x[0] +". "+ x.split(' ')[1] +'  '+  x.split(' ')[2])
        df_winner=df_winner.set_index('s+a')
        df_winner=df_winner.reindex(df_complete.index)
        list_winner=df_winner['winner'].tolist()
        # print(df_winner)
        # print(df_complete.index)


        df_anno=pd.DataFrame( index=df_complete.index,columns=tool_list)
        i=0
        for each_combi in list_winner:
            for each_tool in tool_list:
                if each_tool in each_combi:
                    df_anno.loc[df_anno.index[i],each_tool]=1
            i+=1


        data_anno=df_anno.to_numpy()

        if j==0:
            im = heatmap(True,data_mean,  df_complete.index.tolist(),x_axis_lable, ax=ax1,cmap=mymap, cbarlabel="harvest")
            texts = annotate_heatmap(data_anno,im,data=data_complete, valfmt="{x:.1f}",textcolors=("white", "white"))

        elif j==1:
            im = heatmap(False,data_mean,  df_complete.index.tolist(),x_axis_lable, ax=ax2,cmap=mymap, cbarlabel="harvest")
            texts = annotate_heatmap(data_anno,im,data=data_complete, valfmt="{x:.1f}",textcolors=("white", "white"))
        elif j==2:
            im = heatmap(False,data_mean,  df_complete.index.tolist(),x_axis_lable, ax=ax3,cmap=mymap, cbarlabel="harvest")
            texts = annotate_heatmap(data_anno,im,data=data_complete, valfmt="{x:.1f}",textcolors=("white", "white"))
            # cbar = ax3.figure.colorbar(im,ax=ax3 ,location='right',shrink=0.3 )# ax=ax1,
            # cbar.ax.set_ylabel("harvest", rotation=-90, va="bottom")
        j+=1
    ax1.set_aspect(0.38)
    ax2.set_aspect(0.38)
    ax3.set_aspect(0.38)
    cax = fig.add_axes([0.15, .98, 0.35, 0.01])
    cbar=fig.colorbar(im, orientation='horizontal', cax=cax)
    # cbar = fig.colorbar(im,ax=ax[:3], orientation='horizontal')# ax=ax1,,shrink=0.5,pad=0.05
    cbar.ax.set_ylabel("F1-macro", rotation=0,labelpad=8.0,loc='top',fontsize=20)
    cbar.ax.tick_params(axis='x', which='major', labelsize=20)

    fig.text(0.11, 0.97, '0 ~', fontsize=20 )
    fig.text(0.2, 0.95, 'A.  Random folds', fontsize=23,weight='bold')
    fig.text(0.42, 0.95, 'B.  Phylogeny-aware folds', fontsize=23,weight='bold')
    fig.text(0.7, 0.95, 'C.  Homology-aware folds', fontsize=23,weight='bold')


    file_utility.make_dir(os.path.dirname(save_file_name))
    fig.savefig(save_file_name+ 'heatmap.png')
    fig.savefig(save_file_name+ 'heatmap.pdf')

