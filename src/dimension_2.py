import sys
import os
sys.path.insert(0, os.path.abspath('../'))
import amr_utility.name_utility
import amr_utility.load_data
import amr_utility.file_utility
import zipfile
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# species_k_cutting|counting.
# Escherichia coli 6 mer 0.5 cutting
# Escherichia coli 6 mer 0.75 cutting
# Escherichia coli 6 mer
# ...
# ...
# Escherichia coli 8 mer 0.5 cutting
# Escherichia coli 8 mer
# ...
# ...
# Escherichia coli 10 mer
# ...
# ...
# Escherichia coli 13 mer
# ...
# ...
# Escherichia coli 31 mer
# ...
# ...

def extract_info(level,s, fscore, cv_number, f_phylotree, f_kma,f_all):
    data = pd.read_csv('metadata/' + str(level) + '_Species_antibiotic_FineQuality.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    if f_all ==False:
        data = data.loc[s, :]

    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()

    amr_utility.file_utility.make_dir('log/results/'+fscore)
    df_plot = pd.DataFrame(columns=['species_k_cutting','counting_Dimension'])
    columns_name=df_plot.columns.tolist()
    df_plot_sub = pd.DataFrame(columns=columns_name)


    # for k in [6,8,10,13,31]:
    for k in [6,8,10,31]:#8,10,31
        if k==6:
            for species, antibiotics in zip(df_species, antibiotics):
                antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
                i_anti=0
                for anti in antibiotics:
                    id=ID[i_anti]
                    i_anti+=1
                    for cutting in [0.5,0.75,'no ']:
                        for canonical in [True]:
                            _, save_name_kmer = amr_utility.name_utility.GETsave_name_kmer(species,canonical, k)
                            save_name_kmer='./patric_Mar/'+save_name_kmer #todo need rename for github,before publication.
                            data_feature = pd.read_hdf(save_name_kmer)
                            data_feature = data_feature.T
                            # print(data_feature.shape)

                            init_feature = np.zeros((len(id), 1), dtype='uint16')
                            data_model_init = pd.DataFrame(init_feature, index=id, columns=['initializer'])
                            # print('data_model_init',data_model_init.shape)
                            data_X = pd.concat([data_model_init, data_feature.reindex(data_model_init.index)], axis=1)
                            data_X = data_X.drop(['initializer'], axis=1)
                            # print('======================')
                            # print('X_train',X_train.shape)
                            # cut kmer counting >255 all to 255.
                            if cutting != 'no ':
                                #only use the genomic information of the training set. So we are evaluating a ready to use model for clinical case.
                                df_quantile = data_X.loc[id,:].to_numpy()
                                quantile = np.quantile(df_quantile, cutting)
                                quantile = int(quantile)

                                data_X[data_X > quantile] = quantile
                                #if a cloumn the same quantile value, then rm. dimension deduction.
                                data_X=data_X.loc[:, (data_X != quantile).any(axis=0)]
                            data_X=data_X.loc[:, (data_X != 0).any(axis=0)] #rm columns padded with only 0. works for k=8.10
                            # print(data_X)

                            dim=len(data_X.columns.tolist())
                            df_plot_sub.loc['s'] = [species+ ' '+str(k)+'-mer',dim]
                            # if canonical==True:
                            #     df_plot_sub.loc['s'] = [species+' canonical '+str(k)+'-mer '+'cutting to '+str(cutting)+'quantile',dim]
                            # else:
                            #     df_plot_sub.loc['s'] = [species+' '+str(k)+'-mer '+'cutting to '+str(cutting)+'quantile',dim]
                            df_plot = df_plot.append(df_plot_sub, sort=False)

        if k==8:
            for species, antibiotics in zip(df_species, antibiotics):
                antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
                i_anti=0
                for anti in antibiotics:
                    id=ID[i_anti]
                    i_anti+=1
                    for cutting in [0.5,'no ']:
                        for canonical in [True]:
                            _, save_name_kmer = amr_utility.name_utility.GETsave_name_kmer(species,canonical, k)
                            save_name_kmer='./patric_Mar/'+save_name_kmer #todo need rename for github,before publication.
                            data_feature = pd.read_hdf(save_name_kmer)
                            data_feature = data_feature.T
                            # print(data_feature.shape)

                            init_feature = np.zeros((len(id), 1), dtype='uint16')
                            data_model_init = pd.DataFrame(init_feature, index=id, columns=['initializer'])
                            # print('data_model_init',data_model_init.shape)
                            data_X = pd.concat([data_model_init, data_feature.reindex(data_model_init.index)], axis=1)
                            data_X = data_X.drop(['initializer'], axis=1)
                            # print('======================')
                            # print('X_train',X_train.shape)
                            # cut kmer counting >255 all to 255.
                            if cutting != 'no ':
                                #only use the genomic information of the training set. So we are evaluating a ready to use model for clinical case.
                                df_quantile = data_X.loc[id,:].to_numpy()
                                quantile = np.quantile(df_quantile, cutting)
                                quantile = int(quantile)

                                data_X[data_X > quantile] = quantile
                                #if a cloumn the same quantile value, then rm. dimension deduction.
                                data_X=data_X.loc[:, (data_X != quantile).any(axis=0)]
                            data_X=data_X.loc[:, (data_X != 0).any(axis=0)] #rm columns padded with only 0. works for k=8.10
                            # print(data_X)

                            dim=len(data_X.columns.tolist())
                            df_plot_sub.loc['s'] = [species+ ' '+str(k)+'-mer',dim]
                            # if canonical==True:
                            #     df_plot_sub.loc['s'] = [species+' canonical '+str(k)+'-mer '+'cutting to '+str(cutting)+'quantile',dim]
                            # else:
                            #     df_plot_sub.loc['s'] = [species+' '+str(k)+'-mer '+'cutting to '+str(cutting)+'quantile',dim]
                            df_plot = df_plot.append(df_plot_sub, sort=False)
        if k==10:
            for species, antibiotics in zip(df_species, antibiotics):
                antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
                i_anti=0
                for anti in antibiotics:
                    id=ID[i_anti]
                    i_anti+=1

                    for canonical in [True]:
                        if species=='Mycobacterium tuberculosis' and canonical==False:
                            # kmer_temp, save_name_kmer = amr_utility.name_utility.GETsave_name_kmer(species,canonical, k)
                            kmer_temp='./patric_Mar/kmc/'
                            voca=[]
                            for i in id:

                                with zipfile.ZipFile(kmer_temp+"non_cano10mer.zip") as z:
                                    with z.open('non_cano10mer/merge_'+str(k)+'mers_'+str(i) + '.txt') as myZip:

                                        f=pd.read_csv(myZip,
                                                    names=['combination', str(i)],dtype={'genome_id': object}, sep="\t")
                                        voca_sub=f['combination'].to_list()
                                        voca=voca+voca_sub
                                        voca=list(dict.fromkeys(voca))
                            print(len(voca))
                            dim=len(voca)
                        else:
                            _, save_name_kmer = amr_utility.name_utility.GETsave_name_kmer(species,canonical, k)
                            save_name_kmer='./patric_Mar/'+save_name_kmer #todo need rename for github,before publication.
                            data_feature = pd.read_hdf(save_name_kmer)
                            data_feature = data_feature.T
                            # print(data_feature.shape)

                            init_feature = np.zeros((len(id), 1), dtype='uint16')
                            data_model_init = pd.DataFrame(init_feature, index=id, columns=['initializer'])
                            # print('data_model_init',data_model_init.shape)
                            data_X = pd.concat([data_model_init, data_feature.reindex(data_model_init.index)], axis=1)
                            data_X = data_X.drop(['initializer'], axis=1)

                            data_X=data_X.loc[:, (data_X != 0).any(axis=0)] #rm columns padded with only 0. works for k=8.10
                            # print(data_X)

                            dim=len(data_X.columns.tolist())
                        df_plot_sub.loc['s'] = [species+ ' '+str(k)+'-mer',dim]
                        # if canonical==True:
                        #     df_plot_sub.loc['s'] = [species+' canonical '+str(k)+'-mer',dim]
                        # else:
                        #     df_plot_sub.loc['s'] = [species+' '+str(k)+'-mer',dim]
                        df_plot = df_plot.append(df_plot_sub, sort=False)
        if k==31:
            for species, antibiotics in zip(df_species, antibiotics):
                antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
                i_anti=0
                for anti in antibiotics:
                    id=ID[i_anti]
                    i_anti+=1
                    for canonical in [False]:#todo wait!
                        voca=[]
                        if canonical:
                            kmer_temp='./patric_2022/K-mer_lists/cano31mer'
                        else:

                            kmer_temp='./patric_2022/K-mer_lists/non_cano31mer'
                        for i in id:
                            # with zipfile.ZipFile(kmer_temp+".zip") as z:
                            #     with z.open(kmer_temp.split('/')[-1]+'/merge_'+str(k)+'mers_' + str(i) + '.txt') as myZip:

                            f=pd.read_csv(kmer_temp+'/merge_'+str(k)+'mers_' + str(i) + '.txt.zip',compression='zip',
                                        names=['combination', str(i)],dtype={'genome_id': object}, sep="\t")
                            voca_sub=f['combination'].to_list()
                            voca=voca+voca_sub
                            voca=list(dict.fromkeys(voca))
                        print(len(voca))
                        df_plot_sub.loc['s'] = [species+ ' '+str(k)+'-mer',dim]
                        # if canonical==True:
                        #     df_plot_sub.loc['s'] = [species+' canonical '+str(k)+'-mer',dim]
                        # else:
                        #     df_plot_sub.loc['s'] = [species+' '+str(k)+'-mer',dim]
                        df_plot = df_plot.append(df_plot_sub, sort=False)
    print(df_plot)
    df_plot['counting_Dimension'] = df_plot['counting_Dimension'].astype(int)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize = (10,30))
    ax = sns.violinplot(x='counting_Dimension', y='species_k_cutting', data=df_plot,inner=None, color=".8")
    ax = sns.stripplot(x='counting_Dimension', y='species_k_cutting', data=df_plot)
    plt.axvline(1000)
    trans = ax.get_xaxis_transform()
    plt.text(1000, 1, 'PhenotypeSeeker', transform=trans)
    plt.tight_layout(pad=4)
    # ax = sns.boxplot(x="tip", y="day", data=tips, whis=np.inf)
    fig = ax.get_figure()
    fig.savefig("log/results/kmer_dimensions2.png")
