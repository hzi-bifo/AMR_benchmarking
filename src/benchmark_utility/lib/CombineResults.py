#!/usr/bin/python
import sys,os
# sys.path.append('../')
sys.path.insert(0, os.getcwd())
from src.amr_utility import name_utility,load_data
import pandas as pd
import numpy as np




'''[Codes maintaining Note]: 
functions in this file is used by 
./src/benchmark_utility/lib/MAINtable.py
./src/benchmark_utility/lib/AytanAktug/excel_multi_analysis.py 
./src/benchmark_utility/lib/AytanAktug/excel_multi.py
./src/benchmark_utility/lib/table_analysis.py
./src/benchmark_utility/lib/pairbox.py
./src/benchmark_utility/lib/pairbox_majority.py
./src/benchmark_utility/lib/ByAnti_errorbar.py
./src/benchmark_utility/lib/ByAnti_errorbar_each.py
'''


def combine_data_get_score(species,tool,anti,f_phylotree,f_kma,fscore,fscore_format,output_path):



    if tool=='KMA-based Point-/ResFinder': #without folds
        if species not in ['Neisseria gonorrhoeae']:
            results_file=name_utility.GETname_ResfinderResults(species,'resfinder_k',output_path)
            results = pd.read_csv(results_file + '.csv',index_col=0,header=0 ,sep="\t")
            score=results.loc[anti,fscore]
        else:
            score=np.nan
    if tool=='Blastn-based Point-/ResFinder': #without folds
        results_file=name_utility.GETname_ResfinderResults(species,'resfinder_b',output_path)
        results = pd.read_csv(results_file + '.csv',index_col=0 ,header=0,sep="\t")
        score=results.loc[anti,fscore]


    if tool=='Point-/ResFinder':#folds version.
        _, results_file= name_utility.GETname_result('resfinder_folds', species, '',f_kma,f_phylotree,'',output_path)
        results=pd.read_csv(results_file + '_SummaryBenchmarking.txt', header=0, index_col=0,sep="\t")
        score=results.loc[anti,fscore]

    if tool=='Aytan-Aktug':
        learning, epochs,f_fixed_threshold,f_nn_base,f_optimize_score=0.0,0,True,False,'f1_macro'
        results_file =  name_utility.GETname_AAresult('AytanAktug',species,learning, epochs,\
                      f_fixed_threshold,f_nn_base,f_optimize_score,f_kma,f_phylotree,'SSSA',output_path)
        results_file=results_file+'_SummaryBenchmarking.txt'
        results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
        score=results.loc[anti,fscore_format]

    if tool=='Seq2Geno2Pheno':
        if species !='Mycobacterium tuberculosis':#no MT information.
            _, results_file= name_utility.GETname_result('seq2geno', species, fscore,f_kma,f_phylotree,'',output_path)
            results=pd.read_csv(results_file + '_SummaryBenchmarking.txt', header=0, index_col=0,sep="\t")
            score=results.loc[anti,fscore]
        else:
            score=np.nan
    if tool=='PhenotypeSeeker':
        # if species !='Mycobacterium tuberculosis':
        _, results_file= name_utility.GETname_result('phenotypeseeker', species, fscore,f_kma,f_phylotree,'',output_path)
        results=pd.read_csv(results_file + '_SummaryBenchmarking.txt', header=0, index_col=0,sep="\t")
        score=results.loc[anti,fscore]
        # else:
        #     score=np.nan
    if tool=='Kover':

        _, results_file= name_utility.GETname_result('kover', species,fscore,f_kma,f_phylotree,'',output_path)
        results=pd.read_csv(results_file + '_SummaryBenchmarking.txt', header=0, index_col=0,sep="\t")
        score=results.loc[anti,fscore]

    if tool=='ML Baseline (Majority)':
        _, results_file= name_utility.GETname_result('majority', species, '',f_kma,f_phylotree,'',output_path)
        results=pd.read_csv(results_file + '_SummaryBenchmarking.txt', header=0, index_col=0,sep="\t")
        score=results.loc[anti,fscore]

    if tool=='Single-species-antibiotic Aytan-Aktug':
        learning, epochs,f_fixed_threshold,f_nn_base,f_optimize_score=0.0,0,True,False,'f1_macro'
        results_file =  name_utility.GETname_AAresult('AytanAktug',species,learning, epochs,\
                      f_fixed_threshold,f_nn_base,f_optimize_score,f_kma,f_phylotree,'SSSA',output_path)
        results_file=results_file+'_SummaryBenchmarking.txt'
        results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
        score=results.loc[anti,fscore_format]

    if tool=='Single-species multi-antibiotics Aytan-Aktug':
        learning, epochs,f_fixed_threshold,f_nn_base,f_optimize_score=0.0,0,True,False,'f1_macro'
        results_file =  name_utility.GETname_AAresult('AytanAktug',species, learning, epochs,\
                      f_fixed_threshold,f_nn_base,f_optimize_score,f_kma,f_phylotree,'SSMA',output_path)
        results_file=results_file+'_SummaryBenchmarking.txt'

        if species in ['Campylobacter jejuni','Enterococcus faecium']:
            score=np.nan
        else:
            results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
            score=results.loc[anti,fscore]

    if tool=='Single-species-antibiotics default':
        learning, epochs,f_fixed_threshold,f_nn_base,f_optimize_score=0.001, 1000,True,True,'f1_macro'
        results_file =  name_utility.GETname_AAresult('AytanAktug',species,learning, epochs,\
                          f_fixed_threshold,f_nn_base,f_optimize_score,f_kma,f_phylotree,'SSSA',output_path)
        results_file=results_file+'_SummaryBenchmarking.txt'
        results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
        score=results.loc[anti,fscore_format]

    if tool=='Discrete databases multi-species model':
        learning, epochs,f_fixed_threshold,f_nn_base,f_optimize_score=0.0,0,True,False,'f1_macro'
        results_file =  name_utility.GETname_AAresult('AytanAktug','Mt_Se_Sp_Ec_Sa_Kp_Ab_Pa_Cj',learning,\
              epochs,f_fixed_threshold,f_nn_base,f_optimize_score,f_kma,f_phylotree,'MSMA_discrete',output_path)

        results_file=results_file+'_split_discrete_model_'+str(fscore)+'.txt'
        results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
        if species in ['Neisseria gonorrhoeae','Enterococcus faecium']:
            score=np.nan

        else:
            if anti in results.columns:
                score=results.loc[species,anti]
            else:
                score=np.nan

    if tool=='Concatenated databases mixed multi-species model':
        learning, epochs,f_fixed_threshold,f_nn_base,f_optimize_score=0.0,0,True,False,'f1_macro'

        results_file = name_utility.GETname_AAresult('AytanAktug','Mt_Se_Sp_Ec_Sa_Kp_Ab_Pa_Cj',learning,\
                 epochs,f_fixed_threshold,f_nn_base,f_optimize_score,f_kma,f_phylotree,'MSMA_concat_mixedS',output_path)
        results_file=results_file+'_split_discrete_model_'+str(fscore)+'.txt'
        results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
        if species in ['Neisseria gonorrhoeae','Enterococcus faecium']:
            score=np.nan

        else:

            if anti in results.columns:
                score=results.loc[species,anti]
            else:
                score=np.nan

    if tool=='Concatenated databases leave-one-out multi-species model':
        learning, epochs,f_fixed_threshold,f_nn_base,f_optimize_score=0.0,0,True,False,'f1_macro'
        results_file = name_utility.GETname_AAresult('AytanAktug','Mt_Se_Sp_Ec_Sa_Kp_Ab_Pa_Cj',learning,\
                     epochs,f_fixed_threshold,f_nn_base,f_optimize_score,f_kma,f_phylotree,'MSMA_concatLOO',output_path)
        results_file=results_file+ '_'+str(species.replace(" ", "_"))+'_SummaryBenchmarking.txt'

        if species in ['Neisseria gonorrhoeae','Enterococcus faecium']:
            score=np.nan

        else:
            results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
            if anti in results.index:
                score=results.loc[anti, fscore]
            else:
                score=np.nan

    return score

def combine_data(species_list,level,fscore,tool_list,folds,output_path):

    # This function makes a matrix of all tools' results.
    df_plot = pd.DataFrame(columns=[fscore, 'species', 'software','antibiotics','folds'])
    columns_name=df_plot.columns.tolist()
    df_plot_sub = pd.DataFrame(columns=columns_name)


    # -----------------------------------
    for fold in folds:
        if fold=='Phylogeny-aware folds':
            f_phylotree=True
            f_kma=False
            fscore_format= fscore
        elif fold=='Homology-aware folds':
            f_phylotree=False
            f_kma=True
            fscore_format="weighted-"+fscore #only for Aytan-Aktug SSSA
        else:#'Random folds', or 'no folds'
            f_phylotree=False
            f_kma=False
            fscore_format= fscore

        for species in species_list:
            if species !='Mycobacterium tuberculosis' or f_phylotree==False:
                antibiotics, _, _ =  load_data.extract_info(species, False, level)
                for anti in antibiotics:
                    for tool in tool_list:
                        score=combine_data_get_score(species,tool,anti,f_phylotree,f_kma,fscore,fscore_format,output_path)

                        df_plot_sub.loc['s'] = [score,species,tool,anti,fold]#[fscore, 'species', 'software','anti','folds']
                        df_plot = df_plot.append(df_plot_sub, sort=False)
    return df_plot


def combine_data_get_score_meanstd(species,tool,anti,f_phylotree,f_kma,fscore,fscore_format,output_path,flag):



    if tool=='Point-/ResFinder':#folds version.
        _, results_file= name_utility.GETname_result('resfinder_folds', species, '',f_kma,f_phylotree,'',output_path)
        results=pd.read_csv(results_file + '_SummaryBenchmarking'+flag+'.txt', header=0, index_col=0,sep="\t")
        score=results.loc[anti,fscore]

    if tool=='Aytan-Aktug':
        learning, epochs,f_fixed_threshold,f_nn_base,f_optimize_score=0.0,0,True,False,'f1_macro'
        results_file =  name_utility.GETname_AAresult('AytanAktug',species,learning, epochs,\
                      f_fixed_threshold,f_nn_base,f_optimize_score,f_kma,f_phylotree,'SSSA',output_path)
        results_file=results_file+'_SummaryBenchmarking'+flag+'.txt'
        results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
        score=results.loc[anti,fscore_format]

    if tool=='Seq2Geno2Pheno':
        if species !='Mycobacterium tuberculosis':#no MT information.
            _, results_file= name_utility.GETname_result('seq2geno', species, fscore,f_kma,f_phylotree,'',output_path)
            results=pd.read_csv(results_file + '_SummaryBenchmarking'+flag+'.txt', header=0, index_col=0,sep="\t")
            score=results.loc[anti,fscore]
        else:
            score=np.nan
    if tool=='PhenotypeSeeker':
        # if species !='Mycobacterium tuberculosis':
        _, results_file= name_utility.GETname_result('phenotypeseeker', species, fscore,f_kma,f_phylotree,'',output_path)
        results=pd.read_csv(results_file + '_SummaryBenchmarking'+flag+'.txt', header=0, index_col=0,sep="\t")
        score=results.loc[anti,fscore]
        # else:
        #     score=np.nan
    if tool=='Kover':

        _, results_file= name_utility.GETname_result('kover', species,fscore,f_kma,f_phylotree,'',output_path)
        results=pd.read_csv(results_file + '_SummaryBenchmarking'+flag+'.txt', header=0, index_col=0,sep="\t")
        score=results.loc[anti,fscore]

    if tool=='ML Baseline (Majority)':
        _, results_file= name_utility.GETname_result('majority', species, '',f_kma,f_phylotree,'',output_path)
        results=pd.read_csv(results_file + '_SummaryBenchmarking'+flag+'.txt', header=0, index_col=0,sep="\t")
        score=results.loc[anti,fscore]

    if tool=='Single-species-antibiotic Aytan-Aktug':
        learning, epochs,f_fixed_threshold,f_nn_base,f_optimize_score=0.0,0,True,False,'f1_macro'
        results_file =  name_utility.GETname_AAresult('AytanAktug',species,learning, epochs,\
                      f_fixed_threshold,f_nn_base,f_optimize_score,f_kma,f_phylotree,'SSSA',output_path)
        results_file=results_file+'_SummaryBenchmarking'+flag+'.txt'
        results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
        score=results.loc[anti,fscore_format]

    if tool=='Single-species multi-antibiotics Aytan-Aktug':
        learning, epochs,f_fixed_threshold,f_nn_base,f_optimize_score=0.0,0,True,False,'f1_macro'
        results_file =  name_utility.GETname_AAresult('AytanAktug',species, learning, epochs,\
                      f_fixed_threshold,f_nn_base,f_optimize_score,f_kma,f_phylotree,'SSMA',output_path)
        results_file=results_file+'_SummaryBenchmarking'+flag+'.txt'

        if species in ['Campylobacter jejuni','Enterococcus faecium']:
            score=np.nan
        else:
            results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
            score=results.loc[anti,fscore]

    if tool=='Single-species-antibiotics default':
        learning, epochs,f_fixed_threshold,f_nn_base,f_optimize_score=0.001, 1000,True,True,'f1_macro'
        results_file =  name_utility.GETname_AAresult('AytanAktug',species,learning, epochs,\
                          f_fixed_threshold,f_nn_base,f_optimize_score,f_kma,f_phylotree,'SSSA',output_path)
        results_file=results_file+'_SummaryBenchmarking'+flag+'.txt'
        results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
        score=results.loc[anti,fscore_format]

    # if tool=='Discrete databases multi-species model':
    #     learning, epochs,f_fixed_threshold,f_nn_base,f_optimize_score=0.0,0,True,False,'f1_macro'
    #     results_file =  name_utility.GETname_AAresult('AytanAktug','Mt_Se_Sp_Ec_Sa_Kp_Ab_Pa_Cj',learning,\
    #           epochs,f_fixed_threshold,f_nn_base,f_optimize_score,f_kma,f_phylotree,'MSMA_discrete',output_path)
    #
    #     results_file=results_file+'_split_discrete_model_'+str(fscore)+'.txt'
    #     results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
    #     if species in ['Neisseria gonorrhoeae','Enterococcus faecium']:
    #         score=np.nan
    #
    #     else:
    #         if anti in results.columns:
    #             score=results.loc[species,anti]
    #         else:
    #             score=np.nan
    #
    # if tool=='Concatenated databases mixed multi-species model':
    #     learning, epochs,f_fixed_threshold,f_nn_base,f_optimize_score=0.0,0,True,False,'f1_macro'
    #
    #     results_file = name_utility.GETname_AAresult('AytanAktug','Mt_Se_Sp_Ec_Sa_Kp_Ab_Pa_Cj',learning,\
    #              epochs,f_fixed_threshold,f_nn_base,f_optimize_score,f_kma,f_phylotree,'MSMA_concat_mixedS',output_path)
    #     results_file=results_file+'_split_discrete_model_'+str(fscore)+'.txt'
    #     results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
    #     if species in ['Neisseria gonorrhoeae','Enterococcus faecium']:
    #         score=np.nan
    #
    #     else:
    #
    #         if anti in results.columns:
    #             score=results.loc[species,anti]
    #         else:
    #             score=np.nan
    #
    # if tool=='Concatenated databases leave-one-out multi-species model':
    #     learning, epochs,f_fixed_threshold,f_nn_base,f_optimize_score=0.0,0,True,False,'f1_macro'
    #     results_file = name_utility.GETname_AAresult('AytanAktug','Mt_Se_Sp_Ec_Sa_Kp_Ab_Pa_Cj',learning,\
    #                  epochs,f_fixed_threshold,f_nn_base,f_optimize_score,f_kma,f_phylotree,'MSMA_concatLOO',output_path)
    #     results_file=results_file+ '_'+str(species.replace(" ", "_"))+'_SummaryBenchmarking.txt'
    #
    #     if species in ['Neisseria gonorrhoeae','Enterococcus faecium']:
    #         score=np.nan
    #
    #     else:
    #         results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
    #         if anti in results.index:
    #             score=results.loc[anti, fscore]
    #         else:
    #             score=np.nan

    return score

def combine_data_meanstd(species_list,level,fscore,tool_list,folds,output_path,flag):

    # This function makes a matrix of all tools' results.
    df_plot = pd.DataFrame(columns=[fscore, 'species', 'software','antibiotics','folds'])
    columns_name=df_plot.columns.tolist()
    df_plot_sub = pd.DataFrame(columns=columns_name)


    # -----------------------------------
    for fold in folds:
        if fold=='Phylogeny-aware folds':
            f_phylotree=True
            f_kma=False
            fscore_format= fscore
        elif fold=='Homology-aware folds':
            f_phylotree=False
            f_kma=True
            fscore_format="weighted-"+fscore #only for Aytan-Aktug SSSA
        else:#'Random folds', or 'no folds'
            f_phylotree=False
            f_kma=False
            fscore_format= fscore

        for species in species_list:
            if species !='Mycobacterium tuberculosis' or f_phylotree==False:
                antibiotics, _, _ =  load_data.extract_info(species, False, level)
                for anti in antibiotics:
                    for tool in tool_list:
                        score=combine_data_get_score_meanstd(species,tool,anti,f_phylotree,f_kma,fscore,fscore_format,output_path,flag)

                        df_plot_sub.loc['s'] = [score,species,tool,anti,fold]#[fscore, 'species', 'software','anti','folds']
                        df_plot = df_plot.append(df_plot_sub, sort=False)
    return df_plot


def combine_data_ByAnti(species_list,anti,fscore, f_phylotree, f_kma,tool,output_path):
    '''for a specific antibiotic, if there exists a relevant combination with a species, return mean, std, and species list'''

    data_mean=[]
    data_std=[]
    if (f_phylotree==False) and (f_kma==True):
        fscore_format="weighted-"+fscore
    else:
        fscore_format= fscore


    for species in species_list:

        score_mean=combine_data_get_score_meanstd(species,tool,anti,f_phylotree,f_kma,fscore,fscore_format,output_path,'_PLOT')
        data_mean.append(score_mean)
        score_std=combine_data_get_score_meanstd(species,tool,anti,f_phylotree,f_kma,fscore,fscore_format,output_path,'_std')
        data_std.append(score_std)

    return data_mean, data_std
