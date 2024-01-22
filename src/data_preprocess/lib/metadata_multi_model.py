#!/usr/bin/python

import pandas as pd
from  src.amr_utility import name_utility
import numpy as np
from ast import literal_eval




# multi-species

def extract_multi_model_summary(level):
    '''
    This function builds the multi-species-antibiotic dataset
    '''
    # check which species' metadata share the same antibiotic.
    # This should be done after the quality control and filter.
    main_meta,main_multi_meta=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    # gather all the possible anti
    data_sub = data['modelling antibiotics'].apply(literal_eval)
    All_anti = np.concatenate(data_sub)
    All_anti = list(set(All_anti))
    All_anti.sort()
    summary = pd.DataFrame(index=data.index, columns=All_anti)  # initialize for visualization

    for i in All_anti:
        summary[i] = data_sub.apply(lambda x: 1 if i in x else 0)
    # select those can be used on multi-species model
    summary = summary.loc[:, (summary.sum() > 1)]
    summary = summary[(summary.T != 0).any()]  # drops rows(bacteria) where all zero
    summary.loc['Total'] = summary.sum()
    summary.to_csv(main_multi_meta, sep="\t")

def extract_multi_model_size(level):
    '''
    This function calculates the data size of the multi-species-antibiotic dataset
    '''
    _,main_multi_meta=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_multi_meta, index_col=0, sep="\t")
    data_size=pd.DataFrame(index=data.index, columns=data.columns)  # initialize for visualization
    for index, row in data.iterrows():
        for column in data.columns:
            cell_value = row[column]
            species=index
            anti=column
            if cell_value==1:

                pheno_summary=pd.read_csv('./data/PATRIC/meta/'+str(level)+'_genomeNumber/log_' + str(species.replace(" ", "_")) + '_pheno_summary' + '.txt', index_col=0,sep="\t")

                genome_count=pheno_summary.at[anti,'Resistant']+pheno_summary.at[anti,'Susceptible']
                data_size.at[species,anti]=genome_count

    print(data_size)
    data_size.to_csv('./data/PATRIC/meta/'+str(level)+'_genomeNumber/multi-species-antibiotic.csv', sep="\t")

