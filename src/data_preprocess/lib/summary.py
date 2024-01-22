#!/usr/bin/python
import ast
import pandas as pd
from src.amr_utility import name_utility,load_data


'''
A summary of dataset after pre-selection and QC
'''

def summary_genome(level):
    '''count genome numbers'''
    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0,dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    df_species = data.index.tolist()
    Ngenome=[]
    for species in df_species:
        antibiotics, _, _ = load_data.extract_info(species, False, level)
        for anti in antibiotics:
            save_name_modelID=name_utility.GETname_meta(species,anti,level)
            data_sub_anti = pd.read_csv(save_name_modelID + '_pheno.txt', dtype={'genome_id': object}, index_col=0,sep="\t")
            Ngenome=Ngenome+data_sub_anti['genome_id'].to_list()
            Ngenome = list(dict.fromkeys(Ngenome))
    print('Genome numbers:',len(Ngenome))

def summary_pheno(species,level):
    '''
    A summary of phenotype distribution for each species and antibiotic combination
    '''

    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0,dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]
    data = data.loc[[species], :]

    antibiotics = data['modelling antibiotics'].tolist()[0]
    antibiotics_selected = ast.literal_eval(antibiotics)
    ID_list=[]
    Y=[]
    pheno_summary = pd.DataFrame(index=antibiotics_selected, columns=['Resistant', 'Susceptible','Resistant(downsampling)', 'Susceptible(downsampling)'])
    for anti in antibiotics_selected:

        save_name_modelID=name_utility.GETname_meta(species,anti,level)
        data_sub_anti = pd.read_csv(save_name_modelID + '_pheno.txt', index_col=0, dtype={'genome_id': object}, sep="\t")
        data_sub_anti = data_sub_anti.drop_duplicates()#should no duplicates. Just in case.
        pheno = data_sub_anti.groupby(by="resistant_phenotype").count()
        pheno_summary.loc[str(anti), 'Resistant'] = pheno.iloc[1, 0]
        pheno_summary.loc[str(anti), 'Susceptible'] = pheno.iloc[0, 0]
        # if balance==True:
        balance_check,data_sub_anti=check_balance(data_sub_anti)
        # print('Check phenotype balance after downsampling.', balance_check)
        pheno = data_sub_anti.groupby(by="resistant_phenotype").count()
        pheno_summary.loc[str(anti), 'Resistant(downsampling)'] = pheno.iloc[1, 0]
        pheno_summary.loc[str(anti), 'Susceptible(downsampling)'] = pheno.iloc[0, 0]
        ID_sub_anti=data_sub_anti.genome_id
        ID_list.append(ID_sub_anti)

        y=data_sub_anti['resistant_phenotype'].to_numpy()
        Y.append(y)
    print(pheno_summary)
    pheno_summary.to_csv('./data/PATRIC/meta/'+str(level)+'_genomeNumber/log_' + str(species.replace(" ", "_")) + '_pheno_summary' + '.txt', sep="\t")

    return antibiotics_selected


def check_balance(data_sub_anti):
    '''
    This is only for those inbalance data downsampling.
    :return: balance_check: the distribution of R, S phenotype.
    '''


    balance_check = data_sub_anti.groupby(by="resistant_phenotype").count()
    balance_ratio = balance_check.iloc[0]['genome_id'] / balance_check.iloc[1]['genome_id']

    if balance_ratio > 2 or balance_ratio < 0.5:  # #final selected, need to downsample.
        label_down = balance_check.idxmax().to_numpy()[0]
        label_keep = balance_check.idxmin().to_numpy()[0]
        data_draw = data_sub_anti[data_sub_anti['resistant_phenotype'] == label_down]
        data_left = data_sub_anti[data_sub_anti['resistant_phenotype'] != label_down]
        data_drew = data_draw.sample(n=int(1.5 * balance_check.loc[label_keep, 'genome_id']))
        data_sub_anti_downsampling = pd.concat([data_drew, data_left], ignore_index=True, sort=False)
        balance_check = data_sub_anti_downsampling.groupby(by="resistant_phenotype").count()

    else:
        data_sub_anti_downsampling=data_sub_anti

    return balance_check,data_sub_anti_downsampling
