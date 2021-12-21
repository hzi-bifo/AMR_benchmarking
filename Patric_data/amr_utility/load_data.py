#!/usr/bin/python
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import pandas as pd
import numpy as np
import ast
import amr_utility.name_utility
import amr_utility.graph_utility


#Note, this script should be called by a script in the same folder of metadata folder



def unique_cols(df):#check uniqueness of the training ,testing data
    a = df.to_numpy() # df.values (pandas<0.24)
    return (a[0] == a).all(0)
def check_balance(data_sub_anti):
    '''
    :return: balance_check: the distribution of R, S phenotype.
    '''
    #todo check
    # This is just for those inbalance data downsampling.
    balance_check = data_sub_anti.groupby(by="resistant_phenotype").count()
    # balance_check.to_csv(save_name + 'balance_check.txt', sep="\t")
    # print('Check phenotype balance.', balance_check)
    # if balance_check.index.shape[0] == 2:  # there is Neisseria gonorrhoeae w.r.t. ceftriaxone, no R pheno.
    balance_ratio = balance_check.iloc[0]['genome_id'] / balance_check.iloc[1]['genome_id']
    # data_sub_anti.to_csv(save_name_modelID + '.txt', sep="\t")
    if balance_ratio > 2 or balance_ratio < 0.5:  # #final selected, need to downsample.
        # if not balance, downsampling
        # print('Downsampling starts.....balance_ratio=', balance_ratio)
        label_down = balance_check.idxmax().to_numpy()[0]
        label_keep = balance_check.idxmin().to_numpy()[0]
        # print('!!!!!!!!!!!!!label_down:', label_down)
        data_draw = data_sub_anti[data_sub_anti['resistant_phenotype'] == label_down]
        data_left = data_sub_anti[data_sub_anti['resistant_phenotype'] != label_down]

        data_drew = data_draw.sample(n=int(1.5 * balance_check.loc[label_keep, 'genome_id']))
        data_sub_anti_downsampling = pd.concat([data_drew, data_left], ignore_index=True, sort=False)
        # print('downsampling',data_sub_anti)
        # check balance again:
        balance_check = data_sub_anti_downsampling.groupby(by="resistant_phenotype").count()
        # print('Check phenotype balance after downsampling.', balance_check)
        # balance_check.to_csv(save_name + 'balance_check.txt', mode='a', sep="\t")
    else:
        data_sub_anti_downsampling=data_sub_anti

    return balance_check,data_sub_anti_downsampling

def model(species,antibiotics,balance,level):
    '''
    antibiotics_selected: antibiotics list for each species
    ID_list: sample name matrix. n_anti* n_SampleName. in the order of model/loose/Data_s_anti
    Y:  n_anti* [pheno]
    '''
    antibiotics_selected = ast.literal_eval(antibiotics)

    print(species)
    print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)

    # for anti in ['mupirocin', 'penicillin', 'rifampin', 'tetracycline', 'vancomycin']:
    ID_list=[]
    Y=[]
    pheno_summary = pd.DataFrame(index=antibiotics_selected, columns=['Resistant', 'Susceptible','Resistant(downsampling)', 'Susceptible(downsampling)'])
    for anti in antibiotics_selected:

        save_name_meta, save_name_modelID = amr_utility.name_utility.GETsave_name_modelID(level,species, anti)


        data_sub_anti = pd.read_csv(save_name_modelID + '.txt', index_col=0, dtype={'genome_id': object,'resistant_phenotype':int}, sep="\t")

        # select genome_id and  resistant_phenotype
        #data_sub_anti = data_sub_anti.loc[:, ('genome_id', 'resistant_phenotype')]
        # print(species,',', anti, '=============' )
        # print(data_sub_anti)
        data_sub_anti = data_sub_anti.drop_duplicates()#should no duplicates. Just in case.


        if balance==True:
            balance_check,data_sub_anti=check_balance(data_sub_anti)
            print('Check phenotype balance after downsampling.', balance_check)

        ID_sub_anti=data_sub_anti.genome_id.to_list()
        # print(ID_sub_anti)
        ID_list.append(ID_sub_anti)

        # pheno = {'Resistant': 1, 'Susceptible': 0, 'S': 0, 'Non-susceptible': 1,'RS':1}
        # data_sub_anti.resistant_phenotype = [pheno[item] for item in data_sub_anti.resistant_phenotype]
        # y=data_sub_anti['resistant_phenotype'].to_numpy()
        y = data_sub_anti['resistant_phenotype']
        y = np.array(y)

        Y.append(y)
    # print(pheno_summary)
    return antibiotics_selected,ID_list,Y

def summary(species,level):
    '''
    A summary of phenotype distribution for each species and antibiotic combination
    '''

    data = pd.read_csv('metadata/' + str(level) + '_Species_antibiotic_FineQuality.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    # for training model on part of the dataset:-------------
    # data=data.loc[['Escherichia coli'],:]
    # print('Species overview \n', data)
    data = data.loc[[species], :]
    # data.at['Mycobacterium tuberculosis', 'modelling antibiotics']=['capreomycin', 'ciprofloxacin']
    # --------------------------------------------------------
    df_species = data.index.tolist()
    # print('Selected species: \n', data)
    antibiotics = data['modelling antibiotics'].tolist()[0]
    # print(antibiotics)

    antibiotics_selected = ast.literal_eval(antibiotics)

    print(species)
    # print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)

    # for anti in ['mupirocin', 'penicillin', 'rifampin', 'tetracycline', 'vancomycin']:
    ID_list=[]
    Y=[]
    pheno_summary = pd.DataFrame(index=antibiotics_selected, columns=['Resistant', 'Susceptible','Resistant(downsampling)', 'Susceptible(downsampling)'])
    for anti in antibiotics_selected:

        save_name_meta, save_name_modelID = amr_utility.name_utility.GETsave_name_modelID(level,species, anti)

        data_sub_anti = pd.read_csv(save_name_modelID + '.txt', index_col=0, dtype={'genome_id': object}, sep="\t")

        # select genome_id and  resistant_phenotype
        #data_sub_anti = data_sub_anti.loc[:, ('genome_id', 'resistant_phenotype')]
        # print(species,',', anti, '=============>> loading' )
        # print(data_sub_anti)
        data_sub_anti = data_sub_anti.drop_duplicates()#should no duplicates. Just in case.

        #Downsampling for inbalance data set
        # print(data_sub_anti.groupby(by="resistant_phenotype").count())
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

        # pheno = {'Resistant': 1, 'Susceptible': 0, 'S': 0, 'Non-susceptible': 1,'RS':1}
        # data_sub_anti.resistant_phenotype = [pheno[item] for item in data_sub_anti.resistant_phenotype]
        y=data_sub_anti['resistant_phenotype'].to_numpy()
        Y.append(y)
    print(pheno_summary)
    pheno_summary.to_csv('metadata/balance/'+str(level)+'/log_' + str(species.replace(" ", "_")) + '_ pheno_summary' + '.txt', sep="\t")
    return antibiotics_selected


def extract_info(s,balance,level):

    data = pd.read_csv('metadata/'+str(level)+'_Species_antibiotic_FineQuality.csv', index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]# drop the species with 0 in column 'number'.
    # for training model on part of the dataset:-------------
    # data=data.loc[['Escherichia coli'],:]
    print('Species overview \n',data)
    data = data.loc[[s], :]
    #data.at['Mycobacterium tuberculosis', 'modelling antibiotics']=['capreomycin', 'ciprofloxacin']
    # --------------------------------------------------------
    df_species = data.index.tolist()
    # print('Selected species: \n',data)
    antibiotics = data['modelling antibiotics'].tolist()
    # print(antibiotics)

    for df_species,antibiotics in zip(df_species, antibiotics):
        antibiotics, ID_list, Y=model(df_species, antibiotics,balance,level)

    return antibiotics,ID_list,Y
