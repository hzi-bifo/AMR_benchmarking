
import pickle,ast
import pandas as pd
import numpy as np
import amr_utility.load_data

'''
Summarize all antibioics w.r.t. acronym. NOTE: in the dic there are some antibioitics not included in our benchmarking study.
'''

def extract_info(level):
    data = pd.read_csv('metadata/' + str(level) + '_Species_antibiotic_FineQuality.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")

    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.

    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()

    # print(antibiotics)
    antibioticsAll=[]
    for species,antibiotics in zip(df_species, antibiotics):
        antibiotics_selected = ast.literal_eval(antibiotics)
        # print(species)
        antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
        antibioticsAll=antibioticsAll+antibiotics
        print()

    # print(antibioticsAll)
    # print(len(antibioticsAll))#81
    antibioticsAll = list(dict.fromkeys(antibioticsAll))
    print(antibioticsAll)
    print(len(antibioticsAll)) #46

    #
    acr=pd.read_csv('./src/acronym', index_col=None, sep="\t")
    acr['Antibiotic']=acr['Antibiotic'].str.lower()
    # print(acr)
    map_acr={}
    for i in acr.index.to_list():

        # if ' ' in acr.iloc[i,0].split('/')[0] == True:
        #     map_acr[acr.iloc[i,1]]=acr.iloc[i,0].split('/')[0][:-1]
        # else:
       map_acr[acr.iloc[i,1]]=acr.iloc[i,0].split('/')[0].replace(" ", "")

    map_acr['capreomycin']='CAP'
    map_acr['ethambutol']='EMB'
    map_acr['ethiomide']='ETH'
    map_acr['ethionamide']='ETO'
    map_acr['isoniazid']='INH'
    map_acr['pyrazinamide']='PZA'
    map_acr['rifampin']='RMP'
    map_acr['sulfisoxazole']='SIX'
    map_acr['beta-lactam']='BL'
    map_acr['penicillin']='PCN'
    map_acr['methicillin']='MET'
    for key, value in map_acr.items():
        print (key,'(',value,')')
    print(map_acr.items())
    with open('./src/AntiAcronym_dict.pkl', 'wb') as f:
        pickle.dump(map_acr, f)

    # with open('saved_dictionary.pkl', 'rb') as f:
    #     loaded_dict = pickle.load(f)
    # # antibioticsAll_acro=[]
    # for anti in antibioticsAll:
    #     antibioticsAll_acro.append(map_acr[anti])

    # antibioticsAll_acro = list(dict.fromkeys(antibioticsAll_acro))
    # print(len(antibioticsAll_acro))


def summarize(level):
    data = pd.read_csv('metadata/' + str(level) + '_Species_antibiotic_FineQuality.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    print(data)
    antibiotics_list=[]
    for species,antibiotics in zip(df_species, antibiotics):
        antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
        antibiotics_list=antibiotics_list+antibiotics
    antibiotics_list = list(dict.fromkeys(antibiotics_list))
    print(antibiotics_list)
    #add acronym
    with open('./src/AntiAcronym_dict.pkl', 'rb') as f:
        map_acr = pickle.load(f)
    acr_list= [map_acr[x] for x in antibiotics_list]
    print(antibiotics_list)

    table=pd.DataFrame(data=acr_list, columns=['Acronym'],index=antibiotics_list)
    table.to_csv('log/results/acronym.csv')





