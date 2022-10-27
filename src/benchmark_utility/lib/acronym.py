#!/usr/bin/python
import sys,os
# sys.path.append('../')
sys.path.insert(0, os.getcwd())
import pickle,ast,json
import pandas as pd
from src.amr_utility import load_data,name_utility
'''
Summarize all antibioics w.r.t. acronym. NOTE: in the dic there are some antibioitics not included in our benchmarking study.
Note: only for developers' use. No use in generating benchmarking results.
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
        antibiotics, ID, Y =  load_data.extract_info(species, False, level)
        antibioticsAll=antibioticsAll+antibiotics

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
    # with open('./src/AntiAcronym_dict.pkl', 'wb') as f:
    #     pickle.dump(map_acr, f)

    with open('./data/AntiAcronym_dict.json', 'w') as f:
        json.dump(map_acr, f)

    # with open('saved_dictionary.pkl', 'rb') as f:
    #     loaded_dict = pickle.load(f)
    # # antibioticsAll_acro=[]
    # for anti in antibioticsAll:
    #     antibioticsAll_acro.append(map_acr[anti])



def summarize(level):
    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    print(data)
    antibiotics_list=[]
    for species,antibiotics in zip(df_species, antibiotics):
        antibiotics, ID, Y =  load_data.extract_info(species, False, level)
        antibiotics_list=antibiotics_list+antibiotics
    antibiotics_list = list(dict.fromkeys(antibiotics_list))
    print(antibiotics_list)
    #add acronym
    with open('./data/AntiAcronym_dict.json') as f:
        map_acr = json.load(f)
    acr_list= [map_acr[x] for x in antibiotics_list]
    print(antibiotics_list)

    table=pd.DataFrame(data=acr_list, columns=['Acronym'],index=antibiotics_list)
    table.to_csv('./data/acronym.csv')





