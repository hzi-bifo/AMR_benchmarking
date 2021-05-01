#!/usr/bin/python
import sys
import os
sys.path.append('../')
# sys.path.insert(0, os.getcwd())
import pandas as pd
import numpy as np

def save_quality(species,level):
    save_quality='quality/GenomeFineQuality_' +str(level)+'_'+ str(species.replace(" ", "_")) + '.txt'
    save_all_quality="quality/"+str(species.replace(" ", "_"))+".csv"
    return save_all_quality,save_quality

def save_name_speciesID(species,f=False):#f: flag for the location of calling this function. Flag for metadata.py
    #All the strains before quality control
    if f ==  True:
        save_name_speciesID = 'model/id_' + str(species.replace(" ", "_")) + '.list'
    else:
        save_name_speciesID='metadata/model/id_' + str(species.replace(" ", "_")) + '.list'
    return save_name_speciesID


def save_name_modelID(level,species,anti,f=False):#f: flag for the location of calling this function. Flag for metadata.py
    if f == True:
        save_name_meta = 'balance/'+str(level)+'/log_' + str(species.replace(" ", "_")) + '_' + \
                         str(anti.translate(str.maketrans({'/': '_', ' ': '_'}))) + '_'
        save_name_modelID = 'model/'+str(level)+'/Data_' + str(species.replace(" ", "_")) + '_' + str(
            anti.translate(str.maketrans({'/': '_', ' ': '_'})))

    else:
        save_name_meta = 'metadata/balance/'+str(level)+'/log_' + str(species.replace(" ", "_")) + '_' + \
                         str(anti.translate(str.maketrans({'/': '_', ' ': '_'}))) + '_'
        save_name_modelID = 'metadata/model/'+str(level)+'/Data_' + str(species.replace(" ", "_")) + '_' + str(
            anti.translate(str.maketrans({'/': '_', ' ': '_'})))

    return save_name_meta,save_name_modelID

def save_name_odh(species,anti,k,m,d):#for odh
    save_name_odh ='log/feature/odh/log_' + str(species.replace(" ", "_"))+'.k{}m{}d{}.hd5'.format(k, m, d)
    save_name_odh_score='../log/validation_results/log_' + str(species.replace(" ", "_"))+ str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'.k{}m{}d{}.hd5'.format(k, m, d)
    return save_name_odh,save_name_odh_score

def save_name_kmer(species,canonical,k):#for kmer
    if canonical == True:
        save_name_kmer = 'log/feature/kmer/cano_' + str(species.replace(" ", "_")) + '_' + str(k) + '_mer.h5'
        save_mame_kmc = 'kmc/cano' + str(k) + 'mer/merge_'+str(k)+'mers_'

    else:
        save_name_kmer = 'log/feature/kmer/non_cano_' + str(species.replace(" ", "_")) + '_' + str(k) + '_mer.h5'
        save_mame_kmc = 'kmc/non_cano' + str(k) + 'mer/merge_'+str(k)+'mers_'

    return save_mame_kmc,save_name_kmer

def save_name_score(species,anti,canonical,k):#for kmer
    if canonical == True:
        save_name_score = 'log/validation_results/cano_' + str(k) + '_mer_' +str(species.replace(" ", "_")) + '_' + str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
    else:
        save_name_score = 'log/validation_results/non_cano_' + str(k) + '_mer_'  + str(species.replace(" ", "_")) + '_' + str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
    return save_name_score



def load_metadata(SpeciesFile):#for metadata.py
    '''
    :param SpeciesFile: species list
    :return: Meta data for each strain, which belongs to the species in the parameter file
    '''
    data = pd.read_csv('PATRIC_genomes_AMR.txt', dtype={'genome_id': object, 'genome_name': object}, sep="\t")
    data['genome_name'] = data['genome_name'].astype(str)

    data['species'] = data.genome_name.apply(lambda x: ' '.join(x.split(' ')[0:2]))
    data = data.loc[:, ("genome_id", 'species', 'antibiotic', 'resistant_phenotype')]
    df_species = pd.read_csv(SpeciesFile, dtype={'genome_id': object}, sep="\t", header=0)
    info_species = df_species['species'].tolist()  # 10 species that should be modelled!
    data = data.loc[data['species'].isin(info_species)]
    data = data.dropna()
    data = data.reset_index(drop=True)  # new index now

    return data, info_species

def name_multi_bench(species,antibiotics,cv,innerCV):


    name_weights='log/temp/' + str(species.replace(" ", "_")) +'/'+\
                 str(antibiotics.translate(str.maketrans({'/': '_', ' ': '_'}))) + '_' + str(cv) + str(innerCV)
    return name_weights
def name_multi_bench_save_name_score(species,antibiotics):
    save_name_score = str(species.replace(" ", "_"))  +'/'+ str(antibiotics.translate(str.maketrans({'/': '_', ' ': '_'})))
    return save_name_score




'''
def old(species,anti,canonical,k):
 
    if canonical == True:

        save_name_score_old = 'log/validation_results/cano_' +str(
            species.replace(" ", "_")) + '_' + str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))

    else:
       save_name_score_old = 'log/validation_results/non_cano_' + str(
            species.replace(" ", "_")) + '_' + str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))

    return save_name_score_old
def GetSaveName(species,anti,canonical,k):

    # if '/' in anti:
    save_name = 'log/meta/log_' + str(species.replace(" ", "_")) + '_' + str(anti.translate(str.maketrans({'/': '_', ' ': '_'}))) + '_'
    save_name_model = 'log/model/Data_' + str(species.replace(" ", "_")) + '_' + str(anti.replace("/", "_"))
    if canonical== True:
        save_name_kmer='log/feature/kmer/cano_'+ str(species.replace(" ", "_")) + '_' + str(k) + '_mer.txt'
        save_name_val= 'log/validation_results/cano_' + str(species.replace(" ", "_")) + '_' + str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
    else:
        save_name_kmer = 'log/feature/kmer/non_cano_' + str(species.replace(" ", "_")) + '_' + str(k) + '_mer.txt'
        save_name_val =  'log/validation_results/non_cano_' + str(species.replace(" ", "_")) + '_'+ str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
    # else:
    #     save_name = 'log/meta/log_' + str(species.replace(" ", "_")) + '_' + str(anti) + '_'
    #     save_name_model = 'log/model/Data_' + str(species.replace(" ", "_")) + '_' + str(anti)
    #     if canonical== True:
    #         save_name_kmer='log/feature/kmer/cano_'+ str(species.replace(" ", "_")) + '_' + str(k) + '_mer.txt'
    #         save_name_val  = 'log/validation_results/cano' + str(species.replace(" ", "_")) + '_' + str(anti)
    #     else:
    #         save_name_kmer = 'log/feature/kmer/non_cano_' + str(species.replace(" ", "_")) + '_' + str(k) + '_mer.txt'
    #         save_name_val = 'log/validation_results/non_cano' + str(species.replace(" ", "_")) + '_' + str(anti)
    return save_name,save_name_model,save_name_kmer,save_name_val
'''

# def GetSaveName(species,anti,canonical,k):
#     '''
#
#       :param species: str
#       :param anti: str
#       :param canonical: Boolean
#       :return: save_name: the metadata w.r.t. species and antibiotic. No use and just check
#       save_name_model: the id and pheotype w.r.t. species and antibiotic.
#       save_kmer_name: the kmer w.r.t. spexies.
#       save_name_val: model validation results.
#       '''
#     save_name_meta = 'log/meta/log_' + str(species.replace(" ", "_")) + '_' + str(anti.translate(str.maketrans({'/': '_', ' ': '_'}))) + '_'
#     save_name_modelID = 'log/model/Data_' + str(species.replace(" ", "_")) + '_' + str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
#
#     if canonical == True:
#
#         save_name_score = 'log/validation_results/cano_' + str(k) + '_mer_' +str(species.replace(" ", "_")) + '_' + str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
#     else:
#
#         save_name_score = 'log/validation_results/non_cano_' + str(k) + '_mer_'  + str(species.replace(" ", "_")) + '_' + str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
#     return save_name_meta,save_name_modelID,save_name_score