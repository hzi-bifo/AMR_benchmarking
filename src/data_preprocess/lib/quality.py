#!/usr/bin/python

import os
import pandas as pd
import sys
sys.path.append('../../')
from src.amr_utility import name_utility



def criteria(species, df,level):
    '''
    :param df: (dataframe) ID with quality meta data
    :param level: quality control level
    :return: (dataframe)ID  with quality control applied.
    '''
    if level == 'strict':
        df=df[(df['genome.genome_status'] != 'Plasmid') & (df['genome.genome_quality'] == 'Good') & (
                df['genome.contigs'] <= 100)
       & (df['genome.fine_consistency'] >= 97) & (df['genome.coarse_consistency'] >= 98) & (
                   df['genome.checkm_completeness'] >= 98)
       & (df['genome.checkm_contamination'] <= 2)]
    else:#loose
        df = df[(df['genome.genome_status'] != 'Plasmid') & (df['genome.genome_quality'] == 'Good') & (
                df['genome.contigs'] <= max(100, df['genome.contigs'].quantile(0.75))) & (df['genome.fine_consistency'] >= 97) & (df['genome.coarse_consistency'] >= 98) & (
                (df['genome.checkm_completeness'] >= 98)| (df['genome.checkm_completeness'].isnull())) & ((df['genome.checkm_contamination'] <= 2)|(df['genome.checkm_contamination'].isnull()))]


    # caluculate the mean genome_length
    mean_genome_l = df["genome.genome_length"].mean()
    # filter abs(genome length - mean length) <= mean length/20'''
    df = df[abs(df['genome.genome_length'] - mean_genome_l) <= mean_genome_l / 20]
    if species == 'Pseudomonas aeruginosa':  # Pseudomonas_aeruginosa add on the genomes from S2G2P paper.
        pa_add = pd.read_csv('./data/PATRIC/Pseudomonas_aeruginosa_add.txt', dtype={'genome.genome_id': object}, header=0)
        df = df.append(pa_add, sort=False)
        df = df.drop_duplicates(subset=['genome.genome_id'])
    df = df.reset_index(drop=True)
    return df
def extract_id_quality(temp_path,level):
    '''
    inpout: downloaded quality metadata, saved at the subdirectory: /quality.
    '''

    df=pd.read_csv(temp_path+'list_species_final_bq.txt', dtype={'genome_id': object}, sep="\t", header=0)
    info_species = df['species'].tolist()
    number_All=[]
    number_FineQuality=[]
    for species in info_species:
        save_all_quality,save_quality=name_utility.GETname_quality(species,level)
        df=pd.read_csv(save_all_quality,dtype={'genome.genome_id': object, 'genome.genome_name': object}, sep="\t")
        number_All.append(df.shape[0])
        #=======================
        #Apply criteria
        df=criteria(species, df, level)
        # =========================
        # selected fine quality genome ID
        #=====================
        df.to_csv(save_quality, sep="\t")#'quality/GenomeFineQuality_' + str(species.replace(" ", "_")) + '.txt'
        number_FineQuality.append(df.shape[0])
        #delete duplicates

    # Visualization
    count_quality = pd.DataFrame(list(zip(number_All, number_FineQuality)), index=info_species, columns=['Number of genomes','Number of fine quality genomes'])
    # print('Visualization species with antibiotic selected',count_quality)
    count_species = pd.read_csv(temp_path+'list_species_final_bq.txt', dtype={'genome_id': object}, sep="\t",
                             header=0,index_col='species')
    count_final=pd.concat([count_species, count_quality], axis=1).reindex(count_species.index)# visualization. no selection in this cm.
    # filter Shigella sonnei and Enterococcus faecium, only 25,144
    count_final=count_final[count_final['Number of fine quality genomes']>200]
    count_final.rename(columns={'count': 'Number of genomes with AMR metadata'}, inplace=True)
    count_final.to_csv("./data/PATRIC/meta/fine_quality/"+str(level)+'_list_species_final_quality.csv',sep="\t")#species list.
    # print(count_final)

def unique_cols(df):
    a = df.to_numpy() # df.values (pandas<0.24)
    return (a[0] == a).all(0)
def filter_quality(level,f_balance):
    '''filter quality by each species, and provide infor for each antibiotics
    variable: data_sub: selected genomes w.r.t. each species
    variable: data_sub_anti: slected genomes w.r.t. each species and antibioitc.
    Output: save_name_model :genome_id	resistant_phenotype. w.r.t. ach species and antibioitc, in log/model/
    Output: Species_quality & 'Species_antibiotic_FineQuality.csv': visualization ,selected species and antibioitc.
    '''
    # load in data for the selected 11 species
    #SpeciesFile define the species to be loaded in
    data, info_species = name_utility.load_metadata(SpeciesFile="./data/PATRIC/meta/fine_quality/"+str(level)+'_list_species_final_quality.csv')
    # drop phenotype with 'Intermediate''Not defined'
    data = data[(data.resistant_phenotype != 'Intermediate') & (data.resistant_phenotype != 'Not defined')]
    #=======================================================
    Species_quality=pd.DataFrame(index=info_species, columns=['number','modelling antibiotics'])#initialize for visualization
    #print(Species_quality)
    #for species in ['Pseudomonas aeruginosa']:
    # info_species=['Escherichia coli']#todo delete later
    for species in info_species:

        BAD=[]
        save_all_quality,save_quality=name_utility.GETname_quality(species,level)
        data_sub = data[data['species'] == species]
        #with pd.option_context('display.max_columns', None):
            #print(data_sub)
        # [1]. select the id from genome_list that are also in good quality
        #====================================================================
        df = pd.read_csv(save_quality,dtype={'genome.genome_id': object, 'genome.genome_name': object}, index_col=0,sep="\t")
        id_GoodQuality=df['genome.genome_id']
        data_sub=data_sub[data_sub['genome_id'].isin(id_GoodQuality)]
        #=====================================================================
        # replace amoxicillin-clavulanate'	with 'amoxicillin/clavulanic acid'
        data_sub=data_sub.replace('amoxicillin-clavulanate', 'amoxicillin/clavulanic acid') # two names for the same antibiotic

        # by each antibiotic.
        data_anti = data_sub.groupby(by="antibiotic")['genome_id']
        summary = data_anti.count().to_frame()

        summary = summary[summary['genome_id'] > 200]# select the antibiotic with genome_id.count >200
        select_antibiotic = summary.index.to_list()
        data_sub = data_sub.loc[data_sub['antibiotic'].isin(select_antibiotic)]
        select_antibiotic_final= select_antibiotic.copy()
        for anti in select_antibiotic:
            save_name_modelID=name_utility.GETname_meta(species,anti,level)
            # select genome_id and  resistant_phenotype
            data_sub_anti = data_sub.loc[data_sub['antibiotic'] == anti]
            data_sub_anti = data_sub_anti.loc[:, ('genome_id', 'resistant_phenotype')]

            data_sub_anti=data_sub_anti.drop_duplicates()
            #Drop the all rows with the same 'genome_id' but different 'resistant_phenotype!!! May 21st.
            #
            df_bad=data_sub_anti[data_sub_anti.duplicated(['genome_id'])]#all rows with the same 'genome_id' but different 'resistant_phenotype
            #drop
            bad=df_bad['genome_id'].to_list()
            BAD.append(bad)
            if bad != []:

                data_sub_anti = data_sub_anti[~data_sub_anti['genome_id'].isin(bad)]
            #----------------------------------------------------------------

            pheno = {'Resistant': 1, 'Susceptible': 0, 'S': 0, 'Non-susceptible': 1, 'RS': 1}
            data_sub_anti.resistant_phenotype = [pheno[item] for item in data_sub_anti.resistant_phenotype]
            # check data balance
            balance_check = data_sub_anti.groupby(by="resistant_phenotype").count()
            #print('Check phenotype balance.', balance_check)
            if balance_check.index.shape[0] == 2:# there is Neisseria gonorrhoeae w.r.t. ceftriaxone, no R pheno.
                balance_ratio = balance_check.iloc[0]['genome_id'] / balance_check.iloc[1]['genome_id']
                if min(balance_check.iloc[0]['genome_id'], balance_check.iloc[1]['genome_id']) <100:
                    select_antibiotic_final.remove(anti)
                else:#before final selected, may some bad samples.need to remove in next steps.
                    # save the ID for each species and each antibiotic
                    data_sub_anti.to_csv(save_name_modelID + '_pheno.txt', sep="\t") #dataframe with metadata
                    data_sub_anti['genome_id'].to_csv(save_name_modelID, sep="\t", index=False, header=False)
                    if (f_balance== True) and (balance_ratio > 2 or balance_ratio < 0.5):# #final selected, need to downsample.
                        # if not balance, downsampling
                        print('Downsampling starts.....balance_ratio=', balance_ratio)
                        label_down = balance_check.idxmax().to_numpy()[0]
                        label_keep = balance_check.idxmin().to_numpy()[0]
                        print( 'label_down:', label_down)
                        data_draw = data_sub_anti[data_sub_anti['resistant_phenotype'] == str(label_down)]
                        data_left = data_sub_anti[data_sub_anti['resistant_phenotype'] != str(label_down)]
                        data_drew = data_draw.sample(n=int(1.5 * balance_check.loc[str(label_keep), 'genome_id']))
                        data_sub_anti_downsampling = pd.concat([data_drew, data_left], ignore_index=True, sort=False)
                        balance_check = data_sub_anti_downsampling.groupby(by="resistant_phenotype").count()
                        print('Check phenotype balance after downsampling.', balance_check)
                        balance_check.to_csv(save_name_modelID + 'balance_check.txt', mode='a', sep="\t")
                    else:
                        pass
            else:
                select_antibiotic_final.remove(anti)
        #check if samples with conflicting pheno exit in other antibiotic groups
        BAD=[j for sub in BAD for j in sub]
        if BAD !=[]:
            for anti in select_antibiotic_final:
                save_name_modelID=name_utility.GETname_meta(species,anti,level)
                data_sub_anti = pd.read_csv(save_name_modelID + '_pheno.txt', dtype={'genome_id': object}, index_col=0,sep="\t")
                data_sub_anti = data_sub_anti[~data_sub_anti['genome_id'].isin(BAD)]
                balance_check = data_sub_anti.groupby(by="resistant_phenotype").count()
                if min(balance_check.iloc[0]['genome_id'], balance_check.iloc[1]['genome_id']) <100:
                    select_antibiotic_final.remove(anti)
                    os.remove(save_name_modelID + '_pheno.txt')
                    os.remove(save_name_modelID)

                else:#final selected. overwriting.
                    # save the ID for each species and each antibiotic
                    data_sub_anti.to_csv(save_name_modelID + '_pheno.txt', sep="\t") #dataframe with metadata
                    data_sub_anti['genome_id'].to_csv(save_name_modelID, sep="\t", index=False, header=False)

        if species=='Streptococcus pneumoniae':
            # these two (cotrimoxazole and trimethoprim/sulfamethoxazole) were not merged as we realized this too late,
            # and either were equipped with enough data samples. so we simply remove one overlapping data set
            select_antibiotic_final.remove('cotrimoxazole') #Mar,2022. as 'cotrimoxazole' ='trimethoprim/sulfamethoxazole'
            select_antibiotic_final.remove('beta-lactam') #April,2022. as beta-lactam class includes multiple antibiotics.

        if species=='Mycobacterium tuberculosis':
            ## not merged.
            select_antibiotic_final.remove('rifampin') #May,2023. as 'rifampicin' ='trifampin'


        Species_quality.at[species,'modelling antibiotics']= select_antibiotic_final
        Species_quality.at[species, 'number'] =len(select_antibiotic_final)
    print('selected species and antibiotics:')
    print(Species_quality)#visualization of selected species.
    main_meta,_=name_utility.GETname_main_meta(level)
    Species_quality.to_csv(main_meta, sep="\t")



#todo future: cotrimoxazole and trimethoprim/sulfamethoxazole merge data sets.
