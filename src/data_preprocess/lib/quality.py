#!/usr/bin/python
import os
import pandas as pd
import sys
sys.path.append('../../')
from src.amr_utility import name_utility



def criteria(species, df,level):
    '''
    :param df: (pandas dataframe) genome ID with quality meta data
    :param level: quality control level. In this AMR benchmarking study, we apply the "loose" level.
    :return: (pandas dataframe) genome ID  with good quality
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


    # calculate the mean genome_length
    mean_genome_l = df["genome.genome_length"].mean()
    # filter abs(genome length - mean length) <= mean length/20'''
    df = df[abs(df['genome.genome_length'] - mean_genome_l) <= mean_genome_l / 20]
    if species == 'Pseudomonas aeruginosa':   ## Pseudomonas_aeruginosa: add on the genomes from the Ariane Khaledi et al. EMBO molecular medicine 12.3 (2020) article.
        pa_add = pd.read_csv('./data/PATRIC/Pseudomonas_aeruginosa_add.txt', dtype={'genome.genome_id': object}, header=0)
        df = df.append(pa_add, sort=False)
        df = df.drop_duplicates(subset=['genome.genome_id'])
    df = df.reset_index(drop=True)
    return df


def extract_id_quality(temp_path,level):
    '''
    input: read the downloaded quality metadata, saved at the subdirectory: /quality.
    output: of the 13 previously selected species, select species with more than 200 good-quality genomes;
    and corresponding genome list.
    '''

    info_species=pd.read_csv(temp_path+'list_species_final_bq.txt', dtype={'genome_id': object}, sep="\t", header=0)
    species_list = info_species['species'].tolist()
    number_All=[]
    number_FineQuality=[]
    for species in species_list:
        save_all_quality,save_quality=name_utility.GETname_quality(species,level)
        df=pd.read_csv(save_all_quality,dtype={'genome.genome_id': object, 'genome.genome_name': object}, sep="\t")
        number_All.append(df.shape[0])
        #=======================
        # Apply QC criteria
        df_quality=criteria(species, df, level)
        # =========================
        # Save the selected genome ID of each species to TXT file.
        #=====================
        df_quality.to_csv(save_quality, sep="\t") #'quality/GenomeFineQuality_' + str(species.replace(" ", "_")) + '.txt'
        number_FineQuality.append(df_quality.shape[0])


    ### load 13 species selected (before QC) by genome number
    count_quality = pd.DataFrame(list(zip(number_All, number_FineQuality)), index=info_species, columns=['Number of genomes','Number of fine quality genomes'])
    count_species = pd.read_csv(temp_path+'list_species_final_bq.txt', dtype={'genome_id': object}, sep="\t",
                             header=0,index_col='species')
    count_final=pd.concat([count_species, count_quality], axis=1).reindex(count_species.index) ## no selection in this command

    ### filter out species with no more than 200 genomes. no species was filtered out.
    count_final=count_final[count_final['Number of fine quality genomes']>200]
    ### Save selected species to a file
    count_final.rename(columns={'count': 'Number of genomes with AMR metadata'}, inplace=True)
    count_final.to_csv("./data/PATRIC/meta/fine_quality/"+str(level)+'_list_species_final_quality.csv',sep="\t") ## species list with genome number


def unique_cols(df):
    a = df.to_numpy() # df.values (pandas<0.24)
    return (a[0] == a).all(0)

def filter_phenotype(level,f_balance):
    '''
    This function filters genomes by ill-annotated phenotypes
    This function also filters datasets genome number in each phenotype class (retain those at least 100 genomes for each class)
    Output: genome ID list and  AMR phenotype for each species-antibiotic combination/dataset. Save at
    Output: Species_antibiotic_FineQuality.csv: selected species and antibiotic for benchmarking.
    '''

    ## Load in selected genomes ( i.e. good quality & with AMR metadata ) for the selected 11 species
    genome_amr, info_species = name_utility.load_metadata(SpeciesFile="./data/PATRIC/meta/fine_quality/"+str(level)+'_list_species_final_quality.csv')
    Species_quality=pd.DataFrame(index=info_species, columns=['number','modelling antibiotics']) #initialize

    ## 1. Drop genomes with phenotype annotated as 'Intermediate''Not defined'
    genome_amr = genome_amr[(genome_amr.resistant_phenotype != 'Intermediate') & (genome_amr.resistant_phenotype != 'Not defined')]

    for species in info_species:

        ### 2. Since the genomes selected after quality control  consist of a mix of those with and lacking AMR metadata,
        ### our initial step involves obtaining the intersection of high-quality genomes and those with AMR metadata.
        save_all_quality,save_quality=name_utility.GETname_quality(species,level)
        genome_OneSpecies = genome_amr[genome_amr['species'] == species]
        df = pd.read_csv(save_quality,dtype={'genome.genome_id': object, 'genome.genome_name': object}, index_col=0,sep="\t")
        id_GoodQuality=df['genome.genome_id']
        genome_OneSpecies=genome_OneSpecies[genome_OneSpecies['genome_id'].isin(id_GoodQuality)]

        ### 3. replace amoxicillin-clavulanate'	with 'amoxicillin/clavulanic acid'
        genome_OneSpecies=genome_OneSpecies.replace('amoxicillin-clavulanate', 'amoxicillin/clavulanic acid') # two names for the same antibiotic

        ## 4. select the antibiotic with genome_id.count >200
        genome_OneSpecies_ = genome_OneSpecies.groupby(by="antibiotic")['genome_id']
        summary = genome_OneSpecies_.count().to_frame()
        summary = summary[summary['genome_id'] > 200]

        ### 5. some genomes are annotated with different resistant_phenotype for the same antibiotic.
        ### These genomes are surely ill-annotated, and should be excluded via BAD list.
        BAD=[]

        ### 6. For each species-antibiotic combination, ensure more then 100 genomes at both Resistant & Susceptible classes
        select_antibiotic = summary.index.to_list()
        genome_OneSpecies = genome_OneSpecies.loc[genome_OneSpecies['antibiotic'].isin(select_antibiotic)]
        select_antibiotic_final= select_antibiotic.copy()
        for anti in select_antibiotic:
            save_name_modelID=name_utility.GETname_meta(species,anti,level)

            # select genome_id and resistant_phenotype
            genome_OneSpeciesAnti = genome_OneSpecies.loc[genome_OneSpecies['antibiotic'] == anti]
            genome_OneSpeciesAnti = genome_OneSpeciesAnti.loc[:, ('genome_id', 'resistant_phenotype')]
            genome_OneSpeciesAnti=genome_OneSpeciesAnti.drop_duplicates() ## just in case, there are duplicates items


            ## some genomes are annotated with different resistant_phenotype for the same antibiotic.
            ## Drop the all rows with the same 'genome_id' but different 'resistant_phenotype!!! May 21st.
            df_bad=genome_OneSpeciesAnti[genome_OneSpeciesAnti.duplicated(['genome_id'])]#all rows with the same 'genome_id' but different 'resistant_phenotype
            bad=df_bad['genome_id'].to_list()
            BAD.append(bad) ## for checking if samples with conflicting phenotype exit in other antibiotic groups
            if bad != []:
                genome_OneSpeciesAnti = genome_OneSpeciesAnti[~genome_OneSpeciesAnti['genome_id'].isin(bad)]
            #----------------------------------------------------------------

            pheno = {'Resistant': 1, 'Susceptible': 0, 'S': 0, 'Non-susceptible': 1, 'RS': 1}
            genome_OneSpeciesAnti.resistant_phenotype = [pheno[item] for item in genome_OneSpeciesAnti.resistant_phenotype]
            # check data balance
            balance_check = genome_OneSpeciesAnti.groupby(by="resistant_phenotype").count()
            if balance_check.index.shape[0] == 2:# there is Neisseria gonorrhoeae w.r.t. ceftriaxone, no R pheno.
                balance_ratio = balance_check.iloc[0]['genome_id'] / balance_check.iloc[1]['genome_id']
                if min(balance_check.iloc[0]['genome_id'], balance_check.iloc[1]['genome_id']) <100:
                    select_antibiotic_final.remove(anti)
                else:
                    ## save the ID for each species and each antibiotic
                    ## note: some species-antibiotic combinations will be removed after processing BAD, later.
                    genome_OneSpeciesAnti.to_csv(save_name_modelID + '_pheno.txt', sep="\t") #dataframe with metadata
                    genome_OneSpeciesAnti['genome_id'].to_csv(save_name_modelID, sep="\t", index=False, header=False)

                    ############################################################################################################
                    ######## currently, this downsampling precedure is not applied in our study.
                    if (f_balance== True) and (balance_ratio > 2 or balance_ratio < 0.5):# #final selected, need to downsample.
                        # if not balance, downsampling
                        print('Downsampling starts.....balance_ratio=', balance_ratio)
                        label_down = balance_check.idxmax().to_numpy()[0]
                        label_keep = balance_check.idxmin().to_numpy()[0]
                        print( 'label_down:', label_down)
                        data_draw = genome_OneSpeciesAnti[genome_OneSpeciesAnti['resistant_phenotype'] == str(label_down)]
                        data_left = genome_OneSpeciesAnti[genome_OneSpeciesAnti['resistant_phenotype'] != str(label_down)]
                        data_drew = data_draw.sample(n=int(1.5 * balance_check.loc[str(label_keep), 'genome_id']))
                        genome_OneSpeciesAnti_downsampling = pd.concat([data_drew, data_left], ignore_index=True, sort=False)
                        balance_check = genome_OneSpeciesAnti_downsampling.groupby(by="resistant_phenotype").count()
                        print('Check phenotype balance after downsampling.', balance_check)
                        balance_check.to_csv(save_name_modelID + 'balance_check.txt', mode='a', sep="\t")
                    else:
                        pass
                    ############################################################################################################
            else:
                select_antibiotic_final.remove(anti)


        ## check if samples with conflicting phenotype exit in other antibiotic groups
        ## Although this kind of samples does not influence the other datasets, but we deem them as unreliable. Remove!
        BAD=[j for sub in BAD for j in sub]
        if BAD !=[]:
            for anti in select_antibiotic_final:
                save_name_modelID=name_utility.GETname_meta(species,anti,level)
                genome_OneSpeciesAnti = pd.read_csv(save_name_modelID + '_pheno.txt', dtype={'genome_id': object}, index_col=0,sep="\t")
                genome_OneSpeciesAnti = genome_OneSpeciesAnti[~genome_OneSpeciesAnti['genome_id'].isin(BAD)]
                balance_check = genome_OneSpeciesAnti.groupby(by="resistant_phenotype").count()
                if min(balance_check.iloc[0]['genome_id'], balance_check.iloc[1]['genome_id']) <100:
                    ### remove previously saved datasets due to lack of enough genomes
                    select_antibiotic_final.remove(anti)
                    os.remove(save_name_modelID + '_pheno.txt')
                    os.remove(save_name_modelID)

                else:#final selected. overwriting.
                    # save the ID for each species and each antibiotic
                    genome_OneSpeciesAnti.to_csv(save_name_modelID + '_pheno.txt', sep="\t") #dataframe with metadata
                    genome_OneSpeciesAnti['genome_id'].to_csv(save_name_modelID, sep="\t", index=False, header=False)


        ###  Address duplicate datasets arising from antibiotic alias issues
        if species=='Streptococcus pneumoniae':
            ## these two (cotrimoxazole and trimethoprim/sulfamethoxazole) were not merged
            ## and either were equipped with enough data samples. so we remove one of them
            select_antibiotic_final.remove('cotrimoxazole') #Mar,2022. as 'cotrimoxazole' ='trimethoprim/sulfamethoxazole'
            select_antibiotic_final.remove('beta-lactam') #April,2022. as beta-lactam class includes multiple antibiotics.

        if species=='Mycobacterium tuberculosis':
            ## not merged.
            ## May,2023. as 'rifampicin' ='trifampin'
            select_antibiotic_final.remove('rifampin')


        Species_quality.at[species,'modelling antibiotics']= select_antibiotic_final
        Species_quality.at[species, 'number'] =len(select_antibiotic_final)
    print('selected species and antibiotics:')
    print(Species_quality)# selected species.
    Species_quality = Species_quality[ (Species_quality['number'] > 0)]##drop 0 rows
    main_meta,_=name_utility.GETname_main_meta(level)
    Species_quality.to_csv(main_meta, sep="\t")



#todo future: cotrimoxazole and trimethoprim/sulfamethoxazole merge data sets.
