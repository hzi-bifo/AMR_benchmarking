#!/usr/bin/python
import sys
import os
sys.path.append('../')
sys.path.insert(0, os.getcwd())
import pandas as pd
import amr_utility.name_utility
import amr_utility.graph_utility
import seaborn as sns
import argparse



def download_quality():
    '''download files related to selecting good quality genomes
    Please use the generate command to download files to the sub folder quality/
    '''

    data, info_species=amr_utility.name_utility.load_metadata(SpeciesFile='list_species_final_bq.txt')
    for species in info_species:

        para_genus=species.split(' ')[0]
        para_species=species.split(' ')[1]

        bashCommand = "p3-all-genomes --eq genus,"+str(para_genus)+ \
                      " --eq species,"+ str(para_species)+ \
                      " -a genome_name,genome_status,genome_length,genome_quality,plasmids,contigs," \
                      "fine_consistency,coarse_consistency,checkm_completeness,checkm_contamination " \
                      " >  quality/"+str(para_genus)+"_"+str(para_species)+".csv"
        print(bashCommand)
        #process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        #output, error = process.communicate()
        #by hand...
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
        pa_add = pd.read_csv('Pseudomonas_aeruginosa_add.txt', dtype={'genome.genome_id': object}, header=0)
        df = df.append(pa_add, sort=False)
        df = df.drop_duplicates(subset=['genome.genome_id'])
    df = df.reset_index(drop=True)
    return df
def extract_id_quality(level):
    '''
    inpout: downloaded quality meta data, saved at the subdirectory: /quality.
    '''
    data, info_species = amr_utility.name_utility.load_metadata(SpeciesFile='list_species_final_bq.txt')
    number_All=[]
    number_FineQuality=[]
    for species in info_species:
    #for species in ['Pseudomonas aeruginosa']:
        save_all_quality,save_quality=amr_utility.name_utility.GETsave_quality(species,level)
        df=pd.read_csv(save_all_quality,dtype={'genome.genome_id': object, 'genome.genome_name': object}, sep="\t")
        number_All.append(df.shape[0])
        #with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
            #print(df)
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
    print('Visualization species with antibiotic selected',count_quality)
    count_species = pd.read_csv('list_species_final_bq.txt', dtype={'genome_id': object}, sep="\t",
                             header=0,index_col='species')
    count_final=pd.concat([count_species, count_quality], axis=1).reindex(count_species.index)# visualization. no selection in this cm.
    # filter Shigella sonnei and Enterococcus faecium, only 25,144
    count_final=count_final[count_final['Number of fine quality genomes']>200]
    count_final.rename(columns={'count': 'Number of genomes with AMR metadata'}, inplace=True)
    count_final.to_csv(str(level)+'_list_species_final_quality.csv',sep="\t")#species list.
    print(count_final)
def extract_id_quality_analysis(check_dif,check_all,plot_contig):

    if os.path.exists('QualityControl_summary.csv') and check_all==True:
        os.remove('QualityControl_summary.csv')
    data, info_species = amr_utility.name_utility.load_metadata(SpeciesFile='list_species_final_bq.txt')
    # just check pa
    # data = data[(data.resistant_phenotype != 'Intermediate') & (data.resistant_phenotype != 'Not defined')]
    # g = data[data['genome_id'] == '1163395.3']
    # print(g)
    df_contig=[]#initialize the dataframe for contig number visualization
    number_All=[]
    number_FineQuality=[]
    #for species in ['Pseudomonas aeruginosa']:
    for species in info_species:

        save_all_quality,save_quality = amr_utility.name_utility.GETsave_quality(species, level)
        df = pd.read_csv(save_all_quality, dtype={'genome.genome_id': object, 'genome.genome_name': object,'genome.contigs': int}, sep="\t")
        number_All.append(df.shape[0])
        #1.
        if check_dif==True:
            # with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
            # print(df)
            # Genome Status != Plasmid

            # check if there is not none in any of 'genome.checkm_completeness' 'genome.checkm_contamination'
            # 'genome.coarse_consistency''genome.fine_consistency' is not none
            df1=df[(df['genome.genome_status'] != 'Plasmid') &(df['genome.genome_quality'] == 'Good') &(df['genome.contigs']<= 100)
                    &(df['genome.fine_consistency']>= 97)&(df['genome.coarse_consistency']>= 98)]
            df2 = df[(df['genome.genome_status'] != 'Plasmid') &(df['genome.genome_quality'] == 'Good') &(df['genome.contigs']<= 100)
                    &(df['genome.fine_consistency']>= 97)&(df['genome.coarse_consistency']>= 98)&(df['genome.checkm_completeness']>= 98)
                    &(df['genome.checkm_contamination']<= 2)]



            common = df.merge(df2, on=['genome.genome_id'])

            diff=df[(~df['genome.genome_id'].isin(common['genome.genome_id']))]
            print(species)
            with pd.option_context('display.max_columns', None,'display.max_rows', None):
                print(diff[['genome.genome_id','genome.checkm_completeness','genome.checkm_contamination']])
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            # diff=diff[['genome.genome_id', 'genome.checkm_completeness', 'genome.checkm_contamination']]
            diff.to_csv('quality/check/log_' + str(species.replace(" ", "_")) + '_diff.txt',sep="\t")

        #2.
        elif check_all == True:
            #analsis of the distribution of 'genome.genome_status', 'genome.genome_quality', 'genome.contigs', etc.
            print(species)
            df=df[(df['genome.genome_status'] != 'Plasmid') & (df['genome.genome_quality'] == 'Good')
               & (df['genome.fine_consistency'] >= 97) & (df['genome.coarse_consistency'] >= 98)]

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(df.describe())

            # summary
            des2 = df.isnull().sum().to_frame(name='missing').T
            des3 = (df.isnull().sum()/len(df)*100).to_frame(name='percentage of missing').T
            s=pd.concat([df.describe(include ='all'), des2,des3])
            with open("QualityControl_summary.csv", "a") as file_object:
                # Append 'hello' at the end of file
                file_object.write("\n")
                file_object.write(species)
                # file_object.write(str(df.info(verbose=True)))
            s.to_csv(r'QualityControl_summary.csv',  mode='a')
        #3.
        else:
            #summary of 'genome.checkm_completeness', 'genome.checkm_contamination' from those
            # with (df['genome.fine_consistency']>= 97)&(df['genome.coarse_consistency']>= 98)
            df = df[(df['genome.genome_status'] != 'Plasmid') & (df['genome.genome_quality'] == 'Good')
                     & (df['genome.fine_consistency'] >= 97) & (df['genome.coarse_consistency'] >= 98)]
            des2 = df.isnull().sum().to_frame(name='missing').T
            des3 = (df.isnull().sum() / len(df) * 100).to_frame(name='percentage of missing').T
            summary = pd.concat([df.describe(include='all'), des2, des3])
            with open("QualityControl_summary_2.csv", "a") as file_object:
                # Append 'hello' at the end of file
                file_object.write("\n")
                file_object.write(species)
                # file_object.write(str(df.info(verbose=True)))
            summary.to_csv(r'QualityControl_summary_2.csv', mode='a')

            #----------------------------------------------------------------------------------

        #caluculate the mean genome_length
        mean_genome_l=df["genome.genome_length"].mean()
        df=df[abs(df['genome.genome_length']-mean_genome_l) <= mean_genome_l/20]#filter abs(genome length - mean length) <= mean length/20'''
        df = df.reset_index(drop=True)
        #with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
            #print(df)

        number_FineQuality.append(df.shape[0])

        df_contig.append(df)

    #4.
    if plot_contig==True:#set check_all ==False, check_dif==False
        # draw boxplot of genome.contigs after loose quality control
        # df['genome.genome_status'] != 'Plasmid') & (df['genome.genome_quality'] == 'Good')
        # & (df['genome.fine_consistency'] >= 97) & (df['genome.coarse_consistency'] >= 98) and genome_length control

        df_contig = pd.concat(df_contig)
        df_contig['species'] = df_contig['genome.genome_name'].apply(lambda x: ' '.join(x.split(' ')[0:2]))
        ax = sns.boxplot(x='species',y='genome.contigs', data=df_contig)
        ax = sns.swarmplot(x='species',y='genome.contigs',  data=df_contig, color=".25")
        ax.savefig('quality/check/boxplot_contigs.png')
    # ---------------------------------------

    count_quality = pd.DataFrame(list(zip(number_All, number_FineQuality)), index=info_species, columns=['Number of genomes','Number of fine quality genomes'])
    #print(count_quality)
    #change 'list_species_final.txt'
    count_species = pd.read_csv('list_species_final_bq.txt', dtype={'genome_id': object}, sep="\t",
                             header=0,index_col='species')
    count_final=pd.concat([count_species, count_quality], axis=1).reindex(count_species.index)
    # filter Shigella sonnei and Enterococcus faecium, only 25,144
    count_final=count_final[count_final['Number of fine quality genomes']>200]
    ##count_final.to_csv('list_species_final_quality.csv',sep="\t")
    print(count_final)

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
    data, info_species = amr_utility.name_utility.load_metadata(SpeciesFile=str(level)+'_list_species_final_quality.csv')

    # drop phenotype with 'Intermediate''Not defined'
    data = data[(data.resistant_phenotype != 'Intermediate') & (data.resistant_phenotype != 'Not defined')]
    #=======================================================
    Species_quality=pd.DataFrame(index=info_species, columns=['number','modelling antibiotics'])#initialize for visualization
    #print(Species_quality)
    #for species in ['Pseudomonas aeruginosa']:

    for species in info_species:
        print(species)
        BAD=[]
        save_all_quality,save_quality=amr_utility.name_utility.GETsave_quality(species,level)
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
        data_sub=data_sub.replace('amoxicillin-clavulanate', 'amoxicillin/clavulanic acid')

        # by each antibiotic.
        data_anti = data_sub.groupby(by="antibiotic")['genome_id']
        summary = data_anti.count().to_frame()
        # summary.columns=['antibiotic','genome_id_count']

        summary = summary[summary['genome_id'] > 200]# select the antibiotic with genome_id.count >200
        # print(summary)
        select_antibiotic = summary.index.to_list()
        data_sub = data_sub.loc[data_sub['antibiotic'].isin(select_antibiotic)]
        # with pd.option_context('display.max_columns', None):
        # print(data_sub)
        # [2]. check balance of the phenotype
        # ====================================================================
        # print(species)
        # print('============= Select_antibiotic:', len(select_antibiotic), select_antibiotic)
        select_antibiotic_fianl= select_antibiotic.copy()
        for anti in select_antibiotic:
            save_name_meta,save_name_modelID=amr_utility.name_utility.GETsave_name_modelID(level,species,anti,True)
            # for anti in ['mupirocin', 'penicillin', 'rifampin', 'tetracycline', 'vancomycin']:
            # logDir = os.path.join('log/log_' + str(species.replace(" ", "_"))+'_'+str(anti))

            # select genome_id and  resistant_phenotype
            data_sub_anti = data_sub.loc[data_sub['antibiotic'] == anti]
            data_sub_anti = data_sub_anti.loc[:, ('genome_id', 'resistant_phenotype')]
            # print(species, '=============>>', anti)
            #print(data_sub_anti)
            data_sub_anti=data_sub_anti.drop_duplicates()
            #Drop the all rows with the same 'genome_id' but different 'resistant_phenotype!!! May 21st.
            #
            df_bad=data_sub_anti[data_sub_anti.duplicated(['genome_id'])]#all rows with the same 'genome_id' but different 'resistant_phenotype
            #drop
            bad=df_bad['genome_id'].to_list()
            BAD.append(bad)
            if bad != []:
                # print(bad)
                # # print(len(bad),'out of',data_sub_anti.shape[0] )
                # print(species,anti)
                data_sub_anti = data_sub_anti[~data_sub_anti['genome_id'].isin(bad)]
            #----------------------------------------------------------------

            pheno = {'Resistant': 1, 'Susceptible': 0, 'S': 0, 'Non-susceptible': 1, 'RS': 1}
            data_sub_anti.resistant_phenotype = [pheno[item] for item in data_sub_anti.resistant_phenotype]
            # check data balance
            balance_check = data_sub_anti.groupby(by="resistant_phenotype").count()
            balance_check.to_csv(save_name_meta + 'balance_check.txt', sep="\t")
            # print('balanced dataset.', balance_check)
            #print('Check phenotype balance.', balance_check)
            if balance_check.index.shape[0] == 2:# there is Neisseria gonorrhoeae w.r.t. ceftriaxone, no R pheno.
                balance_ratio = balance_check.iloc[0]['genome_id'] / balance_check.iloc[1]['genome_id']
                if min(balance_check.iloc[0]['genome_id'], balance_check.iloc[1]['genome_id']) <100:
                    select_antibiotic_fianl.remove(anti)
                    # print('not selected')
                    # print(balance_check)
                else:#final selected
                    # save the ID for each species and each antibiotic
                    print(anti)
                    print(balance_check)
                    data_sub_anti.to_csv(save_name_modelID + '.txt', sep="\t") #dataframe with metadata
                    # fna location list
                    data_sub_anti.to_csv(save_name_modelID + 'resfinder', sep="\t", index=False,
                                         header=False)  # for the use of resfinder cluster
                    data_sub_anti['genome_id_location'] = 'to_csv'+ data_sub_anti['genome_id'].astype(str)+'.fna'
                    data_sub_anti['genome_id_location'].to_csv(save_name_modelID+'_path', sep="\t",index=False,header=False)

                    #For Ehsan generating CV splits . Aug 3 not need any more seems.
                    #data_sub_anti.to_csv('model/TO_Ehsan/'+str(level)+'/Data_' + str(species.replace(" ", "_")) + '_' + str(
                    # anti.translate(str.maketrans({'/': '_', ' ': '_'}))), sep="\t", index=False, header=False)
                    data_sub_anti['genome_id'].to_csv(save_name_modelID, sep="\t", index=False, header=False)
                    if (f_balance== True) and (balance_ratio > 2 or balance_ratio < 0.5):# #final selected, need to downsample.
                        # if not balance, downsampling
                        print('Downsampling starts.....balance_ratio=', balance_ratio)
                        label_down = balance_check.idxmax().to_numpy()[0]
                        label_keep = balance_check.idxmin().to_numpy()[0]
                        print('!!!!!!!!!!!!!label_down:', label_down)
                        data_draw = data_sub_anti[data_sub_anti['resistant_phenotype'] == str(label_down)]
                        data_left = data_sub_anti[data_sub_anti['resistant_phenotype'] != str(label_down)]
                        data_drew = data_draw.sample(n=int(1.5 * balance_check.loc[str(label_keep), 'genome_id']))
                        data_sub_anti_downsampling = pd.concat([data_drew, data_left], ignore_index=True, sort=False)
                        # print('downsampling',data_sub_anti)
                        # check balance again:
                        balance_check = data_sub_anti_downsampling.groupby(by="resistant_phenotype").count()
                        print('Check phenotype balance after downsampling.', balance_check)
                        balance_check.to_csv(save_name_modelID + 'balance_check.txt', mode='a', sep="\t")
                    else:
                        # print('balanced dataset.', balance_check)
                        pass
            else:
                select_antibiotic_fianl.remove(anti)
        #check if samples with conflicting pheno exit in other antibiotic groups
        BAD=[j for sub in BAD for j in sub]
        if BAD !=[]:
            for anti in select_antibiotic_fianl:
                save_name_meta, save_name_modelID = amr_utility.name_utility.GETsave_name_modelID(level, species, anti,
                                                                                                  True)
                # for anti in ['mupirocin', 'penicillin', 'rifampin', 'tetracycline', 'vancomycin']:
                # logDir = os.path.join('log/log_' + str(species.replace(" ", "_"))+'_'+str(anti))

                # select genome_id and  resistant_phenotype

                data_sub_anti = pd.read_csv(save_name_modelID + '.txt', dtype={'genome_id': object}, index_col=0,sep="\t")

                maybebad=data_sub_anti[data_sub_anti['genome_id'].isin(BAD)]
                # print(maybebad.shape[0])
                # print('out of ',data_sub_anti.shape[0])
                # print(species,anti)
                # print('==============================================')
                data_sub_anti = data_sub_anti[~data_sub_anti['genome_id'].isin(BAD)]
                balance_check = data_sub_anti.groupby(by="resistant_phenotype").count()
                if min(balance_check.iloc[0]['genome_id'], balance_check.iloc[1]['genome_id']) <100:
                    select_antibiotic_fianl.remove(anti)
                    os.remove(save_name_modelID + '.txt')
                    os.remove(save_name_modelID)
                    os.remove(save_name_modelID+'resfinder')
                    os.remove(save_name_modelID+'_path')
                    # print('not selected in second run!')
                    # print(species,anti)
                    # print(balance_check)
                else:#final selected
                    # save the ID for each species and each antibiotic

                    data_sub_anti.to_csv(save_name_modelID + '.txt', sep="\t") #dataframe with metadata
                    data_sub_anti['genome_id'].to_csv(save_name_modelID, sep="\t", index=False, header=False)
                    # fna location list
                    data_sub_anti.to_csv(save_name_modelID+'resfinder', sep="\t", index=False, header=False)#for the use of resfinder cluster
                    data_sub_anti['genome_id_location'] = '/vol/projects/BIFO/patric_genome/'+ data_sub_anti['genome_id'].astype(str)+'.fna'
                    data_sub_anti['genome_id_location'].to_csv(save_name_modelID+'_path', sep="\t",index=False,header=False)

                    #For Ehsan generating CV splits

        Species_quality.at[species,'modelling antibiotics']= select_antibiotic_fianl
        Species_quality.at[species, 'number'] =len(select_antibiotic_fianl)

    print(Species_quality)#visualization of selected species.
    Species_quality.to_csv(str(level)+'_Species_antibiotic_FineQuality.csv', sep="\t")



if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--k', '--Kmer', default=8, type=int, required=True,
    #                     help='Kmer size')
    # parser.add_argument('-p', '--pca', dest='pca',
    #                     help='Use pca', action='store_true', )
    #



    # download_quality()
    level='strict'# quality control level:'strict','loose'.
    f_balance=False
    # extract_id_quality(level) #downloaded quality meta data, saved at the subdirectory quality.
    filter_quality(level,f_balance)
    #===========================================
    '''only for the author use, delete when completing.
    check_dif: check the difference of included strains for two sets of criteria'''
    # extract_id_quality_analysis(check_dif=False, check_all=True,plot_contig=False)
    #============================================

    '''
    df3=df[df['genome.checkm_completeness'].notnull()]
    print(df3['genome.fine_consistency'].isnull().values.any())
    print(df3['genome.coarse_consistency'].isnull().values.any())
    #check if
    df3 = df[df['genome.checkm_contamination'].notnull()]
    print(df3['genome.fine_consistency'].isnull().values.any())
    print(df3['genome.coarse_consistency'].isnull().values.any())
    pa = pd.read_csv('Pseudomonas_aeruginosa_add.txt', dtype={'genome.genome_id': object}, index_col=0, header=None)
    print(pa.index.tolist())
    result = df[df['genome.genome_id'].isin(pa.index.tolist())]
    print(result)
    result = result.groupby(by="genome.genome_id")['genome.genome_id']
    print(result.count())
    '''
