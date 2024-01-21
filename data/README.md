# Tutorials for creating AMR benchmarking datasets 
Welcome to the tutorial on data preprocessing. This tutorial guides you through the procedures for creating <a href="https://github.com/hzi-bifo/AMR_benchmarking/wiki/Dataset-overview">78 AMR phenotyping benchmarking datasets</a> from the <a href="https://www.bv-brc.org/">PATRIC</a> genome database.

## Table of Contents
- [1. Download metadata from PATRIC FTP](#1)
- [1. Data preprocessing](#2)
	- [2.1 Filter species and antibiotic by genome number](#2.1)
	- [2.2 Download genome quality information](#2.2)
	- [2.3 Filter genomes by genome quality](#2.3)
	- [2.4 Filter out genomes with ill-annotated phenotypes; filter datasets by genome numbers](#2.4)
	- [2.5 Others: dataset summary, multi-species dataset](#2.5)
- [3. Download genome sequences from the PATRIC database](#3)


## <a name="1"></a>1. Download metadata from PATRIC FTP
- Download `PATRIC_genomes_AMR.txt` from https://docs.patricbrc.org/user_guides/ftp.html or find a <a href="https://github.com/hzi-bifo/AMR_benchmarking/blob/main/data/PATRIC/PATRIC_genomes_AMR.txt">version</a> downloaded by us in Dec 2020

## <a name="2"></a>2. Data preprocessing
-  Python file `preprocess.py` processes the metadata in `PATRIC_genomes_AMR.txt` to extract 78 datasets, through steps 2.1 - 2.5.
```console
python ./src/data_preprocess/preprocess.py
```
### <a name="2.1"></a>2.1 Filtering species and antibiotics by genome number

- 2.11 Calculate the total number of genomes 
```python
import pandas as pd
from  src.amr_utility import name_utility
import numpy as np
from ast import literal_eval
def summarise_strain(temp_path):    
    data = pd.read_csv('./data/PATRIC/PATRIC_genomes_AMR.txt', dtype={'genome_id': object}, sep="\t")
    # get the first column, save it in a file named genome_list
    list = data.loc[:, ("genome_id", "genome_name")]
    list = list.groupby(by="genome_id")
    summary = list.describe()
    summary.to_csv(temp_path + 'list_strain.txt', sep="\t")  ## contain 67836 genomes strains 
```
- 2.12  List all the species
```python
def summarise_species(temp_path):
    '''summerise the species info'''
    data = pd.read_csv(temp_path + 'list_strain.txt', dtype={'genome_id': object}, skiprows=2, sep="\t", header=0)
    data.columns = ['genome_id', 'count', 'unique', 'top', 'freq']
    # summarize the strains
    data['top'] = data['top'].astype(str)  # add a new column
    data['species'] = data.top.apply(lambda x: ' '.join(x.split(' ')[0:2]))
    # Note: download genome data from here, i.e. for each strain.
    data.to_csv(temp_path+'list_temp.txt', sep="\t")
    data = data.loc[:, ("genome_id", "species")]
    # make a summary by strain
    data_s = data.groupby(by="species")
    summary_species = data_s.describe()
    summary_species.to_csv(temp_path + 'list_species.txt', sep="\t")  # list of all species. Number: 99.
```
- 2.13 Filter out those species-antibiotic combinations with less than 500 genomes.
	- This results in 13 species:<sub> <em>Mycobacterium tuberculosis, Salmonella enterica, 
	Streptococcus pneumonia, Neisseria gonorrhoeae, Escherichia coli, Staphylococcus aureus, Klebsiella pneumonia, Enterococcus faecium, Acinetobacter baumannii, 		Pseudomonas aeruginosa, Shigella sonnei, Enterobacter cloacae, Campylobacter jejuni</em></sub>.
```python
def sorting_deleting(N, temp_path): ## N=500
    '''retain only this that has >=N strains for a specific antibiotic w.r.t. a species'''
    data = pd.read_csv(temp_path + 'list_species.txt', dtype={'genome_id': object}, skiprows=2, sep="\t", header=0)
    data = data.iloc[:, 0:2]
    data.columns = ['species', 'count']
    data = data.sort_values(by=['count'], ascending=False)  # sorting
    data = data.reset_index(drop=True)
    data.to_csv(temp_path + 'list_species_sorting.txt', sep="\t")
    data = data[data['count'] > N]  # deleting
    data.to_csv(temp_path + 'list_species_final_bq.txt', sep="\t")  # list of all species selected by 1st round.
  ```

 - 2.14  Extract genome PATRIC IDs for each of the 13 species, respectively
```python
def extract_id(temp_path):
    '''extract (useful) PATRIC id to genome_list
    before quality control'''
    data = pd.read_csv(temp_path + 'list_temp.txt', dtype={'genome_id': object}, sep="\t")
    df_species = pd.read_csv(temp_path + 'list_species_final_bq.txt', dtype={'genome_id': object}, sep="\t", header=0)
    species = df_species['species']
    species = species.tolist()
    ### Select rows that the strain name belongs to the 13 selected species in the last step, i.e. step  2.3 
    data = data.loc[data['species'].isin(species)]
    data = data.reset_index(drop=True)
    list_download = data['genome_id']
    list_download.to_csv('./data/PATRIC/meta/genome_list', sep="\t", index=False,
                         header=False)  ### all the genome ID should be downloaded.
```

### <a name="2.2"></a>2.2 Download genome quality information
- Download quality attribute tables for the 13 selected species from Step 2.13. 

	- Example: download the <em>E. coli</em> genome quality attributes from PATRIC database
 	- Note that this will download all the genomes with and without AMR metadata 
```console
p3-all-genomes --eq genus,Escherichia --eq species,coli -a genome_name,genome_status,genome_length,genome_quality,plasmids,contigs,fine_consistency,coarse_consistency,checkm_completeness,checkm_contamination >  data/PATRIC/quality/Escherichia_coli.csv
```
- Alternatively, find <a href="https://github.com/hzi-bifo/AMR_benchmarking/tree/main/data/PATRIC/quality">versions</a> downloaded by us around Dec 2020

### <a name="2.3"></a>2.3 Filter genomes by genome quality 

- 2.31  Define thresholds for quality attributes: A. (1) sequence data is not plasmid-only; (2) genome quality (provided by PATRIC) is Good; (3) contig count is limited to the greater of either 100 or 0.75 quantiles of the contig count across all genomes of the same specie; (4) fine consistency (provided by PATRIC) higher than 97%; (5) coarse consistency (provided by PATRIC) higher than 98%; (6) completeness (provided by PATRIC) higher than 98% and contamination (provided by PATRIC) lower than 2%, or one of them is null value with the other one meets the criteria. B. For each species, we computed the mean genome length of the selected genomes from step A, and then we retained genomes with lengths within the range of one-twentieth of the calculated mean from the calculated mean. 

```python
def criteria(species, df,level):
    '''
    :param df: (pandas dataframe) genome ID  with quality metadata
    :param level: quality control level. In this AMR benchmarking study, we apply the "loose" level. 
    :return: (pandas dataframe) genome ID   with good quality
    '''
 
    df = df[(df['genome.genome_status'] != 'Plasmid') & (df['genome.genome_quality'] == 'Good') & (
	 df['genome.contigs'] <= max(100, df['genome.contigs'].quantile(0.75))) & (df['genome.fine_consistency'] >= 97) & (df['genome.coarse_consistency'] >= 98)  & 	 	((df['genome.checkm_completeness'] >= 98)| (df['genome.checkm_completeness'].isnull())) & ((df['genome.checkm_contamination'] <= 2)|				 
	(df['genome.checkm_contamination'].isnull()))]


    ### Calculate the mean genome_length
    mean_genome_l = df["genome.genome_length"].mean()
    ### filter abs(genome length - mean length) <= mean length/20'''
    df = df[abs(df['genome.genome_length'] - mean_genome_l) <= mean_genome_l / 20]
    if species == 'Pseudomonas aeruginosa':  ## Pseudomonas_aeruginosa: add on the genomes from the Ariane Khaledi et al. EMBO molecular medicine 12.3 (2020) article.
        pa_add = pd.read_csv('./data/PATRIC/Pseudomonas_aeruginosa_add.txt', dtype={'genome.genome_id': object}, header=0)
        df = df.append(pa_add, sort=False)
        df = df.drop_duplicates(subset=['genome.genome_id'])
    df = df.reset_index(drop=True)
    return df
```
- 2.32 From the genome list generated in step 2.14, extract those genomes in compliance with the good-quality criteria in 2.31. After quality control, filter out species with no more than 200 genomes.
	- This results in 11 species: **<em>Escherichia coli, Staphylococcus aureus, Salmonella enterica, Enterococcus faecium, Campylobacter jejuni, Neisseria gonorrhoeae, Klebsiella pneumoniae, Pseudomonas aeruginosa, Acinetobacter baumannii,  Streptococcus pneumoniae, Mycobacterium tuberculosis </em>**
```python
def extract_id_quality(temp_path,level):
    '''
    input: downloaded quality metadata, saved at the subdirectory: /quality.
    output: selected 11 species, and good-quality genome list
    '''

    df_species=pd.read_csv(temp_path+'list_species_final_bq.txt', dtype={'genome_id': object}, sep="\t", header=0)
    info_species = df_species['species'].tolist()
    number_All=[]
    number_FineQuality=[]
    for species in info_species:
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

    ### filter out species with no more than 200 genomes
    count_final=count_final[count_final['Number of fine quality genomes']>200]
    ### Save selected species to a file
    count_final.rename(columns={'count': 'Number of genomes with AMR metadata'}, inplace=True)
    count_final.to_csv("./data/PATRIC/meta/fine_quality/"+str(level)+'_list_species_final_quality.csv',sep="\t") ## species list with genome number
```

### <a name="2.4"></a>2.4 Filter out genomes with ill-annotated phenotypes; filter datasets by genome numbers
- 2.41 Since the genomes selected after quality control in Step 4 consist of a mix of those with and lacking AMR metadata, our initial step involves obtaining the intersection of high-quality genomes and those with AMR metadata.
- 2.42 Drop genomes with phenotype annotated as 'Intermediate''Not defined'
- 2.43 
- 2.44 Drop genomes with phenotype ill-annotated. Those genomes are annotated with different phenotypes for the same antibiotic.
- 2.45 

```python


```
### <a name="2.5"></a>2.5 Others: dataset summary, multi-species datasets
- Get total genome numbers  
```python
def summary_genome(level):
    '''Count total genome numbers'''
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
```
- Get the genome number per dataset
```python
def count():
    file_utility.make_dir('./data/PATRIC/meta/'+str(level)+'_genomeNumber')
    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]
    df_species = data.index.tolist()
    for species  in  df_species :
        lib.summary.summary_pheno(species,level)
```

- Build the multi-species-antibiotic dataset, selecting antibiotics that are shared by multiple species-antibiotic combinations
```python
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
```
-  Calculates the multi-species-antibiotic dataset size
```python
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
```



## <a name="3"></a>3. Download genome sequences from the PATRIC database

```sh
${data_dir}=<path_to_directory_to_save_enomes>

for i in `cat ./doc/genome_list`;do
    if [ ! -f "$i/$i.fna" ]; then
	 printf 'Downloading (%s)\n' "$i/$i.fna"
         wget -qN "ftp://ftp.patricbrc.org/genomes/$i/$i.fna" -P ${data_dir}
    fi
done
```

