# Data preprocessing tutorial
Welcome to the tutorial on data preprocessing. This tutorial guides you through the procedures for creating <a href="https://github.com/hzi-bifo/AMR_benchmarking/wiki/Dataset-overview">78 AMR phenotyping benchmarking datasets</a> from the <a href="https://www.bv-brc.org/">PATRIC</a> genome database.

## Table of Contents
- [1. Download metadata from PATRIC FTP](#1)
- [2. Filter species and antibiotic](#2)
- [3. Download genome quality information](#2)
- [4. Filter genomes / Genome quality control](#4)
- [5. Filter datasets](#5)
- [6. Download genome sequences from the PATRIC database](#6)

## <a name="1"></a>1. Download metadata from PATRIC FTP
- Download `PATRIC_genomes_AMR.txt` from https://docs.patricbrc.org/user_guides/ftp.html or find a <a href="https://github.com/hzi-bifo/AMR_benchmarking/blob/main/data/PATRIC/PATRIC_genomes_AMR.txt">version</a> downloaded by us in Dec 2020

## <a name="2"></a>2. Filtering species and antibiotic
- This procedure can be achieved by one command composed of steps 2.1-2.
```console
python ./src/data_preprocess/preprocess.py
```
- 2.1 Calculate the total number of genomes 
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
- 2.2  list all the species
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
- 2.3 Filter out those species-antibiotic combinations with less than 500 genomes.
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

 - 2.4  Extract genomes' PATRIC ID
```python
def extract_id(temp_path):
    '''extract (useful) patric id to genome_list
    before quality control'''
    data = pd.read_csv(temp_path + 'list_temp.txt', dtype={'genome_id': object}, sep="\t")
    df_species = pd.read_csv(temp_path + 'list_species_final_bq.txt', dtype={'genome_id': object}, sep="\t", header=0)
    species = df_species['species']
    species = species.tolist()
    ### Select rows that the strain name belongs to the 13 selected species in the last step, i.e. 2.3 step
    data = data.loc[data['species'].isin(species)]
    data = data.reset_index(drop=True)
    list_download = data['genome_id']
    list_download.to_csv('./data/PATRIC/meta/genome_list', sep="\t", index=False,
                         header=False)  ### all the genome ID should be downloaded.
```

## <a name="3"></a>3. Download genome quality information
- Download quality attribute tables for the 13 selected species from Step 2.3

	- Example: download the <em>E. coli</em> genome quality attributes from PATRIC database
```console
p3-all-genomes --eq genus,Escherichia --eq species,coli -a genome_name,genome_status,genome_length,genome_quality,plasmids,contigs,fine_consistency,coarse_consistency,checkm_completeness,checkm_contamination >  Escherichia_coli.csv
```
- Alternatively, find <a href="https://github.com/hzi-bifo/AMR_benchmarking/tree/main/data/PATRIC/quality">versions</a> downloaded by us around Dec 2020

## <a name="4"></a>4. Filter genomes /Genome quality control
- 




## <a name="5"></a>5. Filter datasets


 (data size machine learning model )


## <a name="6"></a>6. Download genome sequences from the PATRIC database

```sh
${data_dir}=<path_to_directory_to_save_enomes>

for i in `cat ./doc/genome_list`;do
    if [ ! -f "$i/$i.fna" ]; then
	 printf 'Downloading (%s)\n' "$i/$i.fna"
         wget -qN "ftp://ftp.patricbrc.org/genomes/$i/$i.fna" -P ${data_dir}
    fi
done
```
