# Data preprocessing tutorial
Welcome to the tutorial on data preprocessing. This tutorial guides you through the procedures for creating <a href="https://github.com/hzi-bifo/AMR_benchmarking/wiki/Dataset-overview">78 AMR phenotyping benchmarking datasets</a> from the <a href="https://www.bv-brc.org/">PATRIC</a> genome database.

## Table of Contents
- [1. Download metadata from PATRIC FTP](#1) 
- [2. Filter species and antibiotic](#2)
- [3. Download genome quality information](#2)
- [4. Filter genomes](#4)
- [5. Filter datasets](#5)
- [6. Download genome sequences from the PATRIC database](#6)

## <a name="1"></a>1. Download metadata from PATRIC FTP
- Download `PATRIC_genomes_AMR.txt` from https://docs.patricbrc.org/user_guides/ftp.html or find a <a href="https://github.com/hzi-bifo/AMR_benchmarking/blob/main/data/PATRIC/PATRIC_genomes_AMR.txt">version</a> downloaded by us in Dec 2020


## <a name="2"></a>2. Filtering species and antibiotic

 to species and antibiotic filtering based on phenotype metadata availability

## <a name="3"></a>3. Download genome quality information
- Download quality attribute tables for the 13 selected species from Step 2

	- Example: download the <em>E. coli</em> genome quality attributes from PATRIC database
```console
p3-all-genomes --eq genus,Escherichia --eq species,coli -a genome_name,genome_status,genome_length,genome_quality,plasmids,contigs,fine_consistency,coarse_consistency,checkm_completeness,checkm_contamination >  Escherichia_coli.csv
```
- Alternatively, find <a href="https://github.com/hzi-bifo/AMR_benchmarking/tree/main/data/PATRIC/quality">versions</a> downloaded by us around Dec 2020

## <a name="4"></a>4. Filter genomes

genome quality




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
