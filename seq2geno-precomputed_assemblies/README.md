<!--
SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo

SPDX-License-Identifier: GPL-3.0-or-later
-->

# Seq2Geno

This package is an integrated tool for microbial sequence analyses. We refactored and packed the 
methods of [the published research](https://zenodo.org/record/3591847/export/hx#.YL89KyaxWV5) and
evaluated the reproducibility with the same raw data.

- [Repository structure](#structure)
- [Available functions](#functions) 
- [Download Seq2Geno and ensure the environment](#install) 
- [Usage and Input](#usage) 
    - [GUI](#gui)
    - [command line](#commandline)
    - [arguments](#args)
- [Example data and usages](#example) 
- [Train the phenotypic predictor with the Seq2Geno results](#genyml) 
    - [Start from Seq2Geno](#automatic_submission)
    - [With precomputed data](#visit_gp_server)
- [FAQ](#FAQ)
- [License](#license) 
- [Contact](#contact) 
- [Citation](#citation) 

### <a name="structure"></a>Repository structure
This repository includes:
- install: information and scripts for installation
- examples: the example data ([the tutorial](#example) )
- main: the scripts for user interface and the calling methods of core
  workflows
- snps: the scripts for generating SNPs table
- denovo: the scripts for computing de novo assemblies and the gene
  presence/absence table
- expr: the scripts for calculating expression levels
- phylo: the scripts for phylogenetic tree inference
- difexpr: the methods for identifying differentially expressed genes with the
  expression levels matrix
- cont\_anc\_rcn: ancestral reconstruction for continuous data such as expression levels


### <a name="functions"></a>Available functions
- detect single nucleotide variants
- create de novo assemblies
- compute gene presence/absence and the indels
- count gene expression levels
- infer the phylogenetic tree
- find differentially expressed genes (additional data that won't be used by Geno2Pheno)
- reconstruct ancestral values of expression level (additional data that won't be used by Geno2Pheno)

### <a name="install"></a>Download Seq2Geno and ensure the environment
1. Check the prerequisites

    - [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) (tested version: 4.10.0)
    - file [.condarc](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-channels.html) that includes these channels and is detectable by your conda
      - hzi-bifo
      - conda-forge/label/broken
      - bioconda
      - conda-forge
      - defaults
    - [python](https://www.python.org/downloads/) (tested verson: 3.7)
    - [Linux](https://www.cyberciti.biz/faq/find-linux-distribution-name-version-number/) (tested version: Debian GNU/Linux 8.8 jessie)
    - [git](https://git-scm.com/downloads) (tested version: 2.21)

2. Download this package

```
git clone --recurse-submodules https://github.com/hzi-bifo/seq2geno.git
cd seq2geno
git submodule update --init --recursive
```

The command uses **--recurse-submodules** to download the submodules. The flag is available only in git version >2.13. Earlier git versions might have the substitute. After the package is downloaded, *main/seq2geno* and *main/seq2geno_gui* are the executable scripts for Seq2Geno. 

3. To ensure the environment and test the package, please either follow the steps in *install/README.md* or use the automatic tools:

```
cd install/
SETENV.sh snakemake_env
conda activate snakemake_env
TESTING.sh
```

### <a name="usage"></a>Usage and Input

Once the environment is properly set up, Seq2Geno can be launched using the [graphical user interface (GUI)](#gui) or [command line](#commandline) 

- <a name="gui"></a>GUI

The commands 

```
S2G
``` 
or 

```
seq2geno_gui
``` 

will launch the graphic user interface. Use the tool to read, edit, or save the arguments in a yaml file. Once the arguments are ready, the analyses can be launched with this interface; for large-scale researches, however, generating the yaml file and launching the analyses with the command line method (described below) might be more convenient, as having processes running in the background should be more convenient. To learn more, please read the manual *doc/GUI_manual.pdf*.

- <a name="commandline"></a>command line

```
S2G -d -f [options_yaml] -z [zip_input] -l [log_file] --outzip [output_zip_type] --to_gp
```

Both **options_yaml** and **zip_input** specify the materials to use (see the *examples/*). At least one of them should be used. When **options_yaml** is properly set, **zip_input** will be neglected. The **options_yaml** describes all the options and paths to input data for Seq2Geno. The **zip_input** packs all the materials and has a structure that Seq2Geno can recognize (see *input_zip_structure.md* for more details). 

The **log_file** should be a non-existing filename to store the log information; if not set, the messages will be directed to stdout and stderr. 

The **output_zip_type** should be one of 'none' (default), 'all', 'main', or 'g2p'. The choice specifies whether or how the output results should be packed into an zip file.

The flag `--to_gp` specifies whether to submit the results to the Geno2Pheno server.

- <a name="args"></a>arguments

The input file is an yaml file where all options are described (a template in examples/). The file includes two parts:

1. features:

| option | action | values ([default])|
| --- | --- | --- |
| dryrun | display the processes and exit | [Y]/N |
| snps | SNPs calling | Y/[N] |
| denovo | creating de novo assemblies | Y/[N] |
| expr | counting expression levels | Y/[N] |
| phylo | inferring the phylogeny | Y/[N] |
| de | differential expression | Y/[N] |
| ar | ancestral reconstruction of expression levels | Y/[N] |

To only create the folder and config files, please turn off the last six options. 

2. general (\* mandatory): 

    - cores: number of cpus (integer; automatically adjusted if larger than the available cpu number)

    - mem_mb: memory size to use (integer in mb; automatically adjusted if larger than the free memory). __Note: some processes may crush because of insufficiently allocated  memory__

    - \*wd: the working directory. The intermediate and final files will be stored under this folder. The final outcomes will be symlinked to the sub-directory RESULTS/.

    - \*dna_reads: the list of DNA-seq data 

    It should be a two-column list, where the first column includes all samples and the second column lists the __paired-end reads files__. The two reads file are separated by a comma. The first line is the first sample.
    ```
    sample01	/paired/end/reads/sample01_1.fastq.gz,/paired/end/reads/sample01_2.fastq.gz
    sample02	/paired/end/reads/sample02_1.fastq.gz,/paired/end/reads/sample02_2.fastq.gz
    sample03	/paired/end/reads/sample03_1.fastq.gz,/paired/end/reads/sample03_2.fastq.gz
    ```

    - \*ref_fa, ref_gff, ref_gbk: the data of reference genome

    The fasta, gff, and genbank files of a reference genome. They should have same sequence ids. 


    - old_config: if recognizable, the config files that were previously stored in the working directory will be reused. ('Y': on; 'N': off)

    - rna_reads: the list of RNA-seq data. (string of filename)

    It should be a two-column list, where the first column includes all samples and the second column lists the __short reads files__. The first line is the first sample.
    ```
    sample01	/transcription/reads/sample01.rna.fastq.gz
    sample02	/transcription/reads/sample02.rna.fastq.gz
    sample03	/transcription/reads/sample03.rna.fastq.gz
    ```

    - assemblies: the list of precomputed assemblies data. (string of filename)

    It should be a two-column list, where the first column includes all samples and the second column lists the __genome sequence files__. For the samples that are included in the reads list but not in this assemblies list, Seq2Geno will compute the *de novo* assemblies for them. The first line of the list is the first sample.
    ```
    sample01	/denovo/assemblies/sample01.fa
    sample02	/denovo/assemblies/sample02.fa
    sample03	/denovo/assemblies/sample03.fa
    ```

    - phe_table: the phenotype table (string of filename)

    The table is tab-separated. For n samples with m phenotypes, the table is (n+1)-by-(m+1) as shown below. The first column should be sample names. The header line should includes names of phenotypes. Missing values are acceptable.
    ```
    strains	virulence
    sample01	high
    sample02	mediate
    sample03	low
    ```

    - adaptor: the adaptor file (string of filename)

    The fasta file of adaptors of DNA-seq. It is used to process the DNA-seq reads. 


### <a name="example"></a>Example data and usages
The folder *examples/* includes a structured zip file and a yaml file--the two input formats that Seq2Geno can recognize. The zip file can be used as the input with this command:

```
S2G -z examples/example_input.zip\
 -l examples/example_input_zip.log\
 --outzip g2p
```

To use the configuration yaml file, please ensure unpacked example data (that
is, the zip file) and edit the yaml file to ensure the right paths to
those example data. After they are ready, please run with this command:

```
S2G -f examples/seq2geno_input.yml\
 -l exapmles/seq2geno_input_yml.log\
 --outzip g2p
```

### <a name="genyml"></a>Train the phenotypic predictor with the Seq2Geno results 
- <a name="automatic_submission"></a> Start from Seq2Geno

To include automatic submission to the Geno2Pheno server, just use the flag
`--to_gp`:

```
S2G -f examples/seq2geno_input.yml\
 -l exapmles/seq2geno_input_yml.log\
 --outzip g2p --to_gp
```

- <a name="visit_gp_server"></a> With precomputed data

The precomputed data are unnecessarily generated using Seq2Geno as long as they meet the [correct formats](https://github.com/hzi-bifo/Geno2PhenoClient). 
Please directly visit the [Geno2Pheno server](https://genopheno.bifo.helmholtz-hzi.de).


### <a name="FAQ"></a>FAQ
__Why the analyses crushed?__

Please check the log file or STDOUT and STDERR and determine the exact error. 

__Will every procedure be rerun if I want to add one sample?__

No, you just need to add one more line in your reads list (i.e., the dna or the rna reads. See section [arguments](#args) for more details.) and then run the same workflow again. Seq2Geno uses Snakemake to determine whether certain intermediate data need to be recomputed or not. 

__Will every procedure be rerun if I want to exclude one sample?__

No; however, besides excluding that sample from the reads list, you will need to remove all the subsequent results that were previously computed. That could be risky.

__Will every procedure be rerun if I accidentally delete some data?__

No, only the deleted one and the subsequent data will be recomputed.

__Where is the final data?__

In the working directory, the main results are collected in the subfolder `RESULTS/`. You can also find the other intermediate data in the corresponding folders (e.g., mapping results)

__What is the current status?__

If the log file was specified, you could check the log file to determine the current status. Otherwise, the status should be directed to your STDOUT or STDERR.

### <a name="license"></a>License
GPLv3 (please refer to LICENSE)

### <a name="contact"></a>Contact
Please contact Tzu-Hao Kuo (Tzu-Hao.Kuo@helmholtz-hzi.de) and specify:
- the error message or unexpected situation
- how to reproduce the problem

### <a name="citation"></a>Citation
We will be publishing the paper for the joint work of Seq2Geno and Geno2Pheno.
Before that, please use 

```
Kuo, T.-H., Weimann, A., Bremges, A., & McHardy, A. C. (2021). Seq2Geno (v1.00001) [A reproducibility-aware, integrated package for microbial sequence analyses].
```
or 
```
@software{thkuo2021seq2geno,
  author = {Tzu-Hao Kuo, Aaron Weimann, Andreas Bremges, Alice C. McHardy},
  title = {Seq2Geno: a reproducibility-aware, integrated package for microbial sequence analyses},
  version = {v1.00001},
  date = {2021-06-20},
}
```
