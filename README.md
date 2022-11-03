# Benchmarking machine learning-based software for phenotypic antimicrobial resistance determination from genomic data

## Usage Contents
- [Introduction](#intro)
    - [software list](#software)
    - [Data sets](#data)
    - [Framework](#frame)
- [Prerequirements](#pre)
- [Input](#input)
- [Output](#output)
- [Usage](#usage)  
- [References](#ref)   
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)


## <a name="intro"></a>Introduction
### software list

We compare the binary phenotype prediction performance of four machine learning (ML)- based and one direct association antimicrobial resistance (AMR) determination sofware:
1. [Aytan-Aktug](https://bitbucket.org/deaytan/neural_networks/src/master/) [[1]](#1), 
2. Seq2Geno2Pheno([Seq2Geno](https://github.com/hzi-bifo/seq2geno.git)&[Geno2Pheno](https://galaxy.bifo.helmholtz-hzi.de/galaxy/root?tool_id=genopheno)) [[2]](#2), 
3. [PhenotypeSeeker 0.7.3](https://github.com/bioinfo-ut/PhenotypeSeeker) [[3]](#3), 
4. [Kover 2.0](https://github.com/aldro61/kover) [[4]](#4). 
5. [Point-/ResFinder 4.0](https://bitbucket.org/genomicepidemiology/resfinder/src/master/) [[5]](#5), a direct association software based on AMR determinant database, was used as the baseline.

### <a name="data"></a>Data sets

- <a href="https://github.com/hzi-bifo/AMR_benchmarking/wiki/Species-and-antibiotics">Data sets overview</a>
- Sample list of each data set in the form of `Data_<species>_<antibiotic>` and sample phenotype metadata of each data set `Data_<species>_<antibiotic>_pheno.txt` under the folder <a href="https://github.com/hzi-bifo/AMR_benchmarking/main/data/PATRIC/meta/loose_by_specie">data/PATRIC/meta/loose_by_species</a>
- <a href="https://github.com/hzi-bifo/AMR_benchmarking/data/PATRIC/cv_folds/loose/">Corss-validation folds</a> was generated through Aytan-Aktug (homology-aware folds), Seq2Geno2Pheno(phylogeny-aware and random folds, except for  *M. tuberculosis* folds)), and sklearn package model_selection.KFold
(random folds for  *M. tuberculosis* folds).

###  <a name="frame"></a>Framework

![alt text](./doc/workflow.png)




## <a name="pre"></a>Prerequirements
**To reproduce the output, you need to use `conda` , Miniconda 4.8.3 was used by us.**
-  Create conda env, and install packages.

```
bash ./install/install.sh
```
-  Install pytorch in the `multi_torch_env` manually if GPU is available.

To install pytorch compatible with your CUDA version, please fellow this instruction: https://pytorch.org/get-started/locally/. Our code was tested with pytorch v1.7.1, with CUDA Version: 10.1 and 11.0 .

- Memory requirment: Some procedures require extremely large memory. Aytan-Aktug multi-species model (adaption version) feature building needs ~370G memory. Other ML software needs up to 80G memory, depending on the number of CPU set and specis-antibiotic combination.

- Disk storage requirement: Some procedures generate extremely large intermediate files, although they are deleted once the procedures generate features files. PhenotypeSeeker (adaption version) needs up to on the magnitude of 10T depending on the data set size of different species. 


## <a name="input"></a>Input file
The input file is an yaml file `Config.yaml` at the root folder where all options are described:

**A. Basic/required parameters setting**

- Please do change everything in A after the ":" to your own.

| option | action | values ([default])|
| ------------- | ------------- |------------- |
|dataset_location| To where the PATRIC dataset will be downloaded. ~246G| /vol/projects/BIFO/patric_genome|
|output_path| To where to generate the `Results` folder for the direct results of each software and further visualization. | ./|
|log_path| To where to generate the `log` folder for the tempary files, which you can delete by hand afterwards. Large temp files are stored under `<log_path>/log/software/<software_name>/software_output`. Running benchmarking study scripts from beginning to the end will generate temp files up to the order of 10 terabytes, which means you are suggested to delete temp files via `./src/software_utility/clean.py` as soon as one software finishes evaluation successfully, except Point-/ResFinder. | ./|
|n_jobs| CPU cores to use. | 10 |
|gpu_on| GPU possibility for Aytan-Aktug SSSA model, If set to False, parallelization on cpu will be applied; Otherwise, it will be applied on one gpu core sequentially.  | False |
|kover_location| Kover install path  | /vol/projects/khu/amr/kover/bin/ |

**B. Adanced/optional parameters setting**

- Please change the conda environment names if the same names already exist in your working PC.

|option|	action	|values ([default])|
| ------------- | ------------- |------------- |
|amr_env_name|conda env for general use |amr_env|
|PhenotypeSeeker_env_name|conda env for PhenotypeSeeker |PhenotypeSeeker_env|
|multi_env_name|conda env for |multi_env|
|multi_torch_env_name|conda env for NN model|multi_torch_env|
|kover_env_name|conda env for NN model|kover_env|
|se2ge_env_name|conda env for Seg2Geno|snakemake_env|

**C. Adanced/optional parameters setting (Model)**
 
- Users, who would like to reproduce this AMR benchmarking results, are not advised to change settings in this category. 
 
- You can change them accordingly when you want to make use of this benchamrking software to explore more. For species related settings, we have listed the possible maxium (in terms of data sets this study provides) for each setting, so you can explore by reducing the species, but not by adding others on.
 
|option|	action	|values ([default])|
| ------------- | ------------- |------------- |
|QC_criteria|Sample qaulity control level. Can be loose or strict.| loose|
|species_list|species to be included in for random and homology-aware folds for the five software tools (Aytan-Aktug single-species-antibiotic model)|Escherichia_coli, Staphylococcus_aureus, Salmonella_enterica, Klebsiella_pneumoniae, Pseudomonas_aeruginosa, Acinetobacter_baumannii, Streptococcus_pneumoniae, Mycobacterium_tuberculosis, Campylobacter_jejuni, Enterococcus_faecium, Neisseria_gonorrhoeae|
|species_list_phylotree|species to be included in for phylogeny-aware folds for the five software tools (Aytan-Aktug single-species-antibiotic model)|Escherichia_coli, Staphylococcus_aureus, Salmonella_enterica, Klebsiella_pneumoniae, Pseudomonas_aeruginosa, Acinetobacter_baumannii, Streptococcus_pneumoniae, Campylobacter_jejuni, Enterococcus_faecium, Neisseria_gonorrhoeae|
|species_list_multi_antibiotics|species to be included in for Aytan-Aktug single-species multi-antibiotic model. |Mycobacterium_tuberculosis, Escherichia_coli, Staphylococcus_aureus, Salmonella_enterica, Klebsiella_pneumoniae, Pseudomonas_aeruginosa, Acinetobacter_baumannii, Streptococcus_pneumoniae, Neisseria_gonorrhoeae|
|species_list_multi_species|species to be included in for three variants of Aytan-Aktug multi-species multi-antibiotic models. For user defining species combinations for MSMA, please change species names here and replace -f_all with -s "${species[@]}" in ./scripts/model/AytanAktug_MSMA_concat.sh and ./scripts/model/AytanAktug_MSMA_discrete.sh|Mycobacterium_tuberculosis, Salmonella_enterica, Streptococcus_pneumoniae, Escherichia_coli, Staphylococcus_aureus, Klebsiella_pneumoniae, Acinetobacter_baumannii, Pseudomonas_aeruginosa, Campylobacter_jejuni|
|merge_name| used to notate the folders for saving the results of three Aytan-Aktug multi-species multi-antibiotic models. Always takes such form of a concatenation of species names in order. E.g. only two species of Mycobacterium_tuberculosis and Salmonella_enterica will result in Mt_Se.|Mt_Se_Sp_Ec_Sa_Kp_Ab_Pa_Cj|
|cv_number|the k of k-fold nested cross-validation for the five software tools (Aytan-Aktug single-species-antibiotic model) and Aytan-Aktug single-species multi-antibiotic model|10|
|cv_number_multiS|k+1, where the k corresponds to k-fold cross-validation and 1 correspons to the hold out test set, for three variants of Aytan-Aktug multi-species multi-antibiotics models)|6|



## <a name="output"></a>Output
```
└── Results
    ├── final_figures_tables
    ├── other_figures_tables
    ├── supplement_figures_tables    
    └── software
        ├── AytanAktug
        ├── kover
        ├── majority
        ├── phenotypeseeker
        ├── resfinder_b
        ├── resfinder_folds
        ├── resfinder_k
        └── seq2geno

```

-  Cross-validation results of each ML software and evaluation results of Point-/Resfinder are generated under `output_path/Results/<name of the software>`.
- Visualization tables and graphs are generated under `output_path/Results/final_figures_tables` and `output_path/Results/supplement_figures_tables`.
- Numbers and statistic results mentioned in our benchmarking article are generated under `output_path/Results/other_figures_tables`.
- Stochastic factors in generating results:  [KMA](https://bitbucket.org/genomicepidemiology/kma/src/master/) version, which should be installed manualy by ursers (installation commands attached), and Neural networks dropout mechanism.

## <a name="usage"></a>Usage
```
git clone https://github.com/hzi-bifo/AMR_benchmarking.git
bash ./install/install.sh
bash main.sh
```

- The script `main.sh` goes through the whole benchmarking process from data reprocessing to running software, to benchmarking results visualization. 


- You can refer to `main.sh` and `./scripts/model/<software_name>.sh` for details of  . But you can't finishe this AMR benchamrking study just by submitting `main.sh` to run once and for all for several reasons. You have to access Geno2Pheno website using the feature generated by Seq2Geno. Due to large data sets and time consuming ML model learning process, which altogether may take more than 2 months with 20 CPUs accompanied by 10 GPUs, you may need to run different tasks on different machines and re-run some processes if it accidently terminates unexpectedly during a long periord of time.
- Clean intermediate files: we provide a script to clean large and unimportant intermeidate files. It will skan several predined locations for the targets, and delete them forever. You can run it any time after a corresponding software finishe running (on part of species in the list). Make sure you don't need those intermediate files for debugging before cleaning it.

```
python ./src/software_utility/clean.py
```
#TODO


## <a name="ref"></a>References
<a id="1">[1]</a>  D Aytan-Aktug, Philip Thomas Lanken Conradsen Clausen, Valeria Bortolaia, Frank Møller Aarestrup, and Ole Lund. Prediction of acquired antimicrobial resistance for multiple bacterial species using neural networks.Msystems, 5(1), 2020.

<a id="2">[2]</a>   Ariane Khaledi, Aaron Weimann, Monika Schniederjans, Ehsaneddin Asgari, Tzu-Hao Kuo, Antonio Oliver, Gabriel Cabot, Axel Kola, Petra Gastmeier, Michael Hogardt, et al. Predicting antimicrobial resistance in pseudomonas aeruginosa with machine learning-enabled molecular diagnostics. EMBO molecular medicine, 12(3):e10264, 2020.

<a id="3">[3]</a>  Erki Aun, Age Brauer, Veljo Kisand, Tanel Tenson, and Maido Remm. A k-mer-based method for the identification of phenotype-associated genomic biomarkers and predicting phenotypes of sequenced bacteria. PLoS computational biology, 14(10):e1006434, 2018.

<a id="4">[4]</a> Alexandre Drouin, Gaël Letarte, Frédéric Raymond, Mario Marchand, Jacques Corbeil, and François Laviolette. Interpretable genotype-to-phenotype classifiers with performance guarantees. Scientific reports, 9(1):1–13, 2019.

<a id="5">[5]</a>    Valeria Bortolaia, Rolf S Kaas, Etienne Ruppe, Marilyn C Roberts, Stefan Schwarz, Vincent Cattoir, Alain Philippon, Rosa L Allesoe, Ana Rita Rebelo, Alfred Ferrer Florensa, et al. Resfinder 4.0 for predictions of phenotypes from genotypes. Journal of Antimicrobial Chemotherapy, 75(12): 3491–3500, 2020.


## <a name="license"></a> License

## <a name="citation"></a> Citation

## <a name="contact"></a> Contact
- Open an  [issue](https://github.com/hzi-bifo/AMR_benchmarking/issues) in the repository.
- Send an email to Kaixin Hu (Kaixin.Hu@helmhotz-hzi.de).
