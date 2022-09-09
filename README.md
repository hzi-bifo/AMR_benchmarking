# Benchmarking machine learning-based software for phenotypic antimicrobial resistance determination from genomic data

### Introduction
We compare the binary phenotype prediction performance of four machine learning (ML)- based and one direct association antimicrobial resistance (AMR) determination sofware:
1. [Aytan-Aktug](https://bitbucket.org/deaytan/neural_networks/src/master/) [[1]](#1), 
2. Seq2Geno2Pheno([Seq2Geno](https://github.com/hzi-bifo/seq2geno.git)&[Geno2Pheno](https://galaxy.bifo.helmholtz-hzi.de/galaxy/root?tool_id=genopheno)) [[2]](#2), 
3. [PhenotypeSeeker 0.7.3](https://github.com/bioinfo-ut/PhenotypeSeeker) [[3]](#3), 
4. [Kover 2.0](https://github.com/aldro61/kover) [[4]](#4). 
5. [Point-/ResFinder 4.0](https://bitbucket.org/genomicepidemiology/resfinder/src/master/) [[5]](#5), a direct association software based on AMR determinant database, was used as the baseline.

### Data sets

- <a href="https://github.com/hzi-bifo/AMR_benchmarking_khu/wiki/Species-and-antibiotics">Data sets overview</a>
- Sample list of each data set in the form of `Data_<species>_<antibiotic>` and sample phenotype metadata of each data set `Data_<species>_<antibiotic>_pheno.txt` under the folder <a href="https://github.com/hzi-bifo/AMR_benchmarking_khu/main/data/PATRIC/meta/loose_by_specie">data/PATRIC/meta/loose_by_species</a>
- Corss-validation folds: TODO

### Framework

![alt text](./doc/workflow.png)




### References
<a id="1">[1]</a>  D Aytan-Aktug, Philip Thomas Lanken Conradsen Clausen, Valeria Bortolaia, Frank Møller Aarestrup, and Ole Lund. Prediction of acquired antimicrobial resistance for multiple bacterial species using neural networks.Msystems, 5(1), 2020.

<a id="2">[2]</a>   Ariane Khaledi, Aaron Weimann, Monika Schniederjans, Ehsaneddin Asgari, Tzu-Hao Kuo, Antonio Oliver, Gabriel Cabot, Axel Kola, Petra Gastmeier, Michael Hogardt, et al. Predicting antimicrobial resistance in pseudomonas aeruginosa with machine learning-enabled molecular diagnostics. EMBO molecular medicine, 12(3):e10264, 2020.

<a id="3">[3]</a>  Erki Aun, Age Brauer, Veljo Kisand, Tanel Tenson, and Maido Remm. A k-mer-based method for the identification of phenotype-associated genomic biomarkers and predicting phenotypes of sequenced bacteria. PLoS computational biology, 14(10):e1006434, 2018.

<a id="4">[4]</a> Alexandre Drouin, Gaël Letarte, Frédéric Raymond, Mario Marchand, Jacques Corbeil, and François Laviolette. Interpretable genotype-to-phenotype classifiers with performance guarantees. Scientific reports, 9(1):1–13, 2019.

<a id="5">[5]</a>    Valeria Bortolaia, Rolf S Kaas, Etienne Ruppe, Marilyn C Roberts, Stefan Schwarz, Vincent Cattoir, Alain Philippon, Rosa L Allesoe, Ana Rita Rebelo, Alfred Ferrer Florensa, et al. Resfinder 4.0 for predictions of phenotypes from genotypes. Journal of Antimicrobial Chemotherapy, 75(12): 3491–3500, 2020.




## Contents

- [Prerequirements](#pre)
- [Input](#input)
- [Output](#output)
- [Usage](#usage)
    - [Workflow](#workflow)
    - [Workflow in steps](#workflows)
      
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)

<!--
- [Data](#data)
  - [PATRIC](#patric)
    - [Test](#test)
- [Cross-validation folders preparing](#cv)
    - [Homology-aware folds](#kma)
    - [phylogeny-aware folds split](#tree)
    - [Random folds split](#tree)
- [Experiment environment](#env)
- [Benchmarking software: Point-/ResFinder 4.0](#p)

- [Benchmarking software: Multi-species](#m)
   - [Single-species model](#single)
   -  [Single-species multi-antibiotics model](#multi-anti)
   - [Discrete multi-species multi-antibioticsmodel](#dis)
   - [Concatenated multi-species multi-antibiotics model](#con)
- [Benchmarking software: Seq2Geno2Pheno](#s2g2p)

- [Benchmarking software: PhenotypeSeeker 0.7.3](#PhenotypeSeeker)
- [Benchmarking software: Kover 2.0](#kover)
- [The output structure](#output)
-->


## <a name="pre"></a>Prerequirements
**To reproduce the output, you need to use `conda` , Miniconda 4.8.3 was used by us.**
-  Create conda env, and install packages.

```
bash ./install/install.sh
```
-  Install pytorch in the `multi_torch_env` env if GPU is available.

To install pytorch compatible with your CUDA version, please fellow this instruction: https://pytorch.org/get-started/locally/. Our code was tested with pytorch v1.7.1, with CUDA Version: 10.1 and 11.0 .


## <a name="input"></a>Input file
The input file is an yaml file `Config.yaml` at the root folder where all options are described:

**A. Basic/required parameters setting**

- Please do change everything in A after the ":" to your own.

| option | action | values ([default])|
| ------------- | ------------- |------------- |
|dataset_location| To where the PATRIC dataset will be downloaded. ~246G| /vol/projects/BIFO/patric_genome|
|output_path| To where to generate the `Results` folder for the direct results of each software and further visualization. | ./|
|log_path| To where to generate the `log` folder for the tempary files, which you can delete by hand afterwards. (note:some software will generate large amount of temp files to 60G, if you run all species in parallel.)| ./|
|n_jobs| CPU cores to use.  | 10 |
|gpu_on| GPU possibility for Aytan-Aktug SSSA model, If set to False, parallelization on cpu will be applied; Otherwise, it will be applied on one gpu core sequentially.  | False |

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
|species_list|species to be included in for random and homology-aware folds for the five software tools (Aytan-Aktug single-species-antibiotic model)|'Escherichia coli','Staphylococcus aureus','Salmonella enterica','Klebsiella pneumoniae','Pseudomonas aeruginosa','Acinetobacter baumannii','Streptococcus pneumoniae','Mycobacterium tuberculosis', 'Campylobacter jejuni','Enterococcus faecium','Neisseria gonorrhoeae']|
|species_list_phylotree|species to be included in for phylogeny-aware folds for the five software tools (Aytan-Aktug single-species-antibiotic model)|Escherichia_coli, Staphylococcus_aureus, Salmonella_enterica, Klebsiella_pneumoniae, Pseudomonas_aeruginosa, Acinetobacter_baumannii, Streptococcus_pneumoniae, Campylobacter_jejuni, Enterococcus_faecium, Neisseria_gonorrhoeae|
|species_list_multi_antibiotics|species to be included in for Aytan-Aktug single-species multi-antibiotic model. |Mycobacterium_tuberculosis, Escherichia_coli, Staphylococcus_aureus, Salmonella_enterica, Klebsiella_pneumoniae, Pseudomonas_aeruginosa, Acinetobacter_baumannii, Streptococcus_pneumoniae, Neisseria_gonorrhoeae|
|species_list_multi_species|species to be included in for three variants of Aytan-Aktug multi-species multi-antibiotic models. For user defining species combinations for MSMA, please change species names here and replace -f_all with -s "${species[@]}" in ./scripts/model/AytanAktug_MSMA_concat.sh and ./scripts/model/AytanAktug_MSMA_discrete.sh|Mycobacterium_tuberculosis, Salmonella_enterica, Streptococcus_pneumoniae, Escherichia_coli, Staphylococcus_aureus, Klebsiella_pneumoniae, Acinetobacter_baumannii, Pseudomonas_aeruginosa, Campylobacter_jejuni|
|merge_name| used to notate the folders for saving the results of three Aytan-Aktug multi-species multi-antibiotic models. Always takes such form of a concatenation of species names in order. E.g. only two species of Mycobacterium_tuberculosis and Salmonella_enterica will result in Mt_Se.|Mt_Se_Sp_Ec_Sa_Kp_Ab_Pa_Cj|
|cv_number|the k of k-fold nested cross-validation for the five software tools (Aytan-Aktug single-species-antibiotic model) and Aytan-Aktug single-species multi-antibiotic model|10|
|cv_number_multiS|the k of k-fold cross-validation for three variants of Aytan-Aktug multi-species multi-antibiotics models)|6|



## <a name="output"></a>Output

-  Cross-validation results of each ML software and evaluation results of Point-/Resfinder are generated under `output_path/Results/<name of the software>`.
- Visualization tables and graphs are generated under `output_path/Results/final_tables` and `output_path/Results/final_figures`.

## <a name="usage"></a>Usage
### A.  <a name="workflow"></a>Workflow
- The script `main.sh`, which is a combination of codes and pseudocodes, goes through the whole benchmarking process from data reprocessing to running software, to basic benchmarking results visualization. 
- The pseudocodes resulted from the reason that you have to manually install  Seq2Geno yourself and then access Geno2Pheno website. What's more, the ML model learning process is very time consuming, taking at least 2 months with 20+ CPUs accompanied by 10+ GPUs, so it's not realistic to warp all benchmarking scripts in a pipeline.
- We will guide you through the Workflow in steps:


### B.  <a name="workflows"></a>Workflow in steps
Add root address to your PATH (for every step):
```
git clone https://github.com/hzi-bifo/AMR_benchmarking_khu.git
cd AMR_benchmarking_khu
export PATH="$PWD"/src:$PATH
export PYTHONPATH=$PWD
```

**B1.  Sample phenotype availability and quality control (PQC). You can skip this step, as we provided the sample list after PQC: ./data/PATRIC/meta

```
bash ./scripts/data/preprocess.sh
```

**B2.  PATRIC dataset downloading.**

```
bash ./scripts/data_preprocess/retrive_PATRIC_data.sh
```

**B3.  Point-/Resfinder**

- No need to install from the original Point-/Resfinder source.\
  We provided the modified version of KMA-based Point-/Resfinder that can use genomic data. Origianl KMA-based version can only process read               data(FASTQ files). This is done because NN multi-species model (Aytan-Aktug et al.) can only be used accompanied with KMA-based Point-/Resfinder.\
- Optional: you can install Blastn-based version from https://bitbucket.org/genomicepidemiology/resfinder/src/master\
- Reference database version 2021-05-06

```
cd ./AMR_software/resfinder/cge
git clone https://bitbucket.org/genomicepidemiology/kma.git
cd kma && make
bash ./scripts/resfinder.sh
```

**B4. Aytan-Aktug software**
- Please use our version. We adapted their codes for our benchmarking work.
- Homology-aware(KMA-based) folds were generated through NN model.
```
bash ./scripts/nn.sh
```
  Early stopping for PyTorch https://github.com/Bjarten/early-stopping-pytorch



**B5.  Seq2Geno2Pheno**

- Please install/use from the original source. 
- The original Seq2Geno software, which deals with the original sequence, can be found here: https://github.com/hzi-bifo/seq2geno.git
Please then replace the main folder, denovo/denovo.in_one.smk file, all the yml files under denovo folder in seq2geno-precomputed_assemblies this repository.
- Install Seq2Geno according to the instruction from: https://github.com/hzi-bifo/seq2geno.git. Through this step, an conda envioronment named snakemake_env will be created.
- Phylogeny-aware folds and random folds were generated through Seq2Geno2Pheno.
```
cd ./AMR_software/
git clone --recurse-submodules https://github.com/hzi-bifo/seq2geno.git
cd seq2geno
git submodule update --init --recursive
#----------------------------------------------------------------------------
# Now, install Seq2Geno according to https://github.com/augustkx/seq2geno
#----------------------------------------------------------------------------

cp ../seq2gen_assemble/* .  # Afterwards, update the seq2geno root folds with files from this(our) repo
bash ./scripts/model/Seg2Geno.sh #Run.
```


### C. Visualization
- Compare results. TODO: list the things to compare
```
bash ./scripts/visualization/compare.sh
```

- Generates all visualization graphs and excel tables.



- As we stored 10-folds nested Cross-validation performance results as stated in our article in `./log/results`, so you can also run `python scripts` to selectively generate some of the visualization results.
```
python 
```








<!--
- Prerequirements: 

Annotate FASTA files with PROKKA

Roary –e –mafft *.gff

```
conda install -c bioconda -c conda-forge prokka
conda config --add channels r
conda config --add channels defaults
conda config --add channels conda-forge
conda config --add channels bioconda
conda install roary
pip3 install biopython
conda install -c conda-forge scikit-learn 
pip install seaborn
conda install pandas
```
- Reference:

https://github.com/microgenomics/tutorials/blob/master/pangenome.md






## <a name="env"></a> Experiment environment




## <a name="p"></a> ResFinder

Bortolaia, Valeria, et al. "ResFinder 4.0 for predictions of phenotypes from genotypes." Journal of Antimicrobial Chemotherapy 75.12 (2020): 3491-3500.

### 1. Install ResFinder 4.0 from:

https://bitbucket.org/genomicepidemiology/resfinder/src/master/

Database version: 2021.May. 6th
To use the reference database version we used, please unzip the database.zip, otherwise you can follow the instructions of ResFinder 4.0 to download the latest version.

### 2. Preparing.

Copy the contents of /ResFinder to installed ResFinder 4.0 Folder, i.e. /resfinder.

Copy the contents of /Patric_data to installed ResFinder 4.0 Folder, i.e. /resfinder.

### 3. SNP and AMR gene information extraqction.
```
usage: Kaixin_ResFinder_PointFinder.py [-h] [--s S [S ...]] [--n_jobs N_JOBS]
                                       [--check]

optional arguments:
  -h, --help            show this help message and exit
  --s S [S ...], --species S [S ...]
                        species to run: e.g.'seudomonas aeruginosa'
                        'Klebsiella pneumoniae' 'Escherichia coli'
                        'Staphylococcus aureus' 'Mycobacterium tuberculosis'
                        'Salmonella enterica' 'Streptococcus pneumoniae'
                        'Neisseria gonorrhoeae'
  --n_jobs N_JOBS       Number of jobs to run in parallel.
  --check               debug

```
### 4. AMR determination testing score.

```
usage: Kaixin_Predictions_Res_PointFinder_tools.py [-h] --l L --t T
                                                   [--s S [S ...]] [-v]
                                                   [--score SCORE]

optional arguments:
  -h, --help            show this help message and exit
  --l L, --level L      Quality control: strict or loose
  --t T, --tool T       res, point, both
  --s S [S ...], --species S [S ...]
                        species to run: e.g.'seudomonas aeruginosa'
                        'Klebsiella pneumoniae' 'Escherichia coli'
                        'Staphylococcus aureus' 'Mycobacterium tuberculosis'
                        'Salmonella enterica' 'Streptococcus pneumoniae'
                        'Neisseria gonorrhoeae'
  -v, --visualize       visualize the final outcome
  --score SCORE         Score:f1-score, precision, recall, all. All scores are
                        macro.
Namespace(l='loose', s=['Escherichia coli'], score='all', t='both', v=True)

```

### 5. Example.
```
python ./data_preparation/Kaixin_ResFinder_PointFinder.py  --n_jobs=1 --s 'Escherichia coli'

python Kaixin_Predictions_Res_PointFinder_tools.py --s 'Escherichia coli'

```
or 

```
python ./data_preparation/Kaixin_ResFinder_PointFinder_kma.py  --n_jobs=1 -f_all

python Kaixin_Predictions_Res_PointFinder_tools.py -f_all
```


## <a name="m"></a> Multi-species
D Aytan-Aktug, Philip Thomas Lanken Conradsen Clausen, Valeria Bortolaia, Frank Møller Aarestrup, and Ole Lund. Prediction of acquired antimicrobial resistance for multiple bacterial species using neural networks.Msystems, 5(1), 2020.

Version: 2021-04-26

### <a name="Prerequirements"></a>Prerequirements

Pytorch version can be find here: https://pytorch.org/get-started/previous-versions/ . You have to select the right version according to the cuda version in your system. We use the the version 1.7.1 , with CUDA Version: 10.1. 

```
conda create -n multi_bench python=3.6
conda activate multi_bench
pip install sklearn numpy pandas seaborn 
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
```

### <a name="References"></a>References

https://bitbucket.org/deaytan/data_preparation (Version: 2021-04-26)

https://bitbucket.org/deaytan/neural_networks/src/master/ (Version: 2019-12-19)

https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py (version: Aug 4, 2020)

### <a name="single"></a>Single-species model

```
python main_feature.py -l "loose" -f_cluster  -s 'Salmonella enterica'
python main_feature.py -l "loose" -f_pre_cluster --n_jobs 9 -s 'Salmonella enterica'
bash ./cv_folders/loose/Salmonella_enterica_kma.sh
python main_feature.py -l "loose" -f_res --n_jobs 14 -s 'Salmonella enterica'
python main_feature.py -l "loose" -f_merge_mution_gene --n_jobs 14 -s 'Salmonella enterica'
python main_feature.py -l "loose" -f_matching_io --n_jobs 14 -s 'Salmonella enterica'
python main_feature.py -f_nn -f_optimize_score 'auc' -learning 0.0 -e 0 -s 'Salmonella enterica'
python main_feature.py -f_nn -f_optimize_score 'f1_macro' -learning 0.0 -e 0 -s 'Salmonella enterica'
python main_feature.py -f_nn -f_fixed_threshold -f_optimize_score 'f1_macro' -learning 0.0 -e 0 -s 'Salmonella enterica'

```




### <a name="dis"></a>Discrete multi-species model

```
python main_discrete_merge.py -f_all  -f_pre_meta
python main_discrete_merge.py -f_all -f_pre_cluster #note: this is merege sequence to one file
for file in cv_folders/loose/multi_species/Mt_Se_Sp_Ec_Sa_Kp_Ab_Pa_Cj/*.sh; do qsub $file; done
python main_discrete_merge.py -f_cluster -f_phylo_roary -f_all #not need redo
python main_discrete_merge.py -f_res -f_merge_mution_gene -f_all
python main_discrete_merge.py -f_matching_io -f_all 

```

```
python  main_discrete_merge.py  -f_nn -f_optimize_score 'f1_macro' -learning 0.0 -e 0 -f_all 
python main_discrete_merge.py  -f_nn -f_optimize_score 'f1_macro' -f_fixed_threshold -learning 0.0 -e 0 -f_all 
python main_discrete_merge.py  -f_nn -f_optimize_score 'auc' -learning 0.0 -e 0 -f_all 

```

### <a name="con"></a>Concatenated multi-species model
1. Merge reference sequences of all the species in db_pointfinder

```
cd ResFinder
python merge_database.py
```

2. Index the newly merged database

Add "merge_species" to the config file under /db_pointfinder, and then index the database with kma_indexing:

```
cd db_pointfinder
python3 INSTALL.py non_interactive
```
3. Run the Resfinder tool with merged database

```
cd ../../
python main_concatenate_merge.py  -n_jobs 20 -f_all -f_run_res
```
4. Neural network validation
```
python main_concatenate_merge.py -f_all  -f_pre_meta
python main_concatenate_merge.py  -f_res  -f_all 
python main_concatenate_merge.py  -f_matching_io  -f_all 
python main_concatenate_merge.py -f_all -f_divers_rank
for file in  cv_folders/loose/multi_concat/Mt_Se_Sp_Ec_Sa_Kp_Ab_Pa_Cj/*.sh; do qsub $file; done
python main_concatenate_merge.py -f_all -f_cluster
```

```
python main_concatenate_merge.py -f_all -f_nn_all_io -f_nn_all -f_optimize_score 'f1_macro' -learning 0.0 -e 0 
python main_concatenate_merge.py -f_all -f_nn_all -f_optimize_score 'f1_macro' -f_fixed_threshold  -learning 0.0 -e 0 
python main_concatenate_merge.py -f_all -f_nn_all -f_optimize_score 'auc' -learning 0.0 -e 0 #on 024, June 13th , qsub. Stoped. June 20th  
python main_concatenate_merge.py -f_all -f_nn_all -f_optimize_score 'f1_macro' -learning 0.0 -e 0 #025 June 12th. Stoped. June 20th 

python main_concatenate_merge.py -f_nn -f_optimize_score 'f1_macro' -f_fixed_threshold  -learning 0.0 -e 0 -f_all
python main_concatenate_merge.py -f_nn -f_optimize_score 'f1_macro' -learning 0.0 -e 0 -f_all 
python main_concatenate_merge.py -f_nn -f_optimize_score 'auc' -learning 0.0 -e 0 -f_all

```

## <a name="s2g2p"></a>Seq2Geno2Pheno

Kuo, T.-H., Weimann, A., Bremges, A., & McHardy, A. C. (2021). Seq2Geno (v1.00001) [A reproducibility-aware, integrated package for microbial sequence analyses].

### <a name="install"></a>Installment of Seq2Geno
version: Jul 11, 2021.

1. The original Seq2Geno software, which deals with the original sequence, can be found here: https://github.com/hzi-bifo/seq2geno.git
2. Please then replace the main folder, denovo/denovo.in_one.smk file, all the yml files under denovo folder in seq2geno-precomputed_assemblies this repository.
3. Install Seq2Geno according to the instruction from: https://github.com/hzi-bifo/seq2geno.git. Through this step, an conda envioronment named snakemake_env will be created.
<!--
2.In order to use the seq2geno software to process assembled sequences instead of raw sequences, we'll use a branch of it, which deals with assembled data: https://github.com/hzi-bifo/seq2geno/tree/precomputed_assemblies
Update the scripts:
```
git submodule update --init
git fetch --prune
git reset --hard HEAD
git pull origin precomputed_assemblies
```



### <a name="s2g"></a>Seq2Geno


Prepare the files for Seq2Geno:
```
source activate multi_bench
python main_s2p.py -f_prepare_meta -f_all
conda deactivate
````
Run the Seq2Geno tool:

```
source activate snakemake_env
bash log/temp/loose/Pseudomonas_aeruginosa/run.sh 

```
Make a phylo-tree for each species
```
source activate multi_bench_phylo
python main_s2p.py -f_tree -f_all
```
Clear the large tempt folders
```
source activate multi_bench_phylo
python main_s2p.py -f_finished -f_all

```




### <a name="g2p"></a>Geno2Pheno
Not open source:
https://galaxy.bifo.helmholtz-hzi.de/galaxy/root?tool_id=genopheno

Version: Nov 26.2021.


## <a name="PhenotypeSeeker"></a>PhenotypeSeeker

Scripts written by us, based on version: 0.7.3.
https://github.com/bioinfo-ut/PhenotypeSeeker

### <a name="install_pts"></a> Install
```
conda create -n PhenotypeSeeker python=3.7
source activate PhenotypeSeeker
conda install -c bioconda mash
conda install -c bioconda genometester4

```


### <a name="c_pts"></a> Commands example
1. Prepare meta files.

```
python  python main_bench.py -s 'Campylobacter jejuni' -f_prepare_meta
```


2. Generate k-mer countings
environment: PhenotypeSeeker

```
bash kmer.sh
```

3. Get kmer results w.r.t. feature space for each sample. Use mesh and PyCodent(chi-squared_test) to calculate weigts.
4. k-mer filtering on training set according to the Chi-squared test; k-mer filtering on testing set according to the training set.
5. prepare training and testing set for nested CV.
```
bash map.sh "Campylobacter_jejuni"
```


6. ML evaluation using nested CV.

```
python main_bench.py -f_ml --n_jobs 10 -s 'Campylobacter jejuni' -f_kma

```


## <a name="kover"></a> Kover 2.0

Version 2.0
https://github.com/aldro61/kover

### <a name="install_kover"></a> Install

Install according to the documentations in Kover 2.0. https://aldro61.github.io/kover/doc_installation.html



### <a name="c_kover"></a> Commands example
1. Prepare the meta files to run Kover 2.0.
```
python main_bench.py -s 'Campylobacter jejuni' -f_prepare_meta
```


2. Use the Kover 2.0 to generate performance results for each of 10 folders using risk bound selection.
```
bash run_cv.sh "Pseudomonas_aeruginosa" 
```

3. Summerize and visualize the results.

 ```
python cv_folder_checking.py -s 'Campylobacter jejuni'
python kover_analyse.py -s 'Campylobacter jejuni' -f_kma
python kover_analyse.py -s 'Campylobacter jejuni' -f_kma -f_benchmarking
 ```

-->


## <a name="license"></a> License

## <a name="citation"></a> Citation

## <a name="contact"></a> Contact
