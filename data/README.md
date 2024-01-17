# AMR phenotyping benchmarking tutorial

- [1. Setup](#setup)
  - [1.1 Dependencies](#Dependencies)
  - [1.2 Installation](#Installation)
- [2. A typical ML-based methods evaluation](#evaluation1)
  - [2.1 Feature building](#feature)
  - [2.2 Nested cross-evaluation](#nCV)

## <a name="setup"></a>1. Setup
### 1.1 Dependencies
  -    Linux OS and `conda`. Miniconda2 4.8.4 was used by us
### 1.2 Installation
  To install with conda:
  ```
  conda env create -n  amr_env  -f ./install/amr_env.yml python=3.7 
  ```
## <a name="setup"></a>1. Setup
  
An example of evaluating a ML-based software that could use the scikit-learn module for training classifiers, and the feature matrix could be built without phenotype information. Among our benchmarked methods, Seq2Geno2Pheno falls into this category.

(1). <a href="https://github.com/hzi-bifo/AMR_benchmarking/blob/main/AMR_software/Pseudo/benchmarking.py"> Integrate the ML model to the evaluation framework based on instructions </a>  
(2). Run the <a href="https://github.com/hzi-bifo/AMR_benchmarking/blob/main/AMR_software/Pseudo/evaluate.py"> evaluation python scripts</a> 

```
usage: evaluate.py [-h] -software SOFTWARE_NAME -sequence PATH_SEQUENCE [-temp TEMP_PATH] [-l LEVEL] [-f_all] [-f_phylotree] [-f_kma] [-s SPECIES [SPECIES ...]] [-f_ml] [-cv CV_NUMBER]
                   [-n_jobs N_JOBS]

options:
  -h, --help            show this help message and exit
  -software SOFTWARE_NAME, --software_name SOFTWARE_NAME
                        software_name
  -sequence PATH_SEQUENCE, --path_sequence PATH_SEQUENCE
                        Path of the directory with PATRIC sequences
  -temp TEMP_PATH, --temp_path TEMP_PATH
                        Directory to store temporary files, like features.
  -l LEVEL, --level LEVEL
                        Quality control: strict or loose
  -f_all, --f_all       Benchmarking on all the possible species.
  -f_phylotree, --f_phylotree
                        phylogeny-aware evaluation
  -f_kma, --f_kma       homology-aware evaluation
  -s SPECIES [SPECIES ...], --species SPECIES [SPECIES ...]
                        species to run: e.g.'Pseudomonas aeruginosa' 'Klebsiella pneumoniae' 'Escherichia coli' 'Staphylococcus aureus' 'Mycobacterium tuberculosis' 'Salmonella enterica'
                        'Streptococcus pneumoniae' 'Neisseria gonorrhoeae'
  -f_ml, --f_ml         Perform neseted CV on ML models
  -cv CV_NUMBER, --cv_number CV_NUMBER
                        CV splits number. Default=10
  -n_jobs N_JOBS, --n_jobs N_JOBS
                        Number of jobs to run in parallel.

```
```
python AMR_software/Pseudo/evaluate.py -software 'seq2geno' -s 'Escherichia coli' -sequence '/vol/projects/patric_genome' -temp './' -f_phylotree -cv 10 -n_jobs 10
```

 
