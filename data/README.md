# Datasets Usage

## 1. Single-species-antibiotic dataset usage

- <a href="https://github.com/hzi-bifo/AMR_benchmarking/tree/main/data/PATRIC/meta/loose_by_species">Sample list</a>  of each species-antibiotic combination in the form of `Data_<species>_<antibiotic>`
- <a href="https://github.com/hzi-bifo/AMR_benchmarking/tree/main/data/PATRIC/meta/loose_by_species">Sample phenotype metadata</a> of each dataset in the form of `Data_<species>_<antibiotic>_pheno.txt`. 1 represents the phenotype of resistance; 0 represents susceptibility.
- <a href="https://github.com/hzi-bifo/AMR_benchmarking/tree/main/data/PATRIC/cv_folds/loose/single_S_A_folds">Single-species-antibiotic evaluation folds</a> in the form of [ [sample list of fold 1], [sample list of fold 2],...[sample list of fold 10] ].


### Example
An example of evaluating a ML-based software that could use the scikit-learn module for training classifiers, and the feature matrix could be built without phenotype information.

(1). <a href="https://github.com/hzi-bifo/AMR_benchmarking/blob/main/AMR_software/Pseudo/benchmarking.py"> Integrate the ML model to the evaluation framework based on instructions </a>  
(2). Run the evaluation python scripts

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

## 2. Single-species multi-antibiotic dataset usage

 

### Example 
```
bash ./scripts/model/AytanAktug_SSMA.sh
```


## 3. Multi-species-antibiotic dataset usage

 



### Example 1
```
bash ./scripts/model/AytanAktug_MSMA_discrete.sh
```
### Example 2

```
bash ./scripts/model/phenotypeseeker_MS.sh
```
