# Dataset Usage



## 1. Single-species-antibiotic dataset usage
- Each genome (sample) is represented by its unique PATRIC ID.
- In total, there are 78 datasets, each corresponding to a species-antibiotic combination.
- <a href="https://github.com/hzi-bifo/AMR_benchmarking/tree/main/data/PATRIC/meta/loose_by_species">Sample list</a> of each species-antibiotic combination. The files are named as `Data_<species>_<antibiotic>`. Each file contains all the genome samples for a dataset, i.e. each file corresponds to a dataset.
- <a href="https://github.com/hzi-bifo/AMR_benchmarking/tree/main/data/PATRIC/meta/loose_by_species">Sample phenotype metadata</a> of each dataset. The files are named as `Data_<species>_<antibiotic>_pheno.txt`. 1 represents the resistance phenotype; 0 represents the susceptibility phenotype. Each file contains all the genome samples for a dataset.
- <a href="https://github.com/hzi-bifo/AMR_benchmarking/tree/main/data/PATRIC/cv_folds/loose/single_S_A_folds">Single-species-antibiotic evaluation folds</a> in the form of [ [sample list of fold 1], [sample list of fold 2],...[sample list of fold 10] ].


### Example
An example of evaluating a ML-based software that could use the scikit-learn module for training classifiers, and the feature matrix could be built without phenotype information. Among our benchmarked methods, Seq2Geno2Pheno falls into this category.

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
 - Each genome (sample) is represented by its unique PATRIC ID.
 - In total, there are nine datasets, each corresponding to a species (<em>M. tuberculosis, E. coli, S. aureus, S. enterica, K. pneumoniae, P. aeruginosa, A. baumannii, S. pneumoniae, N. gonorrhoeae</em>).
 - <a href="https://github.com/hzi-bifo/AMR_benchmarking/tree/main/data/PATRIC/cv_folds/loose/single_S_multi_A_folds">Multi-antibiotic evaluation folds</a> in the form of [ [sample list of fold 1], [sample list of fold 2],...[sample list of fold 10] ] for each species.
 - For each dataset, please refer to corresponding species' <a href="https://github.com/hzi-bifo/AMR_benchmarking/tree/main/data/PATRIC/meta/loose_by_species">`metadata`</a> for the phenotype of each genome mentioned in above folds. The files are named `Data_<species>_<antibiotic>_pheno.txt`. 1 represents the resistance phenotype; 0 represents the susceptibility phenotype. If a genome (for example, in the <em>E. coli</em> multi-antibiotic dataset) is absent in a Data_Escherichia_coli_<antibiotic>_pheno.txt file, it means there is no phenotype information of this specific antibiotic for this genome.

### Example 
Neural networks model via nested CV 
```
bash ./scripts/model/AytanAktug_SSMA.sh
```


## 3. Multi-species-antibiotic dataset usage
 - Each genome (sample) is represented by its unique PATRIC ID.
 - In total, the dataset is composed of 54 species-antimicrobial combinations. Nine species, <em>M. tuberculosis, E. coli, S. aureus, S. enterica, K. pneumoniae, P. aeruginosa, A. baumannii, S. pneumonia, C. jejuni</em>, are involved.
 - <a href="https://github.com/hzi-bifo/AMR_benchmarking/tree/main/data/PATRIC/cv_folds/loose/multi_S_folds">Multi-species-antibiotic evaluation folds</a> for Aytan-Aktug control multi-species model solely, in the form of [ [sample list of fold 1], [sample list of fold 2],...[sample list of fold 10] ]. Phenotype metadata for each relevant genome can be found the same way as multi-antibiotic dataset usage.
 - <a href="https://github.com/hzi-bifo/AMR_benchmarking/tree/main/data/PATRIC/cv_folds/loose/multi_S_LOO_folds"> Leave-one-species-out multi-species-antibiotic evaluation folds</a>. Each file in the folder contains samples associated with a specific species, thus making up a fold. Each time use all the samples in one file as the test set, and use all the samples in the rest files, that are associated with the other eight species, as the training set.



### Example 1
Neural networks model (Aytan-Aktug control multi-species) via conventional CV 
```
bash ./scripts/model/AytanAktug_MSMA_discrete.sh
```
### Example 2
Neural networks model via Leave-one-species-out evaluation
```
bash ./scripts/model/AytanAktug_MSMA_concat.sh
```
