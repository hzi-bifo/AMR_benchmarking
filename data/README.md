# Datasets Usage

1. <a href="https://github.com/hzi-bifo/AMR_benchmarking/blob/main/AMR_software/Pseudo/benchmarking.py"> Integrate the ML model to the evaluation framework based on instructions </a>  
2. Run the evaluation python scripts

### Full help

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
### Example

```
python AMR_software/Pseudo/evaluate.py -software 'seq2geno' -s 'Escherichia coli' -sequence '/vol/projects/patric_genome' -temp './' -f_phylotree -cv 10 -n_jobs 1

```



