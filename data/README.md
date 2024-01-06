# How to evaluate a model using the datasets
## Usage

`
usage: evaluate.py [-h] -software_name SOFTWARE_NAME -path_sequence PATH_SEQUENCE [-temp TEMP_PATH] [-l LEVEL] [-f_all] [-f_phylotree] [-f_kma] [-s SPECIES [SPECIES ...]] [-f_ml] [-cv CV_NUMBER]
                   [-n_jobs N_JOBS]

options:
  -h, --help            show this help message and exit
  -software_name SOFTWARE_NAME, --software_name SOFTWARE_NAME
                        software_name
  -path_sequence PATH_SEQUENCE, --path_sequence PATH_SEQUENCE
                        Path of the directory with PATRIC sequences
  -temp TEMP_PATH, --temp_path TEMP_PATH
                        Directory to store temporary files.
  -l LEVEL, --level LEVEL
                        Quality control: strict or loose
  -f_all, --f_all       all the possible species.
  -f_phylotree, --f_phylotree
                        phylo-tree based cv folds.
  -f_kma, --f_kma       kma based cv folds.
  -s SPECIES [SPECIES ...], --species SPECIES [SPECIES ...]
                        species to run: e.g.'Pseudomonas aeruginosa' 'Klebsiella pneumoniae' 'Escherichia coli' 'Staphylococcus aureus' 'Mycobacterium tuberculosis' 'Salmonella enterica'
                        'Streptococcus pneumoniae' 'Neisseria gonorrhoeae'
  -f_ml, --f_ml         ML
  -cv CV_NUMBER, --cv_number CV_NUMBER
                        CV splits number. Default=10
  -n_jobs N_JOBS, --n_jobs N_JOBS
                        Number of jobs to run in parallel.


`
## Example

`
python AMR_software/Pseudo/evaluate.py -s 
`



