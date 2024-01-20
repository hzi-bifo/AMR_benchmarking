#!/usr/bin/python
"""
Created on Jan 6 2024
@author: Kaixin Hu
Scripts for evaluating ML model
"""
import sys,os
sys.path.insert(0, os.getcwd())
import AMR_software.Pseudo.benchmarking as benchmarking
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-software', '--software_name', type=str, required=True,
                    help='software_name')
parser.add_argument('-sequence', '--path_sequence', type=str, required=True,
                    help='Path of the directory with PATRIC sequences')
parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                    help='Directory to store temporary files, like features.')
parser.add_argument('-l', '--level', default='loose', type=str,
                    help='Quality control: strict or loose')
parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                    help='Benchmarking on all the possible species.')
parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                    help='phylogeny-aware evaluation')
parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                    help='homology-aware evaluation')
parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'Pseudomonas aeruginosa\' \
            \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
            \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
parser.add_argument('-f_ml', '--f_ml', dest='f_ml', action='store_true',
                    help='Perform neseted CV on ML models')
parser.add_argument("-cv", "--cv_number", default=10, type=int,
                    help='CV splits number. Default=10 ')
parser.add_argument('-n_jobs','--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
parsedArgs = parser.parse_args()
parser.print_help()


### initiating
EVAL = benchmarking.AMR_software(parsedArgs.software_name,parsedArgs.path_sequence,parsedArgs.temp_path,parsedArgs.s,
                         parsedArgs.f_all, parsedArgs.f_phylotree, parsedArgs.f_kma,parsedArgs.cv,parsedArgs.n_jobs)

#### 1. Feature building
EVAL.prepare_feature()


#### 2. nested CV
EVAL.nested_cv()


