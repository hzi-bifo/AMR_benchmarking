import sys
import os
sys.path.insert(0, os.path.abspath('../'))
import amr_utility.name_utility
import amr_utility.load_data
import amr_utility.file_utility
import src.BySpecies,src.pairbox_std
import src.ByAnti
import src.dimension,src.dimension_2
import src.pairbox, src.SampleSize,src.excel,src.BySpecies_std,src.Gfolds,src.excel_abstract,src.BySpecies_std_w,src.excel_clinical
import ast
import numpy as np
import argparse
import pickle
import pandas as pd
import seaborn as sns




'''
This script summerizes benchmarking resutls as graphs.
'''




def extract_info(level,s, fscore, cv_number, f_phylotree, f_kma,f_all,f_species, f_anti,f_kmer,f_folds,f_sample,f_table,f_generateFolds):
    if f_species:#benchmarking by species
        # src.BySpecies.ComBySpecies(level,s, fscore, cv_number, f_phylotree, f_kma,f_all)
        src.BySpecies_std.ComBySpecies(level,s, fscore, cv_number, f_phylotree, f_kma,f_all)
        src.BySpecies_std_w.draw(level,s, fscore, cv_number, f_all) #all together in one graph.
    elif f_anti:#benchmarking by antibiotics.
        # src.ByAnti.ComByAnti(level,s, fscore, cv_number, f_phylotree, f_kma,f_all)
        src.ByAnti_errorbar.ComByAnti(level,s, fscore, cv_number, f_phylotree, f_kma,f_all)
    elif f_kmer:#plotting dimensions of each k
        src.dimension.extract_info(level,s, fscore, cv_number, f_phylotree, f_kma,f_all)
        # src.dimension_2.extract_info(level,s, fscore, cv_number, f_phylotree, f_kma,f_all)
    elif f_folds:#plotting pairbox w.r.t. 3 folds split methods.
        src.pairbox.extract_info(level,s, fscore, cv_number, f_phylotree, f_kma,f_all,'1')
        src.pairbox.extract_info(level,s, fscore, cv_number, f_phylotree, f_kma,f_all,'3')
        src.pairbox_std.extract_info(level,s, fscore, cv_number, f_phylotree, f_kma,f_all,'1')
        src.pairbox_std.extract_info(level,s, fscore, cv_number, f_phylotree, f_kma,f_all,'3')
    elif f_sample:#plotting sample size.
        src.SampleSize.extract_info(level,s, f_all )
    elif f_table:#performance 4 scores for Supplementary materials.
        # src.excel.extract_info(level,s,fscore, f_all )
        src.excel_clinical.extract_info(level,s,fscore, f_all )
        # src.excel_abstract.extract_info(level,s,fscore, f_all )
    elif f_generateFolds:#generate 10 folds for 3 set of folds.
        src.Gfolds.extract_info(level, f_phylotree, f_kma)

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--l', '--level', default='loose', type=str, required=False,
                        help='Quality control: strict or loose')
    parser.add_argument("-cv", "--cv_number", default=10, type=int,
                        help='CV splits number')
    # parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
         \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
         \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument('-f_species', '--f_species', dest='f_species', action='store_true',
                        help='benchmarking by species.')
    parser.add_argument('-f_anti', '--f_anti', dest='f_anti', action='store_true',
                        help='benchmarking by antibiotics.')
    parser.add_argument('-fscore', '--fscore', default='f1_macro', type=str, required=False,
                        help='the score used to choose the best classifier for each antibiotic. Can be one of: \'f1_macro\',\'f1_positive\',\'f1_negative\',\'accuracy\'')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                        help='kma based cv folders.')
    parser.add_argument('-f_kmer', '--f_kmer', dest='f_kmer', action='store_true',
                        help='plotting dimensions of each k')
    parser.add_argument('-f_folds', '--f_folds', dest='f_folds', action='store_true',
                        help='plotting pairbox w.r.t. 3 folds split methods.')
    parser.add_argument('-f_sample', '--f_sample', dest='f_sample', action='store_true',
                        help='plotting sample size bar graph.')
    parser.add_argument('-f_table', '--f_table', dest='f_table', action='store_true',
                        help='performance 4 score for Supplementary materials.')
    parser.add_argument('-f_generateFolds', '--f_generateFolds', dest='f_generateFolds', action='store_true',
                        help='generate 10 folds for 3 set of folds.')
    # parser.set_defaults(canonical=True)
    parsedArgs = parser.parse_args()
    # parser.print_help()
    print(parsedArgs)
    extract_info(parsedArgs.l,parsedArgs.species,parsedArgs.fscore,parsedArgs.cv_number,parsedArgs.f_phylotree,parsedArgs.f_kma,parsedArgs.f_all,
                 parsedArgs.f_species,parsedArgs.f_anti,parsedArgs.f_kmer,parsedArgs.f_folds,parsedArgs.f_sample,parsedArgs.f_table,parsedArgs.f_generateFolds)
