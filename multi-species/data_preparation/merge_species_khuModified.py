import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import sys
sys.path.append('../')
sys.path.insert(0, os.getcwd())
import numpy as np
import ast
import time
import amr_utility.name_utility
import amr_utility.graph_utility
import argparse
import amr_utility.load_data
import pandas as pd
from itertools import repeat
import multiprocessing as mp
import subprocess
import data_preparation.merge_scaffolds_khuModified
import data_preparation.scored_representation_blast_khuModified
import data_preparation.ResFinder_analyser_blast_khuModified
import data_preparation.merge_resfinder_pointfinder_khuModified
import data_preparation.merge_input_output_files_khuModified
import data_preparation.merge_resfinder_khuModified



def
	#merge list ID to one file,used for kmc



def merge_feature(anti,list_species,debug,level,n_jobs):
	#merge feature matrix directly
	for species in list_species:
		#get the metadata
		save_name_anti = str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
		path_feature = './log/temp/' + str(level) + '/' + str(species.replace(" ", "_"))
		path_mutation_gene_results = path_feature + '/' + save_name_anti + '_res_point.txt'
		s_feature=np.genfromtxt(path_mutation_gene_results, dtype="str")









