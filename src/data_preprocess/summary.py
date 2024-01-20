#!/usr/bin/python

import argparse
import lib.metadata,lib.quality,lib.summary
from  src.amr_utility import file_utility
from src.amr_utility import name_utility
import pandas as pd

def workflow(level,logfile,temp_path):

    ## multi-species model metadata
    lib.metadata.extract_multi_model_summary(level)

    ##  get genome number. Print to the console.
    lib.summary.summary_genome(level)

    ## get genome number per combination. Save to ./data/PATRIC/meta/'+str(level)+'_genomeNumber/
    file_utility.make_dir('./data/PATRIC/meta/'+str(level)+'_genomeNumber')
    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]
    df_species = data.index.tolist()
    for species  in  df_species :
        lib.summary.summary_pheno(species,level)

    ## save genome number for each of s-a combination in multi-s-a dataset. Sep 2023
    lib.metadata.extract_multi_model_size(level)



if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--level', default='loose', type=str, required=False,
                        help='Quality control: strict or loose.default=\'loose\'.')
    parser.add_argument( '--logfile', default=None, type=str, required=False,
                        help='The log file')
    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                        help='The log file')
    parsedArgs=parser.parse_args()
    workflow(parsedArgs.level,parsedArgs.logfile,parsedArgs.temp_path)
