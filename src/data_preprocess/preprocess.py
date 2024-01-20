#!/usr/bin/python

import logging,argparse
import lib.metadata,lib.quality,lib.summary
from  src.amr_utility import file_utility
from src.amr_utility import name_utility
import pandas as pd

def workflow(level,logfile,temp_path):
    handlers = [
        logging.StreamHandler()
    ]
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile, mode='w'))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=handlers

    )

    logger = logging.getLogger('data_preprocess')

    #  Pre-selection
    temp_path=temp_path+'log/temp/data/'
    file_utility.make_dir(temp_path)
    lib.metadata.summarise_strain(temp_path)
    logger.info('finish extracting information from PATRIC_genomes_AMR.txt')
    lib.metadata.summarise_species(temp_path)
    lib.metadata.sorting_deleting(500,temp_path) #500: retain only this that has >=500 strains for a specific antibotic w.r.t. a species
    lib.metadata.extract_id(temp_path)
    lib.metadata.extract_id_species(temp_path)




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
