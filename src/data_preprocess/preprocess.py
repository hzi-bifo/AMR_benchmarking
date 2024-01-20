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

    # 1. pre-selection
    temp_path=temp_path+'log/temp/data/'
    file_utility.make_dir(temp_path)
    lib.metadata.summarise_strain(temp_path)
    logger.info('finish extracting information from PATRIC_genomes_AMR.txt')
    lib.metadata.summarise_species(temp_path)
    lib.metadata.sorting_deleting(500,temp_path) #500: retain only this that has >=500 strains for a specific antibotic w.r.t. a species
    lib.metadata.extract_id(temp_path)
    lib.metadata.extract_id_species(temp_path)

    # 2. QC
    lib.quality.extract_id_quality(temp_path,level)
    lib.quality.filter_quality(level,False) #False: No handling related to imbalance dataset.
    lib.metadata.extract_multi_model_summary(level)#multi-species model metadata



    #3. get genome number. Print to the console.
    lib.summary.summary_genome(level)

    #4.  get genome number per combination. Save to ./data/PATRIC/meta/'+str(level)+'_genomeNumber/
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
