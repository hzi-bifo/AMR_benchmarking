#!/usr/bin/python

import  argparse
import lib.metadata,lib.quality,lib.summary

def workflow(level,temp_path):

    # genome quality control
    lib.quality.extract_id_quality(temp_path,level)
    lib.quality.filter_phenotype(level,False) #False indicates No extra sampling handling for imbalance dataset.




if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--level', default='loose', type=str, required=False,
                        help='Quality control: strict or loose.default=\'loose\'.')
    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                        help='The log file')
    parsedArgs=parser.parse_args()
    workflow(parsedArgs.level,parsedArgs.temp_path)
