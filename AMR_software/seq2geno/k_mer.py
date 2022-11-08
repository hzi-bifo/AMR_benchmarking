#!/usr/bin/python
import sys
import os
# sys.path.append('../')
sys.path.insert(0, os.getcwd())
import pandas as pd
import numpy as np
from itertools import repeat
import multiprocessing as mp
from src.amr_utility import name_utility,math_utility
import argparse





def kmer(species,k,canonical,vocab,temp_path):

    '''
    Output: saved h5 files of kmer features, w.r.t. each species.
    '''


    save_name_speciesID='./data/PATRIC/meta/by_species_bq/id_' + str(species.replace(" ", "_"))
    save_name_kmer ,_, _, _, _,_ =  name_utility.GETname_S2Gfeature(species, temp_path,k)
    save_mame_kmc=str(temp_path)+'log/software/seq2geno/software_output/cano'+ str(k) + 'mer/merge_'+str(k)+'mers_'

    data_sub=pd.read_csv(save_name_speciesID, index_col=0,dtype={'genome_id': object},sep="\t")
    data_sub_uniqueID = data_sub.groupby(by="genome_id").count()
    ID = data_sub_uniqueID.index.tolist()
    print('Number of strains:', len(ID))

    # initialize!
    # k mer features
    data = np.zeros((len(vocab), 1), dtype='uint16')
    feature_matrix = pd.DataFrame(data, index=vocab, columns=['initializer'])  # delete later
    feature_matrix.index.name = 'feature'
    l = 0
    for i in ID:
        l += 1
        if (l % 1000 == 0): #just for checking the process.
            print(l,species)
        # map feature txt from stored data(KMC tool processed) to feature matrix.
        f = pd.read_csv(save_mame_kmc + str(i) + '.txt',
                        names=['combination', str(i)],dtype={'genome_id': object}, sep="\t")
        f = f.set_index('combination')
        feature_matrix = pd.concat([feature_matrix, f], axis=1)

    feature_matrix = feature_matrix.drop(['initializer'], axis=1)
    feature_matrix = feature_matrix.fillna(0)
    feature_matrix.to_hdf(save_name_kmer, key='df', mode='w', complevel=9)#overwriting.
    print(feature_matrix)


def extract_info(s,k,canonical,level,temp_path,n_jobs,f_all):
    '''extract feature w.r.t. each species and drug
    k: k mer feature
    '''

    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    if f_all == False:
        data = data.loc[s, :]
    df_species = data.index.tolist()

    vocab=math_utility.vocab_build(canonical,k)# all the kmer combinations in list
    pool = mp.Pool(processes=n_jobs)
    pool.starmap(kmer, zip(df_species,repeat(k),repeat(canonical),repeat(vocab),repeat(temp_path)))

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k',default=6, type=int, required=True,
                        help='Kmer size')
    parser.add_argument('-n_jobs','--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parser.add_argument('-c','--canonical', dest='canonical',action='store_true',
                        help='Canonical kmer or not: True')
    parser.add_argument('-n','--non_canonical',dest='canonical',action='store_false',
                        help='Canonical kmer or not: False')
    parser.add_argument('-l', '--level', default='loose', type=str, required=False,
                        help='Quality control: strict or loose.default=\'loose\'.')
    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                        help='Directory to store temporary files.')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species.')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
                \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
                \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    #parser.set_defaults(canonical=True)
    # parser.print_help()
    parsedArgs=parser.parse_args()

    extract_info(parsedArgs.species,parsedArgs.k,parsedArgs.canonical,parsedArgs.level,parsedArgs.temp_path,parsedArgs.n_jobs,parsedArgs.f_all)
