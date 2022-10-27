#!/usr/bin/python
from src.amr_utility import name_utility,load_data,file_utility
import argparse,os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import json


def extract_info(level,s,cv,f_all):
    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    if f_all == False:
        data = data.loc[s, :]
    df_species = data.index.tolist()

    for species in df_species:

        antibiotics, ID, _ =  load_data.extract_info(species, False, level)
        i_anti=0

        for anti in antibiotics:
            kf= KFold(n_splits=cv,random_state=0,shuffle=True)

            X= ID[i_anti]
            i_anti+=1
            folds=[]

            for train_index, test_index in kf.split(X):
                X_=np.array(X)
                sample_name=X_[test_index]
                # print(sample_name)
                folds.append(sample_name.tolist())

            folds_txt=name_utility.GETname_folds(species,anti,level,False,False)
            file_utility.make_dir(os.path.dirname(folds_txt))
            with open(folds_txt, 'w') as f:
                json.dump(folds, f)





if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--level', default='loose', type=str, required=False,
                        help='Quality control: strict or loose')
    parser.add_argument("-cv", "--cv_number", default=10, type=int,
                        help='CV splits number')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
         \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
         \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')

    parsedArgs = parser.parse_args()
    extract_info(parsedArgs.level,parsedArgs.species,parsedArgs.cv_number,parsedArgs.f_all)






