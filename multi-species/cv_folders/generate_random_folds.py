import amr_utility.name_utility
import amr_utility.graph_utility
import amr_utility.file_utility
import argparse
import amr_utility.load_data
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import pickle


def extract_info(level,s,cv,f_all):
    data = pd.read_csv('metadata/' + str(level) + '_Species_antibiotic_FineQuality.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    # for training model on part of the dataset:-------------
    # data=data.loc[['Escherichia coli'],:]
    if f_all == False:
        data = data.loc[s, :]
    # --------------------------------------------------------
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    for species, antibiotics in zip(df_species, antibiotics):

        antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)

        i_anti=0

        for anti in antibiotics:
            kf= KFold(n_splits=cv,random_state=0,shuffle=True)

            X= ID[i_anti]
            folds=[]
            for train_index, test_index in kf.split(X):
                X_=np.array(X)
                sample_name=X_[test_index]
                # print(sample_name)
                folds.append(sample_name)
            # pd.DataFrame(folds).to_csv("./cv_folders/"+str(level)+"/"+str(species.replace(" ", "_"))+"/"+
            #                            str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_random_cv.txt", sep="\t",index=False,header=False)
            folds_txt = "./cv_folders/"+str(level)+"/"+str(species.replace(" ", "_"))+"/"+\
                                       str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_random_cv.pickle"
            with open(folds_txt, 'wb') as f:  # overwrite
                pickle.dump(folds, f)
        i_anti+=1




if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--l', '--level', default='loose', type=str, required=False,
                        help='Quality control: strict or loose')
    parser.add_argument("-cv", "--cv_number", default=10, type=int,
                        help='CV splits number')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    # parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
         \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
         \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')


    # parser.set_defaults(canonical=True)
    parsedArgs = parser.parse_args()
    # parser.print_help()
    print(parsedArgs)
    extract_info(parsedArgs.l,parsedArgs.species,parsedArgs.cv_number,parsedArgs.f_all)






