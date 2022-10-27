#!/usr/bin/python

from src.amr_utility import name_utility, file_utility, load_data
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import auc,roc_curve,matthews_corrcoef,f1_score,classification_report
from src.cv_folds import name2index
import pickle,json
import argparse
import itertools
import pandas as pd
import numpy as np
import ast,os

def model(level,species, antibiotics,cv,f_phylotree,f_kma,temp_path):
    antibiotics_selected = ast.literal_eval(antibiotics)

    print(species)
    print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)
    antibiotics, ID, Y =  load_data.extract_info(species, False, level)


    i_anti=0
    for anti in antibiotics:
        print(anti)
        p_names = name_utility.GETname_meta(species,anti,level)
        folds_txt=name_utility.GETname_folds(species,anti,level,f_kma,f_phylotree)
        folders_sample = json.load(open(folds_txt, "rb"))
        folders_index=name2index.Get_index(folders_sample,p_names) # CV folds

        _, _,save_name_score=name_utility.GETname_model('majority', level,species, anti,'majority',temp_path)
        file_utility.make_dir(os.path.dirname(save_name_score))



        id_all = ID[i_anti]#sample name list
        y_all = Y[i_anti]
        i_anti+=1
        id_all =np.array(id_all)
        y_all = np.array(y_all)

        mcc_test = []  # MCC results for the test data
        f1_test = []
        score_report_test = []
        aucs_test = []
        predictY_test=[]
        true_Y=[]
        sampleNames_test=[]

        for out_cv in range(cv):
            print('Starting outer: ', str(out_cv))
            test_samples_index = folders_index[out_cv]# a list of index

            id_test = id_all[test_samples_index]#sample name list
            y_test = y_all[test_samples_index]

            train_val_train_index =folders_index[:out_cv] +folders_index[out_cv + 1:]
            train_val_train_index_list=list(itertools.chain.from_iterable(train_val_train_index))
            id_val_train = id_all[train_val_train_index_list]  # sample name list
            y_val_train = y_all[train_val_train_index_list]

            #find the majority class in the y_val_train
            N_S=y_val_train.tolist().count(0)
            N_R=y_val_train.tolist().count(1)

            if N_S>N_R:
                y_test_pred=np.zeros(len(y_test)).tolist()
            else:
                y_test_pred=np.ones(len(y_test)).tolist()

            # scores
            f1 = f1_score(y_test, y_test_pred, average='macro')
            report = classification_report(y_test, y_test_pred, labels=[0, 1], output_dict=True)
            mcc = matthews_corrcoef(y_test, y_test_pred)
            fpr, tpr, _ = roc_curve(y_test, y_test_pred, pos_label=1)
            roc_auc = auc(fpr, tpr)

            f1_test.append(f1)
            score_report_test.append(report)
            aucs_test.append(roc_auc)
            mcc_test.append(mcc)
            predictY_test.append(y_test_pred)
            true_Y.append(y_test.tolist())
            sampleNames_test.append(folders_sample[out_cv])




        score ={'f1_test':f1_test,'score_report_test':score_report_test,'aucs_test':aucs_test,'mcc_test':mcc_test,
                         'predictY_test':predictY_test,'ture_Y':true_Y,'samples':sampleNames_test}

        with open(save_name_score + '_KMA_' + str(f_kma) + '_Tree_' + str(f_phylotree) + '.json',
                  'w') as f:  # overwrite mode
            json.dump(score, f)




def extract_info(level,s, cv,f_phylotree,f_kma,f_all,temp_path):
    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    if f_all == False:
        data = data.loc[s, :]
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    for species,antibiotics in zip(df_species, antibiotics):

        model(level, species,  antibiotics,cv,f_phylotree,f_kma,temp_path)

if __name__== '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                        help='Directory to store temporary files.')
    parser.add_argument('-l', '--level', default='loose', type=str, required=False,
                        help='Quality control: strict or loose.default=\'loose\'.')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folds.')
    parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                        help='kma based cv folds.')
    parser.add_argument("-cv", "--cv_number", default=10, type=int,
                        help='CV splits number. Default=10 ')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    # parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel. Default=1')
    parser.add_argument('-s','--species', default=[], type=str,nargs='+',help='species to run: e.g.\'seudomonas aeruginosa\' \
     \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
     \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')

    parsedArgs=parser.parse_args()
    extract_info(parsedArgs.level,parsedArgs.species, parsedArgs.cv_number,
                 parsedArgs.f_phylotree,parsedArgs.f_kma,parsedArgs.f_all,parsedArgs.temp_path)
