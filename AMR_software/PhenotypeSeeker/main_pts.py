#!/usr/bin/python

import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import argparse
import pickle,json
from src.cv_folds import name2index
from src.amr_utility import name_utility, file_utility, load_data
import numpy as np
import pandas as pd
import itertools
import math
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc,roc_curve,matthews_corrcoef,f1_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.pipeline import Pipeline
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
'''This script is to provide meta files for benchmarking Phenotypeseek, also also for running the ML nested CV part.'''


def extract_info(s,kmer,f_all,f_prepare_meta,cv,level,n_jobs,f_ml,f_phylotree,f_kma,temp_path):


    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    if f_all == False:
        data = data.loc[s, :]
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    file_utility.make_dir(temp_path+'log/software/phenotypeseeker/software_output/K-mer_lists') #for kmers


    if f_prepare_meta:
        # prepare the anti list and id list for each species, antibiotic, and CV folders.
        for species, antibiotics in zip(df_species, antibiotics):
            print(species)
            antibiotics, ID, _ =  load_data.extract_info(species, False, level)
            i_anti = 0
            antibiotics_=[]
            for anti in antibiotics:
                # prepare features for each training and testing sets, to prevent information leakage
                id_all = ID[i_anti]  # sample names
                i_anti += 1
                id_all = np.array(id_all)
                for out_cv in range(cv):

                    # 1. exrtact CV folders----------------------------------------------------------------

                    p_names = name_utility.GETname_meta(species,anti,level)
                    folds_txt=name_utility.GETname_folds(species,anti,level,f_kma,f_phylotree)
                    folders_sample = json.load(open(folds_txt, "rb"))
                    folders_index=name2index.Get_index(folders_sample,p_names) # CV folds

                    test_samples_index = folders_index[out_cv]
                    train_val_train_index = folders_index[:out_cv] + folders_index[out_cv + 1:]
                    id_val_train = id_all[
                        list(itertools.chain.from_iterable(train_val_train_index))]  # sample name list
                    id_test = id_all[test_samples_index]

                    # 2. prepare meta files for this round of training samples-------------------

                    _,name,meta_txt,_ = name_utility.GETname_model2('phenotypeseeker',level, species, anti,'',temp_path,f_kma,f_phylotree)
                    file_utility.make_dir(os.path.dirname(meta_txt))
                    name_list = pd.read_csv(name, index_col=0, dtype={'genome_id': object}, sep="\t")


                    # only retain those in the training and validataion CV folders
                    name_list_train = name_list.loc[name_list['genome_id'].isin(id_val_train)]
                    name_list_train['genome_id'].to_csv(meta_txt + '_Train_' + str(out_cv) + '_id2', sep="\t", index=False, header=False)
                    name_list_train['ID'] = temp_path+'log/software/phenotypeseeker/software_output/K-mer_lists/'+ \
                                            name_list_train['genome_id'].astype(str)+'_0_'+str(kmer)+'.list'
                    name_list_train['ID'].to_csv(meta_txt + '_Train_' + str(out_cv) + '_id', sep="\t", index=False, header=False)
                    name_list_train.rename(columns={'resistant_phenotype': anti}, inplace=True)
                    name_list_train = name_list_train.loc[:, ['genome_id',anti]]
                    name_list_train.to_csv(meta_txt + '_Train_' + str(out_cv) + '_data.pheno', sep="\t", index=True,
                                           header=True)


                    # 3. prepare meta files for this round of testing samples-------------------

                    # only retain those in the training and validataion CV folders
                    name_list_test = name_list.loc[name_list['genome_id'].isin(id_test)]
                    name_list_test['genome_id'].to_csv(meta_txt + '_Test_' + str(out_cv) + '_id2', sep="\t", index=False,
                                                 header=False)

                anti=str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
                antibiotics_.append(anti)
            # save the list to a txt file.
            anti_list,_,_,_= name_utility.GETname_model2('phenotypeseeker',level, species, '','',temp_path,f_kma,f_phylotree)
            pd.DataFrame(antibiotics_).to_csv(anti_list, sep="\t", index=False, header=False)



    if f_ml: #ML
        for species, antibiotics in zip(df_species, antibiotics):


            antibiotics, _, _ =  load_data.extract_info(species, False, level)
            if species=='Mycobacterium tuberculosis': #for this species, some combinations could not be finished with 2 months.
                antibiotics=['amikacin','capreomycin','ethiomide','ethionamide','kanamycin','ofloxacin','rifampin','streptomycin']


            # antibiotics=antibiotics[12:]
            for anti in antibiotics:
                print(anti)
                for chosen_cl in ['svm', 'lr','rf']:
                # for chosen_cl in ['rf']:
                    hyper_space, cl = hyper_range(chosen_cl)

                    mcc_test = []  # MCC results for the test data
                    f1_test = []
                    score_report_test = []
                    aucs_test = []
                    hyperparameters_test = []
                    score_InnerLoop=[]
                    index_InnerLoop=[]
                    cv_results_InnerLoop=[]
                    predictY_test=[]
                    true_Y=[]
                    sampleNames_test=[]
                    estimator_test=[]

                    _,meta_original, meta_txt,save_name_score=name_utility.GETname_model2('phenotypeseeker',level, species, anti,chosen_cl,temp_path,f_kma,f_phylotree)
                    file_utility.make_dir(os.path.dirname(save_name_score))
                    p_names = name_utility.GETname_meta(species,anti,level)
                    folds_txt=name_utility.GETname_folds(species,anti,level,f_kma,f_phylotree)
                    folders_sample = json.load(open(folds_txt, "rb"))
                    folders_index=name2index.Get_index(folders_sample,p_names) # CV folds

                    for out_cv in range(cv):
                        print('Starting outer: ', str(out_cv), '; chosen_cl: ', chosen_cl)
                        
                        train_set=pd.read_csv(meta_txt+ "_" + str(out_cv) +"_Train_df.csv",dtype={'genome_id': object}, sep="\t")
                        test_set=pd.read_csv(meta_txt+ "_" + str(out_cv) + "_Test_df.csv",dtype={'genome_id': object}, sep=",")
                        train_set=train_set.set_index('genome_id')
                        test_set = test_set.set_index('genome_id')
                        pipe = Pipeline(steps=[('cl', cl)])

                        train_val_train_index = folders_index[:out_cv] + folders_index[out_cv + 1:]
                        main_meta = pd.read_csv(meta_original, index_col=0, dtype={'genome_id': object}, sep="\t")

                        main_meta = main_meta.set_index('genome_id') #the sam eorder ias P_names
                        train_set_new=train_set.reindex(main_meta.index) #[force] the data_x 's order in according with id_list
                        train_set_new = train_set_new.fillna(0)#the nan part will not be used, because cv folders setting. But sklearn requires numerical type.
                        ### Those na indicates samples bolong to the testing set.

                        ###map test set to tain_val_test_set_order('ID') and train_set(columns)
                        test_set['phenotype'] = [main_meta.loc[sample, 'resistant_phenotype'] for sample in
                                                 test_set.index] # add pheno infor to test set.


                        #------------------------------------------
                        #------------------------------------------

                        X = train_set_new.iloc[:,0:-1].values#the whole set, with 0 for samples in test set.
                        y = train_set_new.iloc[:, -1].values.flatten() #the whole set
                        X_train = train_set.iloc[:, 0:-1].values
                        y_train = train_set.iloc[:,-1].values.flatten()


                        X_test = test_set.iloc[:,0:-1].values
                        y_test = test_set.iloc[:,-1].values.flatten()

                        search = GridSearchCV(estimator=pipe, param_grid=hyper_space, n_jobs=n_jobs,
                                              scoring='f1_macro',
                                              cv=create_generator(train_val_train_index), refit=True)

                        search.fit(X, y)
                        hyperparameters_test_sub=search.best_estimator_
                        scores_best=search.best_score_ #### July 2023. newly added. used to select the best classifiers for final outer loop testing.
                        index_best=search.best_index_
                        cv_results=search.cv_results_
                        current_pipe=hyperparameters_test_sub
                        # -------------------------------------------------------
                        ### retrain on train and val.
                        ### Note, the X,y include testing data, so we can't just use best_estimator_.predict() directly.
                        current_pipe.fit(X_train, y_train)
                        y_test_pred = current_pipe.predict(X_test)
                        ### scores
                        f1 = f1_score(y_test, y_test_pred, average='macro')
                        report = classification_report(y_test, y_test_pred, labels=[0, 1], output_dict=True)
                        mcc = matthews_corrcoef(y_test, y_test_pred)
                        fpr, tpr, _ = roc_curve(y_test, y_test_pred, pos_label=1)
                        roc_auc = auc(fpr, tpr)

                        f1_test.append(f1)
                        score_report_test.append(report)
                        aucs_test.append(roc_auc)
                        mcc_test.append(mcc)
                        hyperparameters_test.append(hyperparameters_test_sub)
                        score_InnerLoop.append(scores_best)
                        cv_results_InnerLoop.append(cv_results)
                        index_InnerLoop.append(index_best)
                        predictY_test.append( y_test_pred.tolist())
                        true_Y.append(y_test.tolist())
                        sampleNames_test.append(folders_sample[out_cv])
                        estimator_test.append(current_pipe)

                    score ={'f1_test':f1_test,'score_report_test':score_report_test,'aucs_test':aucs_test,'mcc_test':mcc_test,
                         'predictY_test':predictY_test,'ture_Y':true_Y,'samples':sampleNames_test}
                    score2= {'hyperparameters_test':hyperparameters_test,'estimator_test':estimator_test,
                             'score_InnerLoop':score_InnerLoop,'index_InnerLoop':index_InnerLoop,'cv_results_InnerLoop':cv_results_InnerLoop}
                    with open(save_name_score + '_KMA_' + str(f_kma) + '_Tree_' + str(f_phylotree) + '.json',
                              'w') as f:  # overwrite mode
                        json.dump(score, f)
                    with open(save_name_score + '_kma_' + str(f_kma) + '_tree_' + str(f_phylotree) + '_model.pickle',
                              'wb') as f:  # overwrite mode
                        pickle.dump(score2, f)

def create_generator(nFolds):
    for idx in range(len(nFolds)):
        test =nFolds[idx]
        train = list(itertools.chain(*[fold for idy, fold in enumerate(nFolds) if idy != idx]))
        yield train, test
def hyper_range(chosen_cl):

    if chosen_cl=='svm':
        cl=SVC(random_state=0,class_weight='balanced')
        hyper_space = {'cl__C': _get_gammas(None,1E-3,1E3,13),
                        'cl__gamma':_get_gammas(None,1E-3,1E3,13),
                         'cl__kernel': ['rbf', 'linear']}#'poly','sigmoid',

    if chosen_cl=='lr':
        cl=LogisticRegression( random_state=0,max_iter=10000,class_weight='balanced')
        hyper_space = {'cl__C': _get_gammas(None,1E-6,1E6,25),'cl__penalty':['l1'],'cl__solver':["liblinear", "saga"]}

    if chosen_cl=='rf':
        cl=RandomForestClassifier(random_state=0, class_weight='balanced')
        hyper_space = {'cl__max_features': ["sqrt", "log2", None],'cl__n_estimators':[ 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
        'cl__max_depth': [4, 5, 6, 7, 8, 10, 20, 100, None], 'cl__min_samples_split': [2,5,10],'cl__min_samples_leaf': [1, 2, 4],
                       'cl__criterion': ['gini', 'entropy'],'cl__bootstrap': [True, False]}

    if chosen_cl == 'dt':
        cl=tree.DecisionTreeClassifier(random_state=0,class_weight='balanced')
        hyper_space = {'cl__max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'cl__criterion' :['gini', 'entropy']}

    return hyper_space,cl


def _get_gammas(gammas, gamma_min, gamma_max, n_gammas):
    # Generating the vector of gammas
    # (hyperparameters in SVM kernel analysis)
    # based on the given command line arguments.
    if gammas == None:
        gammas = np.logspace(
            math.log10(gamma_min),
            math.log10(gamma_max), num=n_gammas)
    else:
        gammas = np.array(gammas)
    return gammas


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                        help='Directory to store temporary files.')
    parser.add_argument('-k', '--kmer', default=13, type=int,
                        help='k-mer')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                        help='kma based cv folders.')
    parser.add_argument('-l', '--level', default='loose', type=str,
                        help='Quality control: strict or loose')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument('-f_prepare_meta', '--f_prepare_meta', dest='f_prepare_meta', action='store_true',
                        help='Prepare the list files .')
    parser.add_argument('-f_ml', '--f_ml', dest='f_ml', action='store_true',
                        help='prepare bash')
    parser.add_argument("-cv", "--cv", default=10, type=int,
                        help='CV splits number')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'Pseudomonas aeruginosa\' \
                    \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
                    \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('-n_jobs','--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')

    parsedArgs = parser.parse_args()
    extract_info( parsedArgs.species, parsedArgs.kmer,parsedArgs.f_all, parsedArgs.f_prepare_meta,
                   parsedArgs.cv, parsedArgs.level, parsedArgs.n_jobs, parsedArgs.f_ml,
                 parsedArgs.f_phylotree,parsedArgs.f_kma,parsedArgs.temp_path)
