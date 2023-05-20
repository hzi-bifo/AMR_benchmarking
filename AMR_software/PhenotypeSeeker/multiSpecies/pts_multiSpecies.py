#!/usr/bin/python
from src.amr_utility import name_utility, file_utility,load_data
import argparse,itertools,os
import numpy as np
import pandas as pd
from src.cv_folds import name2index
import json,pickle
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
import itertools
import math
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc,roc_curve,matthews_corrcoef,f1_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.pipeline import Pipeline

"For preparing meta files to run PhenotySeeker over LOSO. " \
"Each time one species-antibiotic combination is used as testing set. " \
"The other combinations sharing the samne antibiotics are used as training set. "



def extract_info(list_species,f_all,kmer, f_prepare_meta ,f_ml,cv,level, temp_path,n_jobs):
    '''
    list_species:
    f_all: using all possible antibiotics that have been assigned to multiple species
    '''


    merge_name = []
    data = pd.read_csv('./data/PATRIC/meta/'+str(level)+'_multi-species_summary.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")

    if f_all:
        list_species = data.index.tolist()[:-1]
        data = data.loc[list_species, :]
    else:
        print('Warning. You are not using all the possible data.')
        data = data.loc[list_species, :]
        data = data.loc[:, (data.sum() > 1)]
    data = data.loc[:, (data != 0).any(axis=0)]
    All_antibiotics = data.columns.tolist()
    df_anti = data.dot(data.columns + ';').str.rstrip(';')
    for n in list_species:
        merge_name.append(n[0] + n.split(' ')[1][0])
    merge_name = '_'.join(merge_name)  # e.g.Se_Kp_Pa
    print(data)
    print(df_anti)

    if f_prepare_meta:
        # prepare the anti list and id list for each species, antibiotic, and CV folds.
        count = 0
        for species_testing in list_species:
            Potential_species_training=list_species[:count] + list_species[count+1 :]
            count += 1
            antibiotics_test=df_anti[species_testing].split(';') #involoved antibiotics
            for anti_test in antibiotics_test:
                print('the species-anti combination to test: ', species_testing, '_',anti_test)
                #########
                ##1. define train & test species,antibiotic. Each time only one combination as test.
                ########
                list_species_training=[]
                for each_species in Potential_species_training:
                    # print(each_species)
                    antibiotics_temp= df_anti[each_species].split(';')
                    # print(antibiotics_temp)
                    if anti_test in antibiotics_temp:
                        list_species_training.append(each_species)
                print('the species for training: ',list_species_training)


                #########
                ##3. training & test set meta data
                ########

                name_list_train = pd.DataFrame(columns=['genome_id', 'resistant_phenotype'])
                for each_species in list_species_training:
                    _,id_name_train,_,_ = name_utility.GETname_model3('phenotypeseeker',level, each_species, anti_test,'',temp_path)
                    name_each = pd.read_csv(id_name_train, index_col=0, dtype={'genome_id': object}, sep="\t")
                    name_list_train = name_list_train.append(name_each, ignore_index=True)

                anti_save,_,meta_save,_ = name_utility.GETname_model3('phenotypeseeker',level, species_testing, anti_test,'',temp_path)
                file_utility.make_dir(os.path.dirname(meta_save))
                name_list_train['genome_id'].to_csv(meta_save + '_Train_whole_id2', sep="\t", index=False, header=False)
                name_list_train.loc[:,'ID'] = temp_path+'log/software/phenotypeseeker/software_output/K-mer_lists/'+ \
                                              name_list_train['genome_id'].astype(str)+'_0_'+str(kmer)+'.list'
                name_list_train['ID'].to_csv(meta_save +  '_Train_whole_id', sep="\t", index=False, header=False)

                name_list_train.rename(columns={'resistant_phenotype': anti_test}, inplace=True)
                name_list_train = name_list_train.loc[:, ['genome_id',anti_test]]
                name_list_train.to_csv(meta_save + '_Train_whole_data.pheno', sep="\t", index=True,header=True)

                #### test meta data
                _,id_name_test,_,_ = name_utility.GETname_model3('phenotypeseeker',level, species_testing, anti_test,'',temp_path)
                name_list_test = pd.read_csv(id_name_test, index_col=0, dtype={'genome_id': object}, sep="\t")
                name_list_test['genome_id'].to_csv(meta_save + '_Test_whole_id2', sep="\t", index=False, header=False)

                name_list_test.rename(columns={'resistant_phenotype': anti_test}, inplace=True)
                name_list_test = name_list_test.loc[:, ['genome_id',anti_test]]
                name_list_test.to_csv(meta_save + '_Test_whole_data.pheno', sep="\t", index=True,header=True)



            antibiotics_test_=[str(anti.translate(str.maketrans({'/': '_', ' ': '_'}))) for anti in antibiotics_test]
            pd.DataFrame(antibiotics_test_).to_csv(anti_save, sep="\t", index=False, header=False)



    if f_ml: #ML #f_kma,f_phylotree=True, False. Use homology-aware folds
        count=0
        for species_testing in list_species:

            ### if count==4: # remove this when fully finished.
            print('the species to be tested : ',species_testing)
            Potential_species_training=list_species[:count] + list_species[count+1 :]
            antibiotics_test=df_anti[species_testing].split(';') #involoved antibiotics


            for anti in antibiotics_test:
                print(anti)
                #########
                ##1. define train & test species,antibiotic. Each time only one combination as test.
                ########
                list_species_training=[]
                for each_species in Potential_species_training:
                    antibiotics_temp= df_anti[each_species].split(';')
                    if anti in antibiotics_temp:
                        list_species_training.append(each_species)
                print('the species for training: ',list_species_training)


                ### for chosen_cl in ['svm', 'lr','rf']:
                for chosen_cl in ['lr']:
                    hyper_space, cl = hyper_range(chosen_cl)

                    mcc_test = []  # MCC results for the test data
                    f1_test = []
                    score_report_test = []
                    aucs_test = []
                    hyperparameters_test = []
                    predictY_test=[]
                    true_Y=[]
                    sampleNames_test=[]
                    estimator_test=[]

                    ### _,meta_original, meta_txt,save_name_score=name_utility.GETname_model2('phenotypeseeker',level, species, anti,chosen_cl,temp_path,f_kma,f_phylotree)
                    _,meta_original, meta_txt,save_name_score= name_utility.GETname_model3('phenotypeseeker',level, species_testing, anti,chosen_cl,temp_path)
                    file_utility.make_dir(os.path.dirname(save_name_score))

                    ################################################################################################
                    ###  1. For all species involved for training, generate folds for CV for hyperparameter selection.
                    ################################################################################################
                    f_kma,f_phylotree=True, False
                    Folds=[]
                    for i_cv in range(cv):
                        Folds.append([])
                    for each in list_species_training: #todo, check. checked.
                        ### note: not the same as AytanAktug MSMA, as single-anti here.
                        folds_txt=name_utility.GETname_folds(each,anti,level,f_kma,f_phylotree)
                        folds_sample = json.load(open(folds_txt, "rb"))
                        for i_cv in range(cv):
                            Folds[i_cv].append(folds_sample[i_cv])
                    p_names=meta_txt + '_Train_whole_id2'
                    folders_index=name2index.Get_index(folds_sample,p_names) # CV folds
                    ################################################################################################


                    ##2. Training & CV
                    train_set=pd.read_csv(meta_txt+ "_whole_Train_df.csv",dtype={'genome_id': object}, sep="\t")
                    test_set=pd.read_csv(meta_txt+ "_whole_Test_df.csv",dtype={'genome_id': object}, sep=",")
                    train_set=train_set.set_index('genome_id')
                    test_set = test_set.set_index('genome_id')
                    pipe = Pipeline(steps=[('cl', cl)])

                    train_val_train_index = folders_index
                    main_meta=np.genfromtxt(meta_txt + '_Train_whole_id2', dtype="str")

                    train_set_new=train_set.reindex(main_meta) #[force] the data_x 's order in according with id_list. but should be already the same for this multi case.
                    # NOTE: only in this multi-species LOSO case, train_set and train_set_new are exactly the same.
                    ### train_set_new = train_set_new.fillna(0)#the nan part will not be used, because cv folders setting. But sklearn requires numerical type.


                    ###3. Test:
                    main_meta_test= pd.read_csv( meta_txt+'_Test_whole_data.pheno', index_col=0, dtype={'genome_id': object}, sep="\t")
                    main_meta_test = main_meta_test.set_index('genome_id')
                    # print(main_meta_test)
                    test_set['phenotype'] = [main_meta_test.loc[sample, anti] for sample in
                                             test_set.index] # add pheno infor to test set.


                    #------------------------------------------
                    #------------------------------------------
                    X = train_set_new.iloc[:,0:-1].values#the whole set
                    y = train_set_new.iloc[:, -1].values.flatten() #the whole set
                    X_train = train_set.iloc[:, 0:-1].values
                    y_train = train_set.iloc[:,-1].values.flatten()


                    X_test = test_set.iloc[:,0:-1].values
                    y_test = test_set.iloc[:,-1].values.flatten()
                    print(len(X) ,len(X_test))
                    search = GridSearchCV(estimator=pipe, param_grid=hyper_space, n_jobs=n_jobs,
                                          scoring='f1_macro',
                                          cv=create_generator(train_val_train_index), refit=True)

                    search.fit(X, y)
                    hyperparameters_test_sub=search.best_estimator_
                    current_pipe=hyperparameters_test_sub
                    # print(current_pipe)#todo check
                    # -------------------------------------------------------
                    ### retrain on train and val
                    ## only in this multi-species LOSO case, we don't need fit here!
                    # current_pipe.fit(X_train, y_train)
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
                    predictY_test.append( y_test_pred.tolist())
                    true_Y.append(y_test.tolist())
                    sampleNames_test.append(test_set.index.tolist())
                    estimator_test.append(current_pipe)

                    ###-------------------

                    score ={'f1_test':f1_test,'score_report_test':score_report_test,'aucs_test':aucs_test,'mcc_test':mcc_test,
                         'predictY_test':predictY_test,'ture_Y':true_Y,'samples':sampleNames_test}
                    score2= {'hyperparameters_test':hyperparameters_test,'estimator_test':estimator_test}
                    with open(save_name_score + '.json',
                              'w') as f:  # overwrite mode
                        json.dump(score, f)
                    with open(save_name_score  + '.pickle',
                              'wb') as f:  # overwrite mode
                        pickle.dump(score2, f)

            count+=1



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
    # parser.add_argument('-path_sequence', '--path_sequence', type=str, required=False,
    #                     help='Path of the directory with PATRIC sequences')
    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                        help='Directory to store temporary files.')
    # parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
    #                     help=' phylo-tree based cv folders.')
    # parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
    #                     help='kma based cv folders.')
    parser.add_argument('-k', '--kmer', default=13, type=int,
                        help='k-mer')
    parser.add_argument('-l', '--level', default='loose', type=str,
                        help='Quality control: strict or loose')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument('-f_prepare_meta', '--f_prepare_meta', dest='f_prepare_meta', action='store_true',
                        help='Prepare the list files for S2G.')
    parser.add_argument('-f_ml', '--f_ml', dest='f_ml', action='store_true',
                        help='prepare bash')
    parser.add_argument("-cv", "--cv", default=10, type=int,
                        help='CV splits number')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'Pseudomonas aeruginosa\' \
                    \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
                    \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('-n_jobs','--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parsedArgs = parser.parse_args()
    extract_info( parsedArgs.species, parsedArgs.f_all,parsedArgs.kmer, parsedArgs.f_prepare_meta,parsedArgs.f_ml,parsedArgs.cv,\
                  parsedArgs.level,parsedArgs.temp_path,parsedArgs.n_jobs)
