import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["BLIS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import copy
import itertools
import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score, KFold,cross_val_predict,cross_validate
from sklearn import svm,preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC,LinearSVC
from sklearn import svm,preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn import tree
from sklearn.metrics import auc,roc_curve,matthews_corrcoef,confusion_matrix,f1_score,precision_recall_fscore_support,classification_report
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,RandomForestClassifier,ExtraTreesClassifier
from xgboost import XGBClassifier
import xgboost as xgb
# from skopt import BayesSearchCV
# # from skopt.space import Integer
# from skopt.space import Real
# from skopt.space import Categorical
# from skopt.utils import use_named_args
from sklearn.pipeline import Pipeline
import multiprocessing as mp
import amr_utility.name_utility
import amr_utility.graph_utility
import cv_folders.cluster_folders
# import classifier
import time
import pickle
import argparse
import amr_utility.load_data
from itertools import repeat
from sklearn.model_selection import GridSearchCV




def hyper_range(chosen_cl):

    if chosen_cl=='svm':
        cl=SVC(random_state=0,class_weight='balanced')
        hyper_space = {'odh': [True,False],'pca': [True,False], 'canonical': [True,False], 'cutting': [0.5,0.75,False],
                       'kmer': [6,8],  'cl__C': [0.01,0.5, 0.1,1],
                        'cl__gamma': [ 0.01,0.001, 0.0001,'scale','auto'],
                         'cl__kernel': ['rbf', 'poly','sigmoid','linear']}

    if chosen_cl=='lr':
        cl=LogisticRegression( random_state=0,max_iter=10000,class_weight='balanced')
        hyper_space = {'odh': [True,False],'pca': [True, False], 'canonical': [True, False], 'cutting': [0.5,0.75,False],
                       'kmer': [6, 8], 'cl__C': [0.01, 0.5, 0.1,1]}

    if chosen_cl=='lsvm':
        cl=LinearSVC(random_state=0, max_iter=10000,class_weight='balanced')
        hyper_space = {'odh': [True,False],'pca': [True, False], 'canonical': [True, False], 'cutting': [0.5,0.75,False],
                       'kmer': [6, 8], 'cl__C': [0.01, 0.5, 0.1,1]}

    # if chosen_cl=='dt':
    #     cl=tree.DecisionTreeClassifier(random_state=0,class_weight='balanced')
    #     hyper_space = {'odh': [True,False],'pca': [True, False], 'canonical': [True, False], 'cutting': [0.5,0.75,False],
    #                    'kmer': [6, 8], 'max_features': ["auto","sqrt","log2",None],
    #     'max_depth': [10,100, None]}

    if chosen_cl=='rf':
        cl=RandomForestClassifier(random_state=0, class_weight='balanced')
        hyper_space = {'odh': [True,False],'pca': [True, False], 'canonical': [True, False], 'cutting':  [0.5,0.75,False],
                       'kmer': [6, 8], 'cl__max_features': ["auto", "sqrt", "log2", None],'cl__n_estimators':[10,100,500],
        'cl__max_depth': [10,100, None], 'cl__min_samples_split': [2,5,10]}
        # SHhould be ok, July 15
    if chosen_cl=='et':
        cl=ExtraTreesClassifier(random_state=0, class_weight='balanced')
        hyper_space = {'odh': [True,False],'pca': [True, False], 'canonical': [True, False], 'cutting':  [0.5,0.75,False],
                       'kmer': [6, 8], 'cl__max_features': ["sqrt", "log2", None],'cl__n_estimators':[10,100,500],
        'cl__max_depth': [10,100, None], 'cl__min_samples_split': [2,5,10]}
        # n_estimators:The number of trees in the forest.
        # SHhould be ok, July 15
    #============================================
    if chosen_cl=='ab':
        cl=AdaBoostClassifier(random_state=0)
        hyper_space = {'odh': [True,False],'pca': [True, False], 'canonical': [True, False], 'cutting':  [0.5,0.75,False],
                       'kmer': [6, 8], 'cl__learning_rate': [0.001,0.1,1],'cl__n_estimators':[10,100,500]}
        #n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure is stopped early.
        #SHhould be ok, July 15
    if chosen_cl=='gb':
        cl=GradientBoostingClassifier(random_state=0)
        hyper_space = {'odh': [True,False],'pca': [True, False], 'canonical': [True, False], 'cutting':  [0.5,0.75,False],
                       'kmer': [6, 8], 'cl__learning_rate': [0.001,0.1,1], 'cl__max_depth': [5,10,100], 'cl__max_features': ["sqrt", "log2", None],
                       'cl__min_samples_split': [2,5,10],'cl__subsample': [0.5, 1], 'cl__n_estimators':[100,500,1000,10000]}
        #n_estimators:The number of boosting stages to perform.
        # Gradient boosting is fairly robust to over-fitting so a large number usually results in better performance.
        # SHhould be ok, July 15
    # ============================================use model_cv.py for 'xgboost'
    # if chosen_cl == 'xgboost':
    #     # cl = XGBClassifier(seed=0,nthread=1)
    #     cl= xgb.XGBClassifier( random_state=42,n_jobs=1,use_label_encoder=False)
    #     hyper_space = {'odh': [True, False], 'pca': [True, False], 'canonical': [True, False],
    #                    'cutting': [0.5, 0.75, False],
    #                    'kmer': [6, 8], 'learning_rate': [0.01,0.1, 1],'n_estimators' : [10,100, 500],'max_depth': [1, 10, 100],
    #                    'gamma': [0.01, 0.001, 0.0001], 'subsample': [0.5, 1],'colsample_bytree':[0.2,0.5,0.7], 'min_child_weight': [1,10]}
    #

    return hyper_space,cl




def process(odh,kmer,species,anti,canonical,id_train,cutting,pca,f_train,clf_pca,scaler,f_preprocess):
    # import feature matrix
    if odh == True:
        kmer = int(kmer / 2)
        save_name_odh = amr_utility.name_utility.GETsave_name_odh(species,
                                                                   anti, kmer,
                                                                   0,
                                                                   10)  # for odh, min_distance=0, max_distance=10

        data_feature = pd.read_hdf(save_name_odh)
        # print(data_feature.shape)
        # get the 75% quantile

    else:
        # save_name_meta, save_name_modelID = amr_utility.name_utility.GETsave_name_modelID(level,species, anti)

        save_mame_kmc, save_name_kmer = amr_utility.name_utility.GETsave_name_kmer(species,
                                                                                   canonical, kmer)
        data_feature = pd.read_hdf(save_name_kmer)
        data_feature = data_feature.T
        # print(data_feature.shape)


    init_feature = np.zeros((len(id_train), 1), dtype='uint16')
    data_model_init = pd.DataFrame(init_feature, index=id_train, columns=['initializer'])
    # print('data_model_init',data_model_init.shape)
    X_train = pd.concat([data_model_init, data_feature.reindex(data_model_init.index)], axis=1)
    X_train = X_train.drop(['initializer'], axis=1)
    # print('======================')
    # print('X_train',X_train.shape)
    # cut kmer counting >255 all to 255.
    if cutting != False:
        df_quantile = data_feature.to_numpy()
        quantile = np.quantile(df_quantile, cutting)
        quantile = int(quantile)
        # print('quantile',quantile)
        X_train[X_train > quantile] = quantile
    # X = normalize(X, norm='l1')# no use any more. as scaler is better.

    # training , val, and test dataset are scaled based on its own mean and std.
    # Centering and scaling happen independently on each feature by computing the relevant statistics on the samples
    # in the training set. individual features should more or less look like standard normally distributed data.
    if f_preprocess:
        pass
        # if f_train==True:
        #     scaler = preprocessing.StandardScaler().fit(X_train)#default: with_mean=True, with_std=True
        #     X_train = pd.DataFrame(data=scaler.transform(X_train),
        #                            index=X_train.index,
        #                            columns=X_train.columns)
        # else:
        #     X_train=pd.DataFrame(data=scaler.transform(X_train),
        #                            index=X_train.index,
        #                            columns=X_train.columns)
        # X_train = X_train.to_numpy()
        #
        # if pca == True and f_train==True:
        #     clf_pca = PCA()#old:n_components=x_train.shape[0]. defalut:n_components == min(n_samples, n_features) - 1
        #     X_train = clf_pca.fit_transform(X_train)
        #     # print('pca finished...')
        # elif pca == True and f_train==False:
        #     X_train = clf_pca.transform(X_train)
    else:
        X_train = X_train.to_numpy()

    return X_train,clf_pca,scaler

# def rename_keys(d):
#     return dict([(k.split('__')[1], v) for k, v in d.items()])
def fitting_cl(current_pipe,x_train, y_train,x_val):
    start_time = time.time()
    # clf_inner = cl
    # current_hyper=rename_keys(current_hyper)
    # clf_inner.set_params(**current_hyper)
    # scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy',]
    # nested_score_all = cross_validate(clf_inner, X, y, cv=outer_cv, scoring=scoring, n_jobs=1)
    current_pipe.fit(x_train, y_train)
    time_cost = time.time() - start_time
    y_val_pred = current_pipe.predict(x_val)
    # print("--- %s minutes ---" % (time_cost / 60))
    return y_val_pred,time_cost/ 60

def fitting_cl_xgb(current_hyper,cl,x_train, y_train,x_val,y_val): #todo
    '''
    with early stop. xgb only.
    '''
    start_time = time.time()
    # pipeline, parameters = classifier.classifer()  # load model estimaotr and parameter grid settings.
    # set the tolerance to a large value to make the example faster
    optimizing_score = 'f1_macro'
    current_hyper_cl = copy.deepcopy(current_hyper)
    for k in ['pca', 'canonical', 'cutting', 'kmer','odh']:
        del (current_hyper_cl[k])
    clf_inner = cl

    clf_inner.set_params(**current_hyper_cl)

    # scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy',]
    #
    # nested_score_all = cross_validate(clf_inner, X, y, cv=outer_cv, scoring=scoring, n_jobs=1)



    clf_inner.fit(x_train, y_train, early_stopping_rounds=50,eval_set=[(x_val, y_val)],verbose=False,eval_metric='logloss')

    time_cost = time.time() - start_time

    y_val_pred = clf_inner.predict(x_val)
    # print("--- %s minutes ---" % (time_cost / 60))
    return y_val_pred,time_cost/ 60


# def grid_run(hyper_space, species, anti,id_train,id_val,cl,y_train,y_val,chosen_cl):
def load_feature(hyper_space, species, anti,id_train,chosen_cl):
    # print('num_hyper', num_hyper, '================>', grid_iteration)
    current_hyper = hyper_space
    pca =hyper_space['pca']
    canonical = hyper_space['canonical']
    cutting =hyper_space['cutting']
    kmer =hyper_space['kmer']
    odh = hyper_space['odh']
    # print(current_hyper)

    # jump over the situation when pca is true and cutting is not false
    if chosen_cl in ['svm','lr']:
        if (current_hyper['pca'] == True and current_hyper['cutting'] != False) or ( current_hyper['kmer'] == 8 and current_hyper['pca'] == False) \
                or ( current_hyper['kmer'] == 6 and current_hyper['pca'] == False and current_hyper['odh'] == True):
            x_train=None
            # print('not trying this hyper-para')
        else:
            x_train, clf_pca,clf_scaler = process(odh, kmer, species, anti, canonical, id_train, cutting, pca, True, None, None, False)
    else:
        x_train, clf_pca, clf_scaler = process(odh, kmer, species, anti, canonical, id_train, cutting, pca, True, None,
                                               None, False)

    return x_train
def create_generator(nFolds):
    for idx in range(len(nFolds)):
        test =nFolds[idx]
        train = list(itertools.chain(*[fold for idy, fold in enumerate(nFolds) if idy != idx]))
        yield train, test
def model(level,species,cl_list,antibiotics,cv,n_jobs,f_phylotree,f_kma):
    '''
    Nested CV, f1_macro based selection.
    '''
    antibiotics_selected = ast.literal_eval(antibiotics)

    print(species)
    print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)
    antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
    # for anti in ['mupirocin', 'penicillin', 'rifampin', 'tetracycline', 'vancomycin']:
    i_anti=0
    for anti in antibiotics:

        save_name_meta, p_names = amr_utility.name_utility.GETsave_name_modelID(level, species, anti)
        Random_State=42
        p_clusters= amr_utility.name_utility.GETname_folder(species,anti,level)
        if f_phylotree:  # phylo-tree based cv folders
            folders_index = cv_folders.cluster_folders.prepare_folders_tree(cv, species, antibiotics, p_names,
                                                                                  False)
        else:  # kma cluster based cv folders
            folders_index, _, _ = cv_folders.cluster_folders.prepare_folders(cv, Random_State, p_names,
                                                                                   p_clusters,
                                                                                   'new')
        # folders_index, _, _ = cv_folders.cluster_folders.prepare_folders(cv, Random_State, p_names, p_clusters,
        #                                                                        'new')#index
        id_all = ID[i_anti]#sample name list, e.g. [1352.10013,1352.10014,1354.10,1366.10]
        y_all = Y[i_anti]
        id_all =np.array(id_all)
        y_all = np.array(y_all)
        # for chosen_cl in ['svm','lr','lsvm','rf','et','ab','gb','xgboost']:
        # for chosen_cl in ['svm', 'lr', 'lsvm', 'rf', 'et', 'ab', 'gb']:
        # for chosen_cl in ['lsvm', 'rf', 'et', 'ab', 'gb']:
        # for chosen_cl in ['svm','lr', 'rf']:
        for chosen_cl in ['rf']:
        # for chosen_cl in cl_list:
            hyper_space,cl=hyper_range(chosen_cl)
            # 1. by each classifier.2. by outer loop. 3. by inner loop. 4. by each hyper-para
            save_name_score = amr_utility.name_utility.GETsave_name_score(species, anti, chosen_cl)

            mcc_test = []  # MCC results for the test data
            f1_test = []
            score_report_test = []
            aucs_test = []
            hyperparameters_test = []
            hyperparameters2_test = []
            time_test=[]
            for out_cv in range(cv):
                print('Starting outer: ', str(out_cv), '; chosen_cl: ', chosen_cl)
                test_samples_index = folders_index[out_cv]# a list of index
                # print(test_samples)
                # print(id_all)
                id_test = id_all[test_samples_index]#sample name list
                y_test = y_all[test_samples_index]

                train_val_train_index =folders_index[:out_cv] +folders_index[out_cv + 1:]
                id_val_train = id_all[list(itertools.chain.from_iterable(train_val_train_index))]  # sample name list
                y_val_train = y_all[list(itertools.chain.from_iterable(train_val_train_index))]

                hyperparameters_test_sub=[]
                Validation_f1=[]


                #separate 'pca', 'canonical', 'cutting', 'kmer', 'odh' hyper-para from classifier related hyper-para
                search_space = copy.deepcopy(hyper_space)
                for k in ['pca', 'canonical', 'cutting', 'kmer', 'odh']:
                    del (search_space[k])
                search_space_feature = dict((k, hyper_space[k]) for k in ('pca', 'canonical', 'cutting', 'kmer', 'odh'))
                # print('check---------> search_space_feature:',search_space_feature)
                hyper_list_feature = list(ParameterGrid(search_space_feature))

                for current_hyper_f in hyper_list_feature:
                    x_all=load_feature(current_hyper_f, species, anti, id_all,chosen_cl)#note: use id_all, because the cv folder use index w.r.t. id_all.
                    if current_hyper_f['pca']==True and chosen_cl in ['svm','lr']:
                        pca = PCA()
                        scaler = preprocessing.StandardScaler()
                        pipe = Pipeline(steps=[('scaler', scaler), ('pca', pca), ('cl', cl)])
                    elif current_hyper_f['pca']==True and chosen_cl in ['rf','et','ab','gb','xgboost']:
                        pca = PCA()
                        # scaler = preprocessing.StandardScaler()
                        pipe = Pipeline(steps=[('pca', pca), ('cl', cl)])
                    elif current_hyper_f['pca']==False and chosen_cl in ['svm','lr']:
                        scaler = preprocessing.StandardScaler()
                        pipe = Pipeline(steps=[('scaler', scaler), ('cl', cl)])
                    else:
                        pipe = Pipeline(steps=[('cl', cl)])

                    #---------------------------------------------------------------------------------------------------
                    if x_all is None:
                        hyperparameters_test_sub.append(None)
                        Validation_f1.append(0)
                    else:
                        # print('x_val_train.shape',x_val_train.shape)
                        # print('len(y_val_train)',len(y_val_train))
                        # print(train_val_train_index)
                        # from joblib import parallel_backend #not efficient!
                        # with parallel_backend('threading', n_jobs=n_jobs):
                        if f_phylotree:
                            pass
                        elif f_kma:
                            search = GridSearchCV(estimator=pipe, param_grid=search_space, n_jobs=n_jobs,
                                                  scoring='f1_macro',
                                                  cv=create_generator(train_val_train_index), refit=True)
                        else:
                            search = GridSearchCV(estimator=pipe, param_grid=search_space, n_jobs=n_jobs,
                                                  scoring='f1_macro',
                                                  cv=9, refit=True)
                        search.fit(x_all, y_all)  # note: use id_all, because the cv folder use index w.r.t. id_all.

                        # hyper_cl=search.best_estimator_
                        hyperparameters_test_sub.append(search.best_estimator_)
                        Validation_f1.append(search.best_score_)
                        print('--')

                #select the best hyper_space
                # print('outer--------------------',out_cv)
                # print('Validation_f1:',Validation_f1)
                # print('hyperparameters_test_sub:',hyperparameters_test_sub)
                Validation_f1 = np.array(Validation_f1)
                ind = np.unravel_index(np.argmax(Validation_f1, axis=None), Validation_f1.shape)
                # print(ind)#todo check
                # print('check Validation_f1_grid',Validation_f1.shape,'check ind',ind)
                hyperparameters_test.append(hyper_list_feature[ind[0]]) #'pca', 'canonical', 'cutting', 'kmer', 'odh'
                hyperparameters2_test.append(hyperparameters_test_sub[ind[0]]) #classifier related hyper-para
                # -------------------------------------------------------
                #retrain on train and val
                # current_hyper =  hyperparameters_test[out_cv]
                current_pipe = hyperparameters2_test[out_cv]
                pca = hyperparameters_test[out_cv]['pca']
                canonical =hyperparameters_test[out_cv]['canonical']
                cutting = hyperparameters_test[out_cv]['cutting']
                kmer =hyperparameters_test[out_cv]['kmer']
                odh =hyperparameters_test[out_cv]['odh']
                # preprocess the data
                x_val_train,_,_ = process(odh, kmer, species, anti, canonical, id_val_train, cutting, pca,True,None,None,False)
                x_test,_,_=process(odh, kmer, species, anti, canonical, id_test, cutting, pca,False,None,None,False)
                # fitting,predicting
                print(current_pipe)
                y_test_pred, time_min = fitting_cl(current_pipe, x_val_train, y_val_train, x_test)

                # scores
                f1 = f1_score(y_test, y_test_pred, average='macro')
                report = classification_report(y_test, y_test_pred, labels=[0, 1], output_dict=True)
                mcc = matthews_corrcoef(y_test, y_test_pred)
                fpr, tpr, _ = roc_curve(y_test, y_test_pred, pos_label=1)
                # tprs.append(interp(mean_fpr, fpr, tpr))
                # tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)

                f1_test.append(f1)
                score_report_test.append(report)
                aucs_test.append(roc_auc)
                mcc_test.append(mcc)
                time_test.append(time_min)
            score = [f1_test,score_report_test,aucs_test,mcc_test, hyperparameters_test,hyperparameters2_test,time_test]
            with open(save_name_score+'_kma_'+str(f_kma)+'_tree_'+str(f_phylotree)+'.pickle', 'wb') as f:  # overwrite mode
                pickle.dump(score, f)
        #finish one anti
        i_anti+=1
    #finish one species



def extract_info(l,s,cl,cv,n_jobs,f_phylotree,f_kma):
    data = pd.read_csv('metadata/' + str(l) + '_Species_antibiotic_FineQuality.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
    # data = pd.read_csv('metadata/Species_antibiotic_FineQuality.csv', index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]# drop the species with 0 in column 'number'.
    # for training model on part of the dataset:-------------
    # data=data.loc[['Escherichia coli'],:]
    data = data.loc[s, :]
    #data.at['Mycobacterium tuberculosis', 'modelling antibiotics']=['capreomycin', 'ciprofloxacin']
    # --------------------------------------------------------
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    print(data)

    for df_species,antibiotics in zip(df_species, antibiotics):
        model(l,df_species,cl, antibiotics,cv,n_jobs,f_phylotree,f_kma)

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-clf', '--Model_classifier', default='svm', type=str, required=True,
    #                     help='svm,logistic,lsvm,all')
    parser.add_argument('--l', '--level', default='loose', type=str, required=False,
                        help='Quality control: strict or loose')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                        help='kma based cv folders.')

    # parser.add_argument('--k','--Kmer',default=8, type=int, required=True,
    #                     help='Kmer size')
    # parser.add_argument('-p', '--pca', dest='pca',
    #                     help='Use pca', action='store_true', )
    # parser.add_argument('-b', '--balance', dest='balance',
    #                     help='use downsampling or not ', action='store_true', )#default:false
    # parser.add_argument('-o', '--odh', dest='odh',
    #                     help='Use odh features',action='store_true',)#default:false
    # parser.add_argument('--m', default=0, type=int,help='odh features: min_distance. Default=0')
    # parser.add_argument('--d', default=None, type=int,help='odh features: distance. Default=None')
    # parser.add_argument('-c','--canonical', dest='canonical',action='store_true',
    #                     help='Canonical kmer or not: True')
    # parser.add_argument('-n','--non_canonical',dest='canonical',action='store_false',
    #                     help='Canonical kmer or not: False')#default:false
    # parser.add_argument('-cut', '--cutting', dest='cutting', action='store_true',
    #                     help='cutting or not: True')
    # parser.add_argument('-ncut', '--non_cutting', dest='cutting', action='store_false',
    #                     help='cutting or not: False')#default:false
    parser.add_argument("-cv", "--cv_number", default=10, type=int,
                        help='CV splits number')
    parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parser.add_argument('-s','--species', default=[], type=str,nargs='+',help='species to run: e.g.\'seudomonas aeruginosa\' \
     \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
     \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('-cl', '--cl', default=['xgboost'], type=str, nargs='+', help='classifier to train: e.g.\'svm\',\'lr\',\'lsvm\','
                                                                             '\'dt\',\'rf\',\'et\',\'ab\',\'gb\',\'xgboost\'')
    # parser.add_argument('--e','--estimator', default='nl_svm', type=str, required=True,
    #                     help='estimator: nl_svm,LogisticRegression,todo more.. ')
    #parser.set_defaults(canonical=True)
    parsedArgs=parser.parse_args()
    # parser.print_help()
    print(parsedArgs)
    extract_info(parsedArgs.l,parsedArgs.species,parsedArgs.cl,parsedArgs.cv_number,parsedArgs.n_jobs,parsedArgs.f_phylotree,parsedArgs.f_kma)
