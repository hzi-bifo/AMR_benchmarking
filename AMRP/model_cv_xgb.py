import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
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
from sklearn.pipeline import Pipeline
import multiprocessing as mp
import tqdm
import amr_utility.name_utility
import amr_utility.graph_utility
import cv_folders.cluster_folders
# import classifier
import time
import pickle
import argparse
import amr_utility.load_data
from itertools import repeat





def hyper_range(chosen_cl,n_jobs,pnr):

    if chosen_cl == 'xgboost':
        # cl = XGBClassifier(seed=0,nthread=1)
        cl= xgb.XGBClassifier(random_state=42,n_jobs=1,use_label_encoder=False,scale_pos_weight=pnr)#TODO scale_pos_weight=
        hyper_space = {'odh': [False], 'pca': [True, False], 'canonical': [True, False],
                       'cutting': [0.5, 0.75, False],
                       'kmer': [6, 8], 'learning_rate': [0.01,0.1, 1],'n_estimators' : [500],'max_depth': [1, 10, 100],
                       'gamma': [0.01, 0.001, 0.0001], 'subsample': [0.5, 1],'colsample_bytree':[0.2,0.5,0.7], 'min_child_weight': [1,10]}
    return hyper_space,cl



def process(odh,kmer,species,anti,canonical,id_train,cutting,pca,f_train,clf_pca,scaler):
    # import feature matrix
    if odh == True:
        kmer = int(kmer / 2)
        save_name_odh = amr_utility.name_utility.GETsave_name_odh(species,anti, kmer,0,10)  # for odh, min_distance=0, max_distance=10

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


    # print('data_feature imported....')
    # print(data_feature)
    # normalize feature
    # data_feature = normalize(data_feature, norm='l1')#
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


    if f_train==True:
        scaler = preprocessing.StandardScaler(
            with_mean=True, with_std=True).fit(X_train)
        X_train = pd.DataFrame(data=scaler.transform(X_train),
                               index=X_train.index,
                               columns=X_train.columns)
    else:
        X_train=pd.DataFrame(data=scaler.transform(X_train),
                               index=X_train.index,
                               columns=X_train.columns)
    x_train = X_train.to_numpy()

    if pca == True and f_train==True:
        clf_pca = PCA(n_components=x_train.shape[0])
        x_train = clf_pca.fit_transform(x_train)
        # print('pca finished...')
    elif pca == True and f_train==False:
        x_train = clf_pca.transform(x_train)

    return x_train,clf_pca,scaler




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
    iteration=clf_inner.best_iteration
    # sliced = clf_inner.get_booster()[: iteration]
    # # time_cost = time.time() - start_time
    # x_val = xgb.DMatrix(x_val)
    # y_val_pred = sliced.predict(x_val)
    # print(y_val_pred)
    # print(clf_inner.get_num_boosting_rounds())
    y_val_pred = clf_inner.predict(x_val)
    # print("--- %s minutes ---" % (time_cost / 60))
    return y_val_pred,iteration


def grid_run(hyper_value, species, anti,id_train,id_val,cl,y_train,y_val):
    # print('num_hyper', num_hyper, '================>', grid_iteration)
    current_hyper = hyper_value
    pca =hyper_value['pca']
    canonical = hyper_value['canonical']
    cutting =hyper_value['cutting']
    kmer =hyper_value['kmer']
    odh = hyper_value['odh']
    # print(current_hyper)

    # jump over the situation when pca is true and cutting is not false
    # if (current_hyper['pca'] == True and current_hyper['cutting'] != False) or ( current_hyper['kmer'] == 8 and current_hyper['pca'] == False) \
    #         or ( current_hyper['kmer'] == 6 and current_hyper['pca'] == False and current_hyper['odh'] == True):
    if ( current_hyper['kmer'] == 8 and current_hyper['pca'] == False) :
        # Validation_f1_grid.append(0)
        f1_val=0
        best_iteration=None
        # print('not trying this hyper-para')
    else:
        # ===================================================================
        # preprocess the data
        x_train, clf_pca,clf_scaler = process(odh, kmer, species, anti, canonical, id_train, cutting, pca, True, None,None)
        x_val, _,_ = process(odh, kmer, species, anti, canonical, id_val, cutting, pca, False, clf_pca,clf_scaler)
        # print('sample length and feature length for inner CV(train & val):', len(x_train),
        #       len(x_train[0]), len(x_val), len(x_val[0]))

        # fitting,predicting

        y_val_pred, best_iteration = fitting_cl_xgb(current_hyper, cl, x_train, y_train, x_val,y_val)
        # print('best_iteration',best_iteration)
        # scores
        f1_val = f1_score(y_val, y_val_pred, average='macro')
        # Validation_f1_grid.append(f1_val)

        # print('-----------------------------------------------------------')
        # ===================================================================
    return f1_val, best_iteration
def fitting_cl(current_hyper,cl,x_train, y_train,x_val):
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
    clf_inner.fit(x_train, y_train)

    time_cost = time.time() - start_time

    y_val_pred = clf_inner.predict(x_val)
    # print("--- %s minutes ---" % (time_cost / 60))
    return y_val_pred,time_cost/ 60

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
    # for anti in ['streptomycin', 'sulfisoxazole', 'tetracycline']:

        save_name_meta, p_names = amr_utility.name_utility.GETsave_name_modelID(level, species, anti)
        Random_State=42
        p_clusters= amr_utility.name_utility.GETname_folder(species,anti,level)
        if f_phylotree:  # phylo-tree based cv folders
            folders_index = cv_folders.cluster_folders.prepare_folders_tree(cv, species, anti, p_names,
                                                                                  False)
        elif f_kma:  # kma cluster based cv folders
            folders_index, _, _ = cv_folders.cluster_folders.prepare_folders(cv, Random_State, p_names,
                                                                                   p_clusters,
                                                                                   'new')
        else:#random
            folders_index = cv_folders.cluster_folders.prepare_folders_random(cv, species, anti, p_names,
                                                                                  False)


        id_all = ID[i_anti]#sample name list
        y_all = Y[i_anti]
        id_all =np.array(id_all)
        y_all = np.array(y_all)
        for chosen_cl in ['xgboost']:

        # for chosen_cl in cl_list:

            # 1. by each classifier.2. by outer loop. 3. by inner loop. 4. by each hyper-para
            save_name_score = amr_utility.name_utility.GETsave_name_score(species, anti, chosen_cl)

            mcc_test = []  # MCC results for the test data
            f1_test = []
            score_report_test = []
            aucs_test = []
            hyperparameters_test = []
            iteration_test=[]
            ntree_limit_test=[]
            time_test=[]
            for out_cv in range(cv):
                test_samples = folders_index[out_cv]# a list of index
                # print(test_samples)
                # print(id_all)
                id_test = id_all[test_samples]#sample name list
                y_test = y_all[test_samples]
                train_val_train_index =folders_index[:out_cv] +folders_index[out_cv + 1:]

                id_val_train = id_all[list(itertools.chain.from_iterable(train_val_train_index))]  # sample name list
                y_val_train = y_all[list(itertools.chain.from_iterable(train_val_train_index))]

                Validation_f1=[]
                Validation_best_iteration=[]
                # Validation_best_ntree_limit=[]
                for innerCV in range(cv - 1):  # e.g. 1,2,3,4
                    print('Starting outer: ', str(out_cv), '; inner: ', str(innerCV), ' inner loop...',chosen_cl)
                    val_samples = train_val_train_index[innerCV]
                    train_samples = train_val_train_index[:innerCV] + train_val_train_index[
                                                                  innerCV + 1:]  # only works for list, not np
                    train_samples = list(itertools.chain.from_iterable(train_samples))
                    id_train, id_val = id_all[train_samples], id_all[val_samples]  # only np
                    y_train, y_val = y_all[train_samples], y_all[val_samples]

                    # Validation_f1_grid=[]
                    # pnr: Control the balance of positive and negative weights, useful for unbalanced classes.
                    # A typical value to consider: sum(negative instances) / sum(positive instances)
                    pnr=y_train.tolist().count(0)/y_train.tolist().count(1)
                    hyper_space,cl=hyper_range(chosen_cl,n_jobs,pnr)
                    hyper_list=list(ParameterGrid(hyper_space))
                    num_hyper = len(list(ParameterGrid(hyper_space)))

                    pool = mp.Pool(processes=n_jobs)
                    inputs=zip(hyper_list, repeat(species),repeat(anti),repeat(id_train),
                                                                  repeat(id_val),repeat(cl),repeat(y_train),repeat(y_val))
                    Validation_f1_grid,best_iteration_grid=pool.starmap(grid_run, tqdm.tqdm(inputs, total=num_hyper),chunksize=1)
                    pool.close()
                    pool.join()
                    print(best_iteration_grid)
                    # Validation_f1_grid=[]
                    # best_iteration_grid,best_ntree_limit_grid=[],[]
                    # count=0
                    # for grid_iteration in np.arange(num_hyper):
                    #     print(count,'/', num_hyper)
                    #     count+=1
                    #     hyper_value=hyper_list[grid_iteration]
                    #     f1_val,best_iteration=grid_run(hyper_value, species, anti,id_train,id_val,cl,y_train,y_val)
                    #     Validation_f1_grid.append(f1_val)
                    #     best_iteration_grid.append(best_iteration)
                    #     # best_ntree_limit_grid.append(best_ntree_limit)
                    #hyper_space, species, anti,id_train,id_val,cl,y_train,y_val
                    # print('check pool output',len(Validation_f1_grid))
                    Validation_f1.append(Validation_f1_grid)#todo need check.
                    # for grid_iteration in np.arange(len(list(ParameterGrid(hyper_space)))):
                    Validation_best_iteration.append(best_iteration_grid)
                    # Validation_best_ntree_limit.append(best_ntree_limit_grid)
                    # #finish grid.
                #finish inner loop. Select the best hyper set.

                Validation_f1 = np.array(Validation_f1)
                print('check Validation_f1',Validation_f1.shape)
                Validation_f1=Validation_f1.mean(axis=0)
                Validation_best_iteration=np.array(Validation_best_iteration)
                Validation_best_iteration=Validation_best_iteration.mean(axis=0)
                # Validation_best_ntree_limit=np.array(Validation_best_ntree_limit)
                # Validation_best_ntree_limit=Validation_best_ntree_limit.mean(axis=0)

                ind=np.unravel_index(np.argmax(Validation_f1, axis=None), Validation_f1.shape)
                print('check Validation_f1',Validation_f1.shape,'check ind',ind)
                hyperparameters_test.append(hyper_list[ind[0]])
                iteration_test.append(Validation_best_iteration[ind[0]])
                # ntree_limit_test.append(Validation_best_ntree_limit[ind[0]])

                #retrain on train and val
                current_hyper =  hyperparameters_test[out_cv]
                current_hyper['n_estimators']=iteration_test[out_cv]#because of early stop
                hyperparameters_test[out_cv]['n_estimators']=iteration_test[out_cv]#only for the sake of hyperparameter storing.
                # current_hyper['ntree_limit']=ntree_limit_test[out_cv]
                pca = hyperparameters_test[out_cv]['pca']
                canonical =hyperparameters_test[out_cv]['canonical']
                cutting = hyperparameters_test[out_cv]['cutting']
                kmer =hyperparameters_test[out_cv]['kmer']
                odh =hyperparameters_test[out_cv]['odh']
                # preprocess the data
                x_val_train,clf_pca,clf_scaler = process(odh, kmer, species, anti, canonical, id_val_train, cutting, pca,True,None,None)
                x_test,_,_=process(odh, kmer, species, anti, canonical, id_test, cutting, pca,False,clf_pca,clf_scaler)
                # fitting,predicting
                pnr_outer=y_val_train.tolist().count(0)/y_val_train.tolist().count(1)
                current_hyper['scale_pos_weight']=pnr_outer
                y_test_pred, time_min = fitting_cl(current_hyper, cl, x_val_train, y_val_train, x_test)

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
            score = [f1_test,score_report_test,aucs_test,mcc_test, hyperparameters_test,time_test]
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
