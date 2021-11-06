import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["BLIS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import argparse
import pickle
import amr_utility.name_utility
import amr_utility.graph_utility
import amr_utility.file_utility
import amr_utility.load_data
import numpy as np
import pandas as pd
import subprocess
import shutil
from pathlib import Path
import cv_folders.cluster_folders
import itertools
import math
from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC,LinearSVC
from sklearn import svm,preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc,roc_curve,matthews_corrcoef,confusion_matrix,f1_score,precision_recall_fscore_support,classification_report
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,RandomForestClassifier,ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.pipeline import Pipeline


def extract_info(path_sequence,s,f_all,f_prepare_meta,f_tree,cv,level,n_jobs,f_ml,f_phylotree,f_kma,f_qsub):

    # if path_sequence=='/net/projects/BIFO/patric_genome':
    #     path_large_temp='/net/sgi/metagenomics/data/khu/benchmarking/s2g2p'#todo, may need a change
    # else:
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    # path_large_temp = os.path.join(fileDir, 'large_temp')
    # # print(path_large_temp)
    #
    # amr_utility.file_utility.make_dir(path_large_temp)

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
    # print(data)


    for species in df_species:
        amr_utility.file_utility.make_dir('log/temp/' + str(level) + '/'+str(species.replace(" ", "_")))
        amr_utility.file_utility.make_dir('log/results/' + str(level) +'/'+ str(species.replace(" ", "_")))


    if f_prepare_meta:
        #prepare the dna list accroding to S2G format.
        for species, antibiotics in zip(df_species, antibiotics):

            antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
            i_anti = 0

            run_file = open('log/temp/' + str(level) +'/'+ str(species.replace(" ", "_"))+'/'+'getfeature.sh', "w")
            run_file.write("#!/bin/bash")
            run_file.write("\n")
            # if path_sequence == '/vol/projects/BIFO/patric_genome':
            if Path(fileDir).parts[1] == 'vol':
                #
                # run_file = amr_utility.file_utility.hzi_cpu_header4(run_file,
                #                                                     str(species.replace(" ", "_"))+'seeker',
                #                                                     8, 'PhenotypeSeeker2','all.q')
                run_file.write("export PYTHONPATH=/vol/projects/khu/amr/PhenotypeSeeker\n")
                run_file.write("#")
            else:
                run_file.write("export PYTHONPATH=/net/sgi/metagenomics/data/khu/benchmarking/PhenotypeSeeker")


            # run_file.write("\n")
            for anti in antibiotics:
                # prepare features for each training and testing sets, to prevent information leakage
                id_all = ID[i_anti]#sample names
                i_anti += 1
                id_all = np.array(id_all)
                print(anti)
                print(id_all.shape)
                print('-------------')
                for out_cv in range(cv):

                    #1. exrtact CV folders----------------------------------------------------------------
                    save_name_meta, p_names = amr_utility.name_utility.GETsave_name_modelID(level, species, anti)
                    Random_State = 42
                    p_clusters = amr_utility.name_utility.GETname_folder(species, anti, level)
                    if f_phylotree:  # phylo-tree based cv folders
                        folders_index = cv_folders.cluster_folders.prepare_folders_tree(cv, species, antibiotics,
                                                                                        p_names,
                                                                                        False)
                    else:  # kma cluster based cv folders
                        folders_index, _, _ = cv_folders.cluster_folders.prepare_folders(cv, Random_State, p_names,
                                                                                         p_clusters,
                                                                                         'new')
                    test_samples_index = folders_index[out_cv]
                    train_val_train_index = folders_index[:out_cv] + folders_index[out_cv + 1:]
                    id_val_train = id_all[list(itertools.chain.from_iterable(train_val_train_index))]# sample name list
                    id_test = id_all[test_samples_index]

                    #2. prepare meta files for this round of training samples-------------------

                    name,meta_txt,_=amr_utility.name_utility.Pts_GETname(level, species, anti,'')
                    name_list = pd.read_csv(name, index_col=0, dtype={'genome_id': object}, sep="\t")

                    if Path(fileDir).parts[1] == 'vol':
                        # path_list=np.genfromtxt(path, dtype="str")
                        name_list['Addresses'] = '/vol/projects/BIFO/patric_genome/' + name_list['genome_id'].astype(str)+'.fna'
                    else:
                        name_list['Addresses']= '/net/projects/BIFO/patric_genome/' + name_list['genome_id'].astype(str)+'.fna'
                    # only retain those in the training and validataion CV folders
                    name_list_train=name_list.loc[name_list['genome_id'].isin(id_val_train)]

                    name_list_train['ID'] = 'iso_' + name_list_train['genome_id'].astype(str)

                    name_list_train.rename(columns={'resistant_phenotype': str(
        anti.translate(str.maketrans({'/': '_', ' ': '_'})))}, inplace=True)
                    name_list_train = name_list_train.loc[:, ['ID', 'Addresses',str(
        anti.translate(str.maketrans({'/': '_', ' ': '_'})))]]
                    name_list_train.to_csv(meta_txt +'_Train_'+ str(out_cv) + '_data.pheno', sep="\t", index=False,
                                             header=True)

                    # 3. prepare meta files for this round of testing samples-------------------


                    # only retain those in the training and validataion CV folders
                    name_list_test = name_list.loc[name_list['genome_id'].isin(id_test)]

                    name_list_test['ID'] = 'iso_' + name_list_test['genome_id'].astype(str)

                    name_list_test.rename(columns={'resistant_phenotype': str(
        anti.translate(str.maketrans({'/': '_', ' ': '_'})))}, inplace=True)
                    name_list_test = name_list_test.loc[:, ['ID', 'Addresses', str(
        anti.translate(str.maketrans({'/': '_', ' ': '_'})))]]
                    name_list_test.to_csv(meta_txt + '_Test_' + str(out_cv) + '_data.pheno', sep="\t", index=False,
                                     header=True)# note: when running this set, set pval_limit==1.0. --pvalue 1.0


                    #4. prepare bash

        #             cmd = 'python %s/scripts/phenotypeseeker_new modeling  ' \
        #                   '--no_assembly --f_train_test %s --cv_K %s %s' % (fileDir,'train',out_cv,str(
        # anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'_Train_'+ str(out_cv) + '_data.pheno')
        #             run_file.write(cmd)
        #             run_file.write("\n")
                    cmd = 'python %s/scripts/phenotypeseeker_new modeling  ' \
                          '--no_assembly --f_train_test %s --no_weights --cv_K %s --pvalue 1 --n_kmers -1 %s' % (fileDir,'test',out_cv, str(
        anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'_Test_' + str(out_cv) + '_data.pheno')
                    run_file.write(cmd)
                    run_file.write("\n")



    if f_tree == True:

            pass
    if f_qsub:#prepare bash scripts for each species for ML
        for species, antibiotics in zip(df_species, antibiotics):
            amr_utility.file_utility.make_dir('log/qsub')
            run_file_name='log/qsub/'+str(species.replace(" ", "_"))+'_kmer.sh'
            amr_utility.file_utility.make_dir('log/qsub')
            run_file = open(run_file_name, "w")
            run_file.write("#!/bin/bash")
            run_file.write("\n")
            # if path_sequence == '/vol/projects/BIFO/patric_genome':
            if Path(fileDir).parts[1] == 'vol':
                run_file = amr_utility.file_utility.hzi_cpu_header4(run_file,
                                                                    str(species.replace(" ", "_"))+'kmer',
                                                                    100, 'amr','uv2000.q')
            # run_file = amr_utility.file_utility.header_THREADS(run_file,
            #                                                    n_jobs)
            cmd = 'python main_bench.py -f_ml --n_jobs %s -s \'%s\' -f_kma' % (100,species)
            run_file.write(cmd)
            run_file.write("\n")

            #------------------------------------------------------------
            run_file_name = 'log/qsub/' + str(species.replace(" ", "_")) + '_kmer2.sh'
            run_file = open(run_file_name, "w")
            run_file.write("#!/bin/bash")
            run_file.write("\n")
            # if path_sequence == '/vol/projects/BIFO/patric_genome':
            if Path(fileDir).parts[1] == 'vol':
                run_file = amr_utility.file_utility.hzi_cpu_header4(run_file,
                                                                    str(species.replace(" ", "_")+'kmer'),
                                                                    20, 'amr', 'all.q')
            # run_file = amr_utility.file_utility.header_THREADS(run_file,
            #                                                     20)
            cmd = 'python main_bench.py -f_ml --n_jobs %s -s \'%s\' -f_kma' % (20, species)
            run_file.write(cmd)
            run_file.write("\n")

    if f_ml: #ML
        for species, antibiotics in zip(df_species, antibiotics):
            antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
            # antibiotics=antibiotics[2:]
            for anti in antibiotics:

                for chosen_cl in ['svm', 'lr','rf']:
                    hyper_space, cl = hyper_range(chosen_cl)

                    mcc_test = []  # MCC results for the test data
                    f1_test = []
                    score_report_test = []
                    aucs_test = []
                    hyperparameters_test = []
                    meta_original, meta_txt,save_name_score=amr_utility.name_utility.Pts_GETname(level, species, anti,chosen_cl)
                    

                    for out_cv in range(cv):
                        print('Starting outer: ', str(out_cv), '; chosen_cl: ', chosen_cl)
                        
                        train_set=pd.read_csv(meta_txt+ "_" + str(out_cv) +"_train_df.csv",index_col=0)
                        test_set=pd.read_csv(meta_txt+ "_" + str(out_cv) + "_test_df_beforecutting.csv",index_col=0)
                        train_set = train_set.drop(train_set.tail(1).index)
                        test_set = test_set.drop(test_set.tail(1).index)
                        # hyper_list_feature = list(ParameterGrid(hyper_space))
                        pipe = Pipeline(steps=[('cl', cl)])


                        #sort the matrix by index order.
                        save_name_meta, p_names = amr_utility.name_utility.GETsave_name_modelID(level, species, anti)
                        Random_State = 42
                        p_clusters = amr_utility.name_utility.GETname_folder(species, anti, level)
                        if f_phylotree:  # phylo-tree based cv folders
                            folders_index = cv_folders.cluster_folders.prepare_folders_tree(cv, species, antibiotics,
                                                                                            p_names,
                                                                                            False)
                        else:  # kma cluster based cv folders
                            f_kma=True# sofar, we can set it automatically, if random cv folders are not considered.
                            folders_index, _, _ = cv_folders.cluster_folders.prepare_folders(cv, Random_State, p_names,
                                                                                            p_clusters,
                                                                                             'new')
                        # id_all = ID[i_anti]  # sample name list, e.g. [1352.10013,1352.10014,1354.10,1366.10]
                        # y_all = Y[i_anti]
                        # i_anti+=1
                        # id_all = np.array(id_all)
                        # y_all = np.array(y_all)

                        train_val_train_index = folders_index[:out_cv] + folders_index[out_cv + 1:]
                        # id_val_train = id_all[
                        #     list(itertools.chain.from_iterable(train_val_train_index))]  # sample name list
                        # # y_val_train = y_all[list(itertools.chain.from_iterable(train_val_train_index))]
                        #


                        main_meta = pd.read_csv(meta_original, index_col=0, dtype={'genome_id': object}, sep="\t")
                        main_meta['ID'] = 'iso_' + main_meta['genome_id'].astype(str)
                        tain_val_test_set_order=main_meta.loc[:,['ID']]
                        main_meta = main_meta.set_index('ID')
                        # map train_set to tain_val_test_   set_order, note there is some Nan data, which is normal, means belonging to test set.
                        #todo check

                        tain_val_test_set_order=tain_val_test_set_order.set_index('ID')
                        train_set_new=pd.merge(tain_val_test_set_order, train_set, left_index=True, right_index=True, how="outer")
                        train_set_new=train_set_new.reindex(main_meta.index) #force the data_x 's order in according with id_list

                        train_set_new = train_set_new.fillna(0)#the nan part will not be used, because cv folders setting. But sklearn requires numerical type.
                        # Those ann indicates samples bolong to the testing set.

                        #map test set to tain_val_test_set_order('ID') and train_set(columns)
                        test_set_new=pd.DataFrame(index=test_set.index,columns=train_set.columns.to_list()[:-1])
                        test_set_new['phenotype'] = [main_meta.loc[sample, 'resistant_phenotype'] for sample in
                                                 test_set.index] # add pheno infor to test set.
                        for col in train_set.columns.to_list()[:-2]:
                            if col in test_set.columns.to_list():
                                test_set_new[col]=test_set[col]
                        # print(test_set_new)

                        test_set_new=test_set_new.fillna(0) #fill nan with 0
                        # print(test_set_new)
                        # test_set_new=test_set.loc[:,train_set.columns.to_list()[:-1]]




                        #------------------------------------------
                        #------------------------------------------

                        X = train_set_new.iloc[:,0:-2].values#the whole set
                        y = train_set_new.iloc[:,-2:-1].values.flatten() #the whole set
                        X_train = train_set.iloc[:, 0:-2].values
                        y_train = train_set.iloc[:, -2:-1].values.flatten()


                        X_test = test_set_new.iloc[:,0:-1].values
                        y_test = test_set_new.iloc[:,-1].values.flatten()

                        search = GridSearchCV(estimator=pipe, param_grid=hyper_space, n_jobs=n_jobs,
                                              scoring='f1_macro',
                                              cv=create_generator(train_val_train_index), refit=True)

                        search.fit(X, y)
                        hyperparameters_test_sub=search.best_estimator_
                        current_pipe=hyperparameters_test_sub
                        # -------------------------------------------------------
                        # retrain on train and val
                        current_pipe.fit(X_train, y_train)
                        y_test_pred = current_pipe.predict(X_test)
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
                        hyperparameters_test.append(hyperparameters_test_sub)
                    score = [f1_test, score_report_test, aucs_test, mcc_test, hyperparameters_test]
                    with open(save_name_score + '_kma_' + str(f_kma) + '_tree_' + str(f_phylotree) + '.pickle',
                              'wb') as f:  # overwrite mode
                        pickle.dump(score, f)


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
        hyper_space = {'cl__max_features': ["auto", "sqrt", "log2", None],'cl__n_estimators':[ 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
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
    # parser.add_argument('-path_sequence', '--path_sequence', default= '/net/projects/BIFO/patric_genome', type=str, required=False,
    #                     help='path of sequence,another option: \'/vol/projects/BIFO/patric_genome\'')
    # parser.add_argument('-path_large_temp', '--path_large_temp', default='/net/sgi/metagenomics/data/khu/benchmarking/phylo', type=str,
    #                     required=False, help='path for large temp files/folders, another option: \'/vol/projects/khu/amr/benchmarking/large_temp\'')
    parser.add_argument('-path_sequence', '--path_sequence', default='/vol/projects/BIFO/patric_genome', type=str,
                        required=False,
                        help='path of sequence,another option: \'/net/projects/BIFO/patric_genome\'')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                        help='kma based cv folders.')
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
    parser.add_argument('-f_tree', '--f_tree', dest='f_tree', action='store_true',
                        help='Kma cluster')
    parser.add_argument('-f_qsub', '--f_qsub', dest='f_qsub',
                        help='Prepare scriptd for qsub.', action='store_true', )
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'Pseudomonas aeruginosa\' \
                    \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
                    \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parsedArgs = parser.parse_args()
    extract_info(parsedArgs.path_sequence, parsedArgs.species, parsedArgs.f_all, parsedArgs.f_prepare_meta,
                 parsedArgs.f_tree, parsedArgs.cv, parsedArgs.level, parsedArgs.n_jobs, parsedArgs.f_ml,
                 parsedArgs.f_phylotree,parsedArgs.f_kma,parsedArgs.f_qsub)
