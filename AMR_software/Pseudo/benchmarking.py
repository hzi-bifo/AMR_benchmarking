#!/usr/bin/python
"""
Created on Jan 6 2024
@author: Kaixin Hu

In this tutorial, we provide a step-by-step fundamental guide for using our workflow to evaluate a model currently out of
our benchmarking choice or that could be released in the feature by any third party. Two types of machine-learning (ML)-based
AMR phenotyping methods are provided as examples.
- The first is a classic ML model 1) that can use the scikit-learn module for training,
    2) in which the feature matrix could be built without phenotype information,
    3) encompasses several classifiers. Among our benchmarked methods, Seq2Geno2Pheno falls into this category. Those methods could be evaluated via nested cross-validations.
- The second ML model does not necessitate hyperparameter cross-validations and encompasses several classifiers,
    thus is evaluated via an iterative evaluation approach.
"""
import os
import numpy as np
from src.amr_utility import name_utility, file_utility, load_data
import itertools
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import auc,roc_curve,matthews_corrcoef,confusion_matrix,f1_score,precision_recall_fscore_support,classification_report
import pickle,json
from src.cv_folds import name2index

def extract_infor(level,f_all,s,temp_path,software_name):
    '''
    Load dataset information, and make directory for generating feature matrix.
    ---------------------------------
    software_name: software name
    temp_path: path to temporary file, like feature matrix
    s: species name
    f_all: if set to true, all the 11 species will be evaluated
    '''
    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]
    if f_all == False:
        data = data.loc[s, :]
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    file_utility.make_dir(temp_path+'log/software/'+software_name+'/software_output/cano6mer/temp') #for kmers
    return df_species,antibiotics

def create_generator(nFolds):
    for idx in range(len(nFolds)):
        test =nFolds[idx]
        train = list(itertools.chain(*[fold for idy, fold in enumerate(nFolds) if idy != idx]))
        yield train, test

class AMR_software():
    '''
    An example of evaluating a ML-based software that could use the scikit-learn module for training classifiers,
    and the feature matrix could be built without phenotype information. Among our benchmarked methods, Seq2Geno2Pheno falls into this category.
    '''
    def __init__(self,software_name,path_sequence,temp_path,s,f_all, f_phylotree, f_kma,cv,n_jobs):
        '''
        software_name: software name
        path_sequence: genome sequence in fna format
        temp_path: path to temporary file, like feature matrix
        s: species name
        f_all: if set to true, all the 11 species will be evaluated
        f_phylotree: flag for phylogeny-aware evaluation
        f_kma: flag for homology-aware evaluation
        cv: number of CV folds
        n_jobs: cpu cores available
        '''

        self.software_name=software_name
        self.path_sequence=path_sequence
        self.temp_path=temp_path
        self.s=s
        self.f_all =f_all
        self.f_phylotree=f_phylotree
        self.f_kma=f_kma
        self.level="loose"
        self.cv=cv
        self.n_jobs=n_jobs


    def prepare_feature(self):
        df_species,antibiotics=extract_infor(self.level,self.f_all,self.s,self.temp_path,self.software_name)
        for species, antibiotics in zip(df_species, antibiotics):
            antibiotics, _, _ =  load_data.extract_info(species, False, self.level)
            pd.DataFrame(antibiotics).to_csv('<path_to_feature>', sep="\t", index=False, header=False)
            for anti in antibiotics:
                name,_,_ = name_utility.GETname_model(self.software_name,self.level, species, anti,'',self.temp_path)
                phenotype_list = pd.read_csv(name, index_col=0, dtype={'genome_id': object}, sep="\t")
                phenotype_list['path']=str(self.path_sequence) +'/'+ phenotype_list['genome_id'].astype(str)+'.fna'

                ################################################################
                '''
                phenotype_list is pandas matrix with each row representing a genome. E.G.
                    genome_id	resistant_phenotype	path 
                0	470.2428	1                   <path_to_sequence>/470.2428.fna
                1	480119.5	0                   <path_to_sequence>/480119.5.fna
                2	470.2433	1                   <path_to_sequence>/470.2433.fna
                3	470.2159	1                   <path_to_sequence>/470.2159.fna
                4	470.2166	1                   <path_to_sequence>/470.2166.fna
                ...
                ...
                
                ![Please here add your codes for building features based on above information for your model]
                '''
                ## save the feature matrix to a folder under temp_path
                data_feature.to_csv('<path_to_feature>', sep="\t")
                ################################################################




    def nested_cv(self): ### nested CV
        df_species,antibiotics=extract_infor(self.level,self.f_all,self.s,self.temp_path,self.software_name)
        for species, antibiotics in zip(df_species, antibiotics):

            ## antibiotics is the python list of benchmarked antibiotics for that species
            ## ID is the python list of PATRIC ID, e.g. [1352.10013,1352.10014, ..., 1354.10,1366.10]
            ## Y is the python list of phenotype for each sample in the same order as ID list
            antibiotics, ID, Y = load_data.extract_info(species, False, self.level)
            i_anti = 0
            for anti in antibiotics: ## evaluate each species-antibiotic dataset sequentially
                id_all = ID[i_anti]
                y_all = Y[i_anti]
                i_anti+=1

                '''                
                ! [Please specifiy the model's classifiers and feature matrix location here. ] 
                For example:
                CLASSIFIERS=['svm','lr', 'rf','lsvm']
                data_feature=pd.read_csv('<path_to_feature>', index_col=0,sep="\t")                                
                '''

                X_all = pd.concat([X_all, data_feature.reindex(X_all.index)], axis=1) ## load the feature matrix
                id_all = np.array(id_all)
                y_all = np.array(y_all)

                ## load in folds for nested CV
                p_names = name_utility.GETname_meta(species,anti,self.level)
                folds_txt=name_utility.GETname_folds(species,anti,self.level,self.f_kma,self.f_phylotree)
                folders_sample = json.load(open(folds_txt, "rb"))
                folders_index=name2index.Get_index(folders_sample,p_names) # CV folds
                for chosen_cl in CLASSIFIERS: # evaluate each classifier sequentially
                    _, _,save_name_score=name_utility.GETname_model(self.software_name, self.level,species, anti,chosen_cl,self.temp_path)
                    file_utility.make_dir(os.path.dirname(save_name_score))


                    ### metrics of 10 folds
                    mcc_test = []
                    f1_test = [] ## F1-macro
                    ## score_report could be used to extract metrics like precision-positive,
                    ## recall-positive, F1-positive, precision-negative, recall-negative, F1-negative, and accuracy
                    score_report_test = []
                    aucs_test = []

                    ### other outputs from nested CV
                    estimator_test=[] ##scikit-learn
                    hyperparameters_test = [] ## hyperparameters selected from inner loop CV for training in each of the 10 outer loop iteration
                    score_InnerLoop = []  ## the best metric scores of inner loops for each of the 10 outer loop iteration
                    index_InnerLoop=[] ## the order of the best hyperparameters in grid search
                    cv_results_InnerLoop=[] ## cv_results_ attributes of scikit-learn  sklearn.model_selection.GridSearchCV

                    ### testing results
                    sampleNames_test=[] ## sample PATRIC ID for a species-antibiotic combination/dataset
                    predictY_test=[] ## predicted AMR phenotype for each sample, ordered the same as sampleNames_test
                    true_Y=[] ## ground truth AMR phenotype for each sample, ordered the same as sampleNames_test



                    for out_cv in range(self.cv): ## outer loop of nested CV
                        print(species,anti,'. Starting outer: ', str(out_cv), '; chosen_cl: ', chosen_cl)

                        test_samples_index = folders_index[out_cv] ## a list of index of the test fold
                        id_test = id_all[test_samples_index]  ## sample name list of the test fold
                        y_test = y_all[test_samples_index] ## ground truth phenotype
                        train_val_index =folders_index[:out_cv] +folders_index[out_cv + 1:] ## 9 folds involved in inner loop for CV
                        id_val_train = id_all[list(itertools.chain.from_iterable(train_val_index))]  ## sample name list of the 9 folds
                        y_val_train = y_all[list(itertools.chain.from_iterable(train_val_index))] ## phenotype of the 9 folds
                        X_val_train=X_all.loc[id_val_train,:] ## feature matrix of samples from the 9 folds
                        X_test=X_all.loc[id_test,:] ##  feature matrix of samples of the test fold

                        '''
                        ! [Please specify the model's classifiers' hyperparameter selection range here. ] 
                        For example:
                        cl = RandomForestClassifier(random_state=1)
                        hyper_space hyper_space = [
                              {
                                "cl__n_estimators": [100, 200, 500, 1000],
                                "cl__criterion": ["entropy", "gini"],
                                "cl__max_features": ["auto"],
                                "cl__min_samples_split": [2,5,10],
                                "cl__min_samples_leaf": [1],
                                "cl__class_weight": ["balanced", None]
                              }
                            ]                         
                        '''
                        ###############################################################################################
                        ## typical procedures for nested CV inner loop hyperparameter selection.
                        ## note: this part can be vaired based on specific techniques, e.g. for Kover, this can be replaced with bound selection,
                        ## for neural networks model, this should be replaced with a hyperparameter optimization procedure accompanied by early stopping
                        ## (see function training in AMR_software/AytanAktug/nn/hyperpara.py)
                        ###############################################################################################
                        ### Grid search for hyperparameter selection
                        pipe = Pipeline(steps=[('cl', cl)])
                        search = GridSearchCV(estimator=pipe, param_grid=hyper_space, n_jobs=self.n_jobs,
                                                  scoring='f1_macro',
                                                  cv=create_generator(train_val_index), refit=True)


                        search.fit(X_all, y_all)
                        hyperparameters_test_sub=search.best_estimator_
                        scores_best=search.best_score_
                        index_best=search.best_index_
                        cv_results=search.cv_results_
                        current_pipe=hyperparameters_test_sub

                        # retrain on train and validation data using the optimal hyperparameters
                        current_pipe.fit(X_val_train, y_val_train)
                        y_test_pred = current_pipe.predict(X_test)
                        ###############################################################################################


                        # calculate metric scores
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
                        index_InnerLoop.append(index_best)
                        cv_results_InnerLoop.append(cv_results)
                        predictY_test.append( y_test_pred.tolist())
                        true_Y.append(y_test.tolist())
                        sampleNames_test.append(folders_sample[out_cv])
                        estimator_test.append(current_pipe)

                    ### Save metric scores and other model evaluation information
                    score ={'f1_test':f1_test,'score_report_test':score_report_test,'aucs_test':aucs_test,'mcc_test':mcc_test,
                         'predictY_test':predictY_test,'ture_Y':true_Y,'samples':sampleNames_test}
                    score2= {'hyperparameters_test':hyperparameters_test,'estimator_test':estimator_test,
                             'score_InnerLoop':score_InnerLoop,'index_InnerLoop':index_InnerLoop,'cv_results_InnerLoop':cv_results_InnerLoop}
                    with open(save_name_score + '_KMA_' + str(self.f_kma) + '_Tree_' + str(self.f_phylotree) + '.json',
                              'w') as f:  # overwrite mode
                        json.dump(score, f)
                    with open(save_name_score + '_kma_' + str(self.f_kma) + '_tree_' + str(self.f_phylotree) + '_model.pickle',
                              'wb') as f:  # overwrite mode
                        pickle.dump(score2, f)




    def prepare_feature_val(self):
        '''
        This feature-preparing procedure is for ML models
        1) possessing an inherent hyperparameter optimization process, which means nested CV is not needed;
        but 2) encompassing several classifiers, which means we should still design a CV on the 9 training folds at each iteration
        to select the optimal classifier
        '''

        df_species,antibiotics=extract_infor(self.level,self.f_all,self.s,self.temp_path,self.software_name)
        for species, antibiotics in zip(df_species, antibiotics):
            ### load antibiotics for this species; load sample PATRIC IDs for this species-antibiotic combination
            antibiotics, ID, _ =  load_data.extract_info(species, False, self.level)
            i_anti = 0

            for anti in antibiotics:
                id_all = ID[i_anti]  # sample names of all the 10 folds
                i_anti += 1
                id_all = np.array(id_all)

                ### 1. load CV folds
                p_names = name_utility.GETname_meta(species,anti,self.level)
                folds_txt=name_utility.GETname_folds(species,anti,self.level,self.f_kma,self.f_phylotree)
                folders_sample = json.load(open(folds_txt, "rb"))
                folders_index=name2index.Get_index(folders_sample,p_names) # CV folds

                for out_cv in range(self.cv):
                    test_samples_index = folders_index[out_cv]
                    train_val_train_index = folders_index[:out_cv] + folders_index[out_cv + 1:]

                    for inner_cv in range(self.cv-1):
                        val_index=train_val_train_index[inner_cv]
                        train_index=train_val_train_index[:inner_cv] + train_val_train_index[inner_cv+1 :]
                        id_train = id_all[list(itertools.chain.from_iterable(train_index))]  ## sample name list of CV training set
                        id_val = id_all[val_index]  ## sample name list of CV test set

                        #################################################################################################
                        ## genome sequence location. Please use this information to load in sequence for building feature matrix.
                        sequence_train=[str(self.path_sequence) +'/'+ genome_id.astype(str)+'.fna' for genome_id in id_train]
                        sequence_testn=[str(self.path_sequence) +'/'+ genome_id.astype(str)+'.fna' for genome_id in id_val]
                        '''
                        2. prepare meta/feature files for this iteration of training samples
                         only retain those in the training and validation CV folds    
                                            
                         ![Please here add your codes for building features]                    
                        
                         '''
                        ## save the feature matrix to a folder under temp_path
                        data_feature.to_csv('<path_to_feature>', sep="\t")
                        #################################################################################################




