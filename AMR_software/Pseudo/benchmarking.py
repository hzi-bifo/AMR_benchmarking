#!/usr/bin/python
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

class main():
    def __init__(self,software_name,path_sequence,temp_path,s,f_all, f_phylotree, f_kma,cv,n_jobs):
        '''
        software_name: software name
        path_sequence: genome sequence in fna format
        temp_path: path to temporary file, like feature matrix
        s: species name
        f_all: if set to true, all the 11 species will be evaluated
        f_phylotree: flag for phylogeny-aware evaluation
        f_kma: flag for homology-aware evaluation
        cv: numner of CV folds
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
                data_feature.to_csv('<path_to_feature>', sep="\t")
                ################################################################




    def ml(self): ### nested CV
        df_species,antibiotics=extract_infor(self.level,self.f_all,self.s,self.temp_path,self.software_name)
        for species, antibiotics in zip(df_species, antibiotics):
            antibiotics, ID, Y = load_data.extract_info(species, False, self.level)
            i_anti = 0
            for anti in antibiotics:

                id_all = ID[i_anti]  # sample name list, e.g. [1352.10013,1352.10014,1354.10,1366.10]
                y_all = Y[i_anti]
                i_anti+=1

                '''                
                ! [Please specifiy the model's classifiers and features here. ] 
                For example:
                CLASSIFIERS=['svm','lr', 'rf','lsvm']
                data_feature=pd.read_csv('<path_to_feature>', index_col=0,sep="\t")                
                
                '''

                X_all = pd.concat([X_all, data_feature.reindex(X_all.index)], axis=1)

                id_all = np.array(id_all)
                y_all = np.array(y_all)

                ## load in folds for nested CV
                p_names = name_utility.GETname_meta(species,anti,self.level)
                folds_txt=name_utility.GETname_folds(species,anti,self.level,self.f_kma,self.f_phylotree)
                folders_sample = json.load(open(folds_txt, "rb"))
                folders_index=name2index.Get_index(folders_sample,p_names) # CV folds
                for chosen_cl in CLASSIFIERS:
                    _, _,save_name_score=name_utility.GETname_model(self.software_name, self.level,species, anti,chosen_cl,self.temp_path)
                    file_utility.make_dir(os.path.dirname(save_name_score))



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



                    for out_cv in range(self.cv): ## outer loop of nested CV
                        print(species,anti,'. Starting outer: ', str(out_cv), '; chosen_cl: ', chosen_cl)

                        test_samples_index = folders_index[out_cv]# a list of index
                        id_test = id_all[test_samples_index]#sample name list
                        y_test = y_all[test_samples_index]
                        train_val_train_index =folders_index[:out_cv] +folders_index[out_cv + 1:]
                        id_val_train = id_all[list(itertools.chain.from_iterable(train_val_train_index))]  # sample name list
                        y_val_train = y_all[list(itertools.chain.from_iterable(train_val_train_index))]
                        X_val_train=X_all.loc[id_val_train,:]
                        X_test=X_all.loc[id_test,:]

                        '''
                        ! [Please specifiy the model's classifiers' hyperparameter selection range here. ] 
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
                        pipe = Pipeline(steps=[('cl', cl)])
                        search = GridSearchCV(estimator=pipe, param_grid=hyper_space, n_jobs=self.n_jobs,
                                                  scoring='f1_macro',
                                                  cv=create_generator(train_val_train_index), refit=True)


                        search.fit(X_all, y_all)
                        hyperparameters_test_sub=search.best_estimator_
                        scores_best=search.best_score_
                        index_best=search.best_index_
                        cv_results=search.cv_results_
                        current_pipe=hyperparameters_test_sub
                        ###############################################################################################



                        # retrain on train and val
                        current_pipe.fit(X_val_train, y_val_train)
                        y_test_pred = current_pipe.predict(X_test)

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




