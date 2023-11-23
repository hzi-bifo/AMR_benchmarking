#!/usr/bin/python

#Modified based on Derya Aytan's work:#https://bitbucket.org/deaytan/neural_networks/src/master/Neural_networks.py

import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import sys
sys.path.insert(0, os.getcwd())
import time
import torch
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_curve, auc,classification_report,f1_score
from sklearn import preprocessing
from src.amr_utility import name_utility
import itertools
import statistics
from src.cv_folds import name2index
import json
from nn.hyperpara import hyper_range,training,OuterLoop_training,training_original,classifier_original,_classifier,_classifier2,_classifier3,Num2l

'''
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)




def multiAnti(species, antibiotics, level, xdata, ydata, p_names, cv, N_cv, f_scaler,f_fixed_threshold,
              f_no_early_stop,f_phylotree, f_kma,f_optimize_score,  save_name_weights,save_name_score):

    #data
    data_x = np.loadtxt(xdata, dtype="float")
    data_y =np.loadtxt(ydata)
    print('dataset shape',data_x.shape)

    ##prepare data stores for testing scores##
    pred_test = []  # probabilities are for the validation set
    # pred_test_binary = [] #binary based on selected
    thresholds_selected_test=[]# cv len, each is the selected threshold.
    weight_test=[]# cv len, each is the index of the selected model in inner CV,# only this weights are preseved in /log/temp/
    mcc_test = []  # MCC results for the test data
    f1_test = []
    score_report_test = []
    predictY_test=[] #predicted Y
    true_Y=[] #corresponding true Y
    aucs_test = []  # all AUC values for the test data
    sampleNames_test=[] #sample names
    hyperparameters_test=[]
    actual_epoc_test = []
    actual_epoc_test_std = []
    # -------------------------------------------------------------------------------------------------------------------
    # model varibales
    anti_number = data_y[0].size  # number of antibiotics
    nlabel = data_y[0].size  # number of neurons in the output layer
    D_input = len(data_x[0])  # number of neurons in the input layer
    # N_sample = len(data_x)  # number of input samples #should be equal to len(names)

    #CV folds
    if f_kma:
        folds_txt=name_utility.GETname_foldsSSMA(species,level,f_kma,f_phylotree)
        folders_sample_name = json.load(open(folds_txt, "rb"))
        folders_sample=name2index.Get_index(folders_sample_name,p_names)
    else:
        print('Error: Only KMA folds possible.')
        exit(1)
    hyper_space = hyper_range(anti_number, f_no_early_stop, antibiotics)

    # for out_cv in range(cv):
    for out_cv in N_cv:

        train_val_samples= folders_sample[:out_cv] + folders_sample[out_cv+1 :]#list
        Validation_mcc_thresholds = []  #  inner CV *11 thresholds value
        Validation_f1_thresholds = []  #  inner CV *11 thresholds value
        Validation_auc = []  # inner CV
        Validation_actul_epoc=[]

        test_samples = folders_sample[out_cv]#modified. khu.May4.2022
        x_test = data_x[test_samples]
        y_test = data_y[test_samples]
        print('x_test shape',x_test.shape)
        for innerCV in range(cv - 1):  # e.g. 1,2,3,4
            print('Starting outer: ', str(out_cv), 'out of ', N_cv, '; inner: ', str(innerCV), ' inner loop...')

            val_samples=train_val_samples[innerCV]
            train_samples=train_val_samples[:innerCV] + train_val_samples[innerCV+1 :]#only works for list, not np
            train_samples=list(itertools.chain.from_iterable(train_samples))
            x_train, x_val = data_x[train_samples], data_x[val_samples]#only np
            y_train, y_val = data_y[train_samples], data_y[val_samples]
            print('sample length and feature length for inner CV(train & val):',len(x_train), len(x_train[0]),len(x_val), len(x_val[0]))


            # normalize the data
            if f_scaler==True:
                scaler = preprocessing.StandardScaler().fit(x_train)
                x_train = scaler.transform(x_train)
                # scale the val data based on the training data
                x_val = scaler.transform(x_val)


            x_train = torch.from_numpy(x_train).float()
            y_train = torch.from_numpy(y_train).float()
            x_val = torch.from_numpy(x_val).float()
            y_val = torch.from_numpy(y_val).float()
            pred_val_inner = []  # predicted results on validation set.
            ###################
            # 1. train the model #
            ###################

            Validation_mcc_thresholds_split = []  # inner CV *11 thresholds value
            Validation_f1_thresholds_split = []  # inner CV *11 thresholds value
            Validation_auc_split = []  # inner CV
            Validation_actul_epoc_split = []


            for grid_iteration in np.arange(len(list(ParameterGrid(hyper_space)))):

                # --------------------------------------------------------------------
                epochs=list(ParameterGrid(hyper_space))[grid_iteration]['epochs']
                n_hidden=list(ParameterGrid(hyper_space))[grid_iteration]['n_hidden']
                learning=list(ParameterGrid(hyper_space))[grid_iteration]['lr']
                n_cl=list(ParameterGrid(hyper_space))[grid_iteration]['classifier']
                dropout=list(ParameterGrid(hyper_space))[grid_iteration]['dropout']
                patience=list(ParameterGrid(hyper_space))[grid_iteration]['patience']
                # generate a NN model
                if n_cl==1:
                    classifier = _classifier(nlabel, D_input, n_hidden,dropout)#reload, after testing(there is fine tune traning!)
                elif n_cl==2:
                    classifier = _classifier2(nlabel, D_input, n_hidden,dropout)
                elif n_cl==3:
                    classifier = _classifier3(nlabel, D_input, n_hidden,dropout)
                elif n_cl==4:
                    classifier =classifier_original(nlabel, D_input, n_hidden)
                print(list(ParameterGrid(hyper_space))[grid_iteration])
                classifier.to(device)
                # generate the optimizer - Stochastic Gradient Descent
                optimizer = optim.SGD(classifier.parameters(), lr=learning)
                optimizer.zero_grad()  # Clears existing gradients from previous epoch
                #loop:
                print(species, antibiotics, level, out_cv, innerCV)
                start = time.time()
                if f_no_early_stop==True:
                    classifier,actual_epoc = training_original(classifier,epochs,optimizer,x_train,y_train,anti_number)
                else:
                    classifier,actual_epoc =training(classifier, epochs, optimizer, x_train, y_train,x_val,y_val, anti_number,patience)
                Validation_actul_epoc_split.append(actual_epoc)
                end = time.time()
                print('Time used: ',end - start)

                ######################
                # 2. validate the model #
                ######################

                classifier.eval()  # eval mode
                pred_val_sub = []
                x_val=x_val.to(device)
                for v, v_sample in enumerate(x_val):

                    # val = Variable(torch.FloatTensor(v_sample)).view(1, -1)
                    val = Variable(v_sample).view(1, -1)
                    output_test = classifier(val)
                    out = output_test
                    temp = []
                    for h in out[0]:
                        temp.append(float(h))
                    pred_val_sub.append(temp)

                pred_val_sub = np.array(pred_val_sub)# for this innerCV at this out_cv

                #--------------------------------------------------
                #auc score, threshould operation is contained in itself definition.

                if f_optimize_score=='auc':
                    # y_val=np.array(y_val)
                    aucs_val_sub_anti=[]
                    for i in range(anti_number):
                        comp = []
                        if anti_number == 1:
                            fpr, tpr, _ = roc_curve(y_val, pred_val_sub, pos_label=1)
                            # tprs.append(interp(mean_fpr, fpr, tpr))
                            # tprs[-1][0] = 0.0
                            roc_auc = auc(fpr, tpr)
                            Validation_auc_split.append(roc_auc)
                        else:  # multi-out

                            for t in range(len(y_val)):
                                if -1 != y_val[t][i]:
                                    comp.append(t)

                            y_val_anti=y_val[comp]
                            pred_val_sub_anti=pred_val_sub[comp]
                            fpr, tpr, _ = roc_curve(y_val_anti[:, i], pred_val_sub_anti[:, i], pos_label=1)
                            roc_auc = auc(fpr, tpr)
                            aucs_val_sub_anti.append(roc_auc)
                    if anti_number > 1:# multi-out, scores based on mean of all the involved antibotics
                        aucs_val_sub=statistics.mean(aucs_val_sub_anti)#dimension: n_anti to 1
                        Validation_auc_split.append(aucs_val_sub)# D: n_innerCV


                #====================================================
                elif f_optimize_score=='f1_macro':
                    # Calculate macro f1. for thresholds from 0 to 1.
                    mcc_sub = []
                    f1_sub = []

                    for threshold in np.arange(0, 1.1, 0.1):
                        # predictions for the test data
                        #turn probabilty to binary
                        threshold_matrix = np.full(pred_val_sub.shape, threshold)
                        y_val_pred = (pred_val_sub > threshold_matrix)
                        y_val_pred = 1*y_val_pred
                        mcc_sub_anti = []
                        f1_sub_anti = []
                        for i in range(anti_number):

                            comp = []  # becasue in the multi-species model, some species,anti combination are missing data
                            # so those won't be counted when evaluating scores.

                            if anti_number == 1:
                                mcc = matthews_corrcoef(y_val, y_val_pred)
                                # report = classification_report(y_val, y_val_pred, labels=[0, 1], output_dict=True)
                                f1 = f1_score(y_val, y_val_pred, average='macro')
                                mcc_sub.append(mcc)
                                f1_sub.append(f1)
                            else:  # multi-out
                                for t in range(len(y_val)):
                                    if -1 != y_val[t][i]:
                                        comp.append(t)
                                y_val_multi_sub = y_val[comp]
                                y_val_pred_multi_sub = y_val_pred[comp]

                                if  comp==[]:#exceptions: for MT multi-anti model
                                    pass
                                else:
                                    mcc = matthews_corrcoef(y_val_multi_sub[:, i], y_val_pred_multi_sub[:, i])
                                    f1 = f1_score(y_val_multi_sub[:, i], y_val_pred_multi_sub[:, i], average='macro')
                                    mcc_sub_anti.append(mcc)
                                    f1_sub_anti.append(f1)
                        if anti_number > 1:  # multi-out, scores based on mean of all the involved antibotics
                            mcc_sub.append(statistics.mean(mcc_sub_anti))  # mcc_sub_anti dimension: n_anti
                            f1_sub.append(statistics.mean(f1_sub_anti))
                    Validation_mcc_thresholds_split.append(mcc_sub)
                    Validation_f1_thresholds_split.append(f1_sub)

            Validation_mcc_thresholds.append(Validation_mcc_thresholds_split)  # inner CV * hyperpara_combination
            Validation_f1_thresholds.append(Validation_f1_thresholds_split)  # inner CV * hyperpara_combination
            Validation_auc.append(Validation_auc_split)  # inner CV * hyperpara_combination
            Validation_actul_epoc.append(Validation_actul_epoc_split) # inner CV * hyperpara_combination

        # finish evaluation in the inner loop.
        if f_fixed_threshold==True and f_optimize_score=='f1_macro':#finished.May 30. 7am
            thresholds_selected=0.5
            Validation_f1_thresholds = np.array(Validation_f1_thresholds)
            #exceptions: for MT multi-anti model
            Validation_f1_thresholds = Validation_f1_thresholds.mean(axis=0)
            Validation_actul_epoc = np.array(Validation_actul_epoc)
            Validation_actul_epoc_std = Validation_actul_epoc.std(axis=0)
            Validation_actul_epoc = Validation_actul_epoc.mean(axis=0)

            #select the inner loop index with the highest f1 score in the column w.r.t. 0.5
            aim_column=np.where(np.arange(0, 1.1, 0.1) == 0.5)
            aim_f1=Validation_f1_thresholds[:,aim_column]
            weights_selected=np.argmax(aim_f1)
            ind=np.unravel_index(np.argmax(aim_f1, axis=None), aim_f1.shape)
            hyperparameters_test.append(list(ParameterGrid(hyper_space))[ind[0]])
            actual_epoc_test.append(Validation_actul_epoc[ind[0]])
            actual_epoc_test_std.append(Validation_actul_epoc_std[ind[0]])

        elif f_fixed_threshold==False and f_optimize_score=='f1_macro':#finished.May 30. 7am
            # select the inner loop index,and threshold with the highest f1 score in the matrix
            Validation_f1_thresholds=np.array(Validation_f1_thresholds)
            Validation_f1_thresholds=Validation_f1_thresholds.mean(axis=0)
            Validation_actul_epoc = np.array(Validation_actul_epoc)

            Validation_actul_epoc_std = Validation_actul_epoc.std(axis=0)
            Validation_actul_epoc = Validation_actul_epoc.mean(axis=0)
            ind = np.unravel_index(np.argmax(Validation_f1_thresholds, axis=None), Validation_f1_thresholds.shape)
            thresholds_selected=np.arange(0, 1.1, 0.1)[ind[1]]
            weights_selected=ind[0]#the order of innerCV# bug ? seems no 13May.
            hyperparameters_test.append(list(ParameterGrid(hyper_space))[ind[0]])
            actual_epoc_test.append(Validation_actul_epoc[ind[0]])
            actual_epoc_test_std.append(Validation_actul_epoc_std[ind[0]])
        elif f_optimize_score=='auc':#finished.May 30. 7am

            thresholds_selected = 0.5
            Validation_auc = np.array(Validation_auc)
            Validation_auc=Validation_auc.mean(axis=0)
            Validation_actul_epoc= np.array(Validation_actul_epoc)
            Validation_actul_epoc_std = Validation_actul_epoc.std(axis=0)
            Validation_actul_epoc=Validation_actul_epoc.mean(axis=0)

            weights_selected = np.argmax(Validation_auc)
            ind = np.unravel_index(np.argmax(Validation_auc, axis=None), Validation_auc.shape)
            hyperparameters_test.append(list(ParameterGrid(hyper_space))[ind[0]])
            actual_epoc_test.append(Validation_actul_epoc[ind[0]])# actually it's mean epoch for that hyperpara.
            actual_epoc_test_std.append(Validation_actul_epoc_std[ind[0]])

        print('hyper_space selected: ',list(ParameterGrid(hyper_space))[ind[0]])
        print('weights_selected', weights_selected)
        print('thresholds_selected', thresholds_selected)


        weight_test.append(weights_selected)
        thresholds_selected_test.append(thresholds_selected)
        #finish innner loop

        #=======================================================
        #3. Re-train on both train and val data with the selected hyper-para
        print('Outer loop, testing.')
        train_val_samples = list(itertools.chain.from_iterable(train_val_samples))
        x_train_val= data_x[train_val_samples] # only np
        y_train_val= data_y[train_val_samples]


        if f_scaler==True:
            scaler = preprocessing.StandardScaler().fit(x_train_val)
            x_train_val = scaler.transform(x_train_val)
            # scale the val data based on the training data
            x_test = scaler.transform(x_test)


        x_test = torch.from_numpy(x_test).float()
        y_test = torch.from_numpy(y_test).float()
        x_test = x_test.to(device)
        x_train_val = torch.from_numpy(x_train_val).float()
        y_train_val = torch.from_numpy(y_train_val).float()

        #-

        n_hidden = hyperparameters_test[-1]['n_hidden']
        learning = hyperparameters_test[-1]['lr']
        n_cl = hyperparameters_test[-1]['classifier']
        dropout=hyperparameters_test[-1]['dropout']
        # generate a NN model
        if n_cl == 1:
            classifier = _classifier(nlabel, D_input, n_hidden,dropout)  # reload, after testing(there is fine tune traning!)
        elif n_cl == 2:
            classifier = _classifier2(nlabel, D_input, n_hidden,dropout)
        elif n_cl == 3:
            classifier = _classifier3(nlabel, D_input, n_hidden,dropout)

        classifier.to(device)
        # generate the optimizer - Stochastic Gradient Descent
        optimizer = optim.SGD(classifier.parameters(), lr=learning)
        optimizer.zero_grad()  # Clears existing gradients from previous epoch
        classifier.train()

        print('actual_epoc_test[out_cv]',actual_epoc_test[-1])
        classifier = OuterLoop_training(classifier, int(actual_epoc_test[-1]), optimizer, x_train_val, y_train_val, anti_number)
        torch.save(classifier.state_dict(), save_name_weights+'_weights_cv'+str(out_cv))

        classifier.eval()

        pred_test_sub = []
        for a, a_sample in enumerate(x_test):
            tested = Variable(a_sample).view(1, -1)
            output_test = classifier(tested)
            out = output_test
            temp = []
            for h in out[0]:
                temp.append(float(h))
            pred_test_sub.append(temp)
        pred_test_sub = np.array(pred_test_sub)
        pred_test.append(pred_test_sub)#len= y_test


        #turn probability to binary.
        threshold_matrix=np.full(pred_test_sub.shape, thresholds_selected)
        pred_test_binary_sub = (pred_test_sub > threshold_matrix)
        pred_test_binary_sub = 1 * pred_test_binary_sub

        y_test = np.array(y_test)
        f1_test_anti=[]# multi-out
        score_report_test_anti=[]# multi-out
        aucs_test_anti=[]# multi-out
        mcc_test_anti=[]# multi-out

        for i in range(anti_number):
            comp = []
            if anti_number == 1:
                mcc = matthews_corrcoef(y_test, pred_test_binary_sub)
                f1 = f1_score(y_test, pred_test_binary_sub, average='macro')
                report = classification_report(y_test, pred_test_binary_sub, labels=[0, 1],output_dict=True)
                fpr, tpr, _ = roc_curve(y_test, pred_test_binary_sub, pos_label=1)
                roc_auc = auc(fpr, tpr)

            else:  # multi-out

                for t in range(len(y_test)):
                    if -1 != y_test[t][i]:
                        comp.append(t)
                y_test_anti=y_test[comp]
                pred_test_binary_sub_anti=pred_test_binary_sub[comp]
                pred_test_sub_anti=pred_test_sub[comp]#June 21st, auc bugs
                if comp==[]:#MT exception.
                    f1=np.nan
                    report=np.nan
                    roc_auc=np.nan
                    mcc=np.nan

                else:
                    mcc = matthews_corrcoef(y_test_anti[:, i], pred_test_binary_sub_anti[:, i])
                    f1 = f1_score(y_test_anti[:, i], pred_test_binary_sub_anti[:, i], average='macro')

                    fpr, tpr, _ = roc_curve(y_test_anti[:, i], pred_test_sub_anti[:, i], pos_label=1)#June 21st, auc bugs
                    roc_auc = auc(fpr, tpr)
                    report = classification_report(y_test_anti[:, i], pred_test_binary_sub_anti[:, i], labels=[0, 1], output_dict=True)

                f1_test_anti.append(f1)
                score_report_test_anti.append(report)
                aucs_test_anti.append(roc_auc)
                mcc_test_anti.append(mcc)


        #summarise each outer loop's results.
        if anti_number>1:
            f1_test.append(f1_test_anti)
            score_report_test.append(score_report_test_anti)
            aucs_test.append(aucs_test_anti)  # multi-out
            mcc_test.append(mcc_test_anti)
        else:
            f1_test.append(f1)
            score_report_test.append(report)
            aucs_test.append(roc_auc)  # single-out
            mcc_test.append(mcc)


        predictY_test.append(pred_test_binary_sub.tolist())
        true_Y.append(y_test.tolist())
        sampleNames_test.append(folders_sample_name[out_cv])
        actual_epoc_test=Num2l(actual_epoc_test)
        actual_epoc_test_std=Num2l(actual_epoc_test_std)

        score = {'f1_test':f1_test,'score_report_test':score_report_test,'aucs_test':aucs_test,'mcc_test':mcc_test,
             'predictY_test':predictY_test,'ture_Y':true_Y,'samples':sampleNames_test,'thresholds_selected_test':thresholds_selected_test,
             'hyperparameters_test':hyperparameters_test,'actual_epoc_test':actual_epoc_test,'actual_epoc_test_std':actual_epoc_test_std}


        with open(save_name_score+str(out_cv)+ '.json', 'w') as f:
            json.dump(score, f)
    torch.cuda.empty_cache()
        # return score


def multiAnti_score( cv,save_name_score ):

    thresholds_selected_test=[]
    f1_test=[]
    mcc_test=[]
    score_report_test=[]
    aucs_test=[]
    hyperparameters_test=[]
    actual_epoc_test=[]
    actual_epoc_test_std=[]
    predictY_test=[]
    true_Y=[]
    sampleNames_test=[]
    for out_cv in range(cv):


        score=json.load(open(save_name_score+str(out_cv)+ '.json', "rb"))

        f1=score['f1_test']
        mcc=score['mcc_test']
        thresholds_selected=score['thresholds_selected_test']
        score_report=score['score_report_test']
        aucs=score['aucs_test']
        hyperparameters=score['hyperparameters_test']
        actual_epoc=score['actual_epoc_test']
        actual_epoc_std=score['actual_epoc_test_std']
        predictY=score['predictY_test']
        ture_y=score['ture_Y']
        sampleNames=score['samples']

        thresholds_selected_test.append(thresholds_selected[-1])
        f1_test.append(f1[-1])
        mcc_test.append(mcc[-1])
        score_report_test.append(score_report[-1])
        aucs_test.append(aucs[-1])
        hyperparameters_test.append(hyperparameters[-1])
        actual_epoc_test.append(actual_epoc[-1])
        actual_epoc_test_std.append(actual_epoc_std[-1])
        predictY_test.append(predictY)
        true_Y.append(ture_y)
        sampleNames_test.append(sampleNames)

    score = {'f1_test':f1_test,'score_report_test':score_report_test,'aucs_test':aucs_test,'mcc_test':mcc_test,
                 'predictY_test':predictY_test,'ture_Y':true_Y,'samples':sampleNames_test,'thresholds_selected_test':thresholds_selected_test,
             'hyperparameters_test':hyperparameters_test,'actual_epoc_test':actual_epoc_test,'actual_epoc_test_std':actual_epoc_test_std}


    with open(save_name_score + '.json', 'w') as f:
        json.dump(score, f)
