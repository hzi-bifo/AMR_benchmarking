#!/usr/bin/python

#nested CV, modified based on Derya Aytan's work.
#https://bitbucket.org/deaytan/neural_networks/src/master/Neural_networks.py

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
# sys.path.append('../')
sys.path.insert(0, os.getcwd())
import time
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import sys
import warnings
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_curve, auc,classification_report,f1_score
from scipy import interp
from sklearn import preprocessing
from src.amr_utility import name_utility
import itertools
import statistics
from AMR_software.AytanAktug.nn.pytorchtools import EarlyStopping
from src.cv_folds import name2index
import json


'''
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)



def Num2l(list_of_array):
    new=[]
    for each in list_of_array:
        new.append(each.tolist())
    return new

class _classifier3(nn.Module):
    def __init__(self, nlabel,D_in,H,dropout):
        super(_classifier3, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H, nlabel),
        )

    def forward(self, input):
        out=self.main(input)
        out=nn.Sigmoid()(out)
        return out
class _classifier2(nn.Module):
    def __init__(self, nlabel,D_in,H,dropout):
        super(_classifier2, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H, nlabel),
        )

    def forward(self, input):
        out=self.main(input)
        out=nn.Sigmoid()(out)
        return out
class _classifier(nn.Module):
    def __init__(self, nlabel,D_in,H,dropout):
        super(_classifier, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H, nlabel),
        )
    def forward(self, input):
        out=self.main(input)
        out=nn.Sigmoid()(out)
        return out

class classifier_original(nn.Module):
    def __init__(self, nlabel,D_in,H):
        super(classifier_original, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, nlabel),
        )
    def forward(self, input):
        out=self.main(input)
        out=nn.Sigmoid()(out)
        return out

def training_original(classifier,epochs,optimizer,x_train,y_train,anti_number):
    # Derya Aytan's original training procedure.
    for epoc in range(epochs):
        x_train=x_train.to(device)
        y_train = y_train.to(device)
        x_train_new = torch.utils.data.TensorDataset(x_train)
        y_train_new = torch.utils.data.TensorDataset(y_train)

        all_data = list(zip(x_train_new, y_train_new))

        # the model is trained for 100 batches
        data_loader = torch.utils.data.DataLoader(
            all_data, batch_size=100, drop_last=False)

        losses = []  # save the error for each iteration
        for i, (sample_x, sample_y) in enumerate(data_loader):
            optimizer.zero_grad()  # zero gradients #previous gradients do not keep accumulating
            inputv = sample_x[0]
            inputv=inputv.to(device)
            inputv = Variable(inputv).view(len(inputv), -1)

            if anti_number == 1:
                labelsv = sample_y[0].view(len(sample_y[0]), -1)
            else:
                labelsv = sample_y[0][:, :]


            weights = labelsv.data.clone().view(len(sample_y[0]), -1)
            # That step is added to handle missing outputs.
            # Weights are not updated in the presence of missing values.
            weights[weights == 1.0] = 1
            weights[weights == 0.0] = 1
            weights[weights < 0] = 0
            weights.to(device)


            # Calculate the loss/error using Binary Cross Entropy Loss

            criterion = nn.BCELoss(weight=weights, reduction="none")
            output = classifier(inputv)
            # print('------------',output.is_cuda)
            # print(labelsv.size())
            loss = criterion(output, labelsv)
            loss = loss.mean()  # compute loss

            loss.backward()  # backpropagation
            optimizer.step()  # weights updated
            losses.append(loss.item())

        if epoc % 100 == 0:
            # print the loss per iteration
            # print(losses)
            # print(np.average(losses))
            print('[%d/%d] Loss: %.3f' % (epoc + 1, epochs,  np.average(losses)))
    return classifier,epoc
def OuterLoop_training(classifier,epochs,optimizer,x_train,y_train,anti_number):
    #for each outer CV, use the best estimator selected from inner CVs.

    for epoc in range(epochs):
        x_train=x_train.to(device)
        y_train = y_train.to(device)
        x_train_new = torch.utils.data.TensorDataset(x_train)
        y_train_new = torch.utils.data.TensorDataset(y_train)

        all_data = list(zip(x_train_new, y_train_new))

        # the model is trained for 100 batches
        data_loader = torch.utils.data.DataLoader(
            all_data, batch_size=100, drop_last=False)

        losses = []  # save the error for each iteration
        for i, (sample_x, sample_y) in enumerate(data_loader):
            optimizer.zero_grad()  # zero gradients #previous gradients do not keep accumulating
            inputv = sample_x[0]
            inputv=inputv.to(device)

            # inputv = torch.FloatTensor(inputv)
            inputv = Variable(inputv).view(len(inputv), -1)
            # print(inputv.size())

            if anti_number == 1:
                labelsv = sample_y[0].view(len(sample_y[0]), -1)
            else:
                labelsv = sample_y[0][:, :]


            weights = labelsv.data.clone().view(len(sample_y[0]), -1)
            # That step is added to handle missing outputs.
            # Weights are not updated in the presence of missing values.
            weights[weights == 1.0] = 1
            weights[weights == 0.0] = 1
            weights[weights < 0] = 0
            weights.to(device)


            # Calculate the loss/error using Binary Cross Entropy Loss

            criterion = nn.BCELoss(weight=weights, reduction="none")
            output = classifier(inputv)
            loss = criterion(output, labelsv)
            loss = loss.mean()  # compute loss

            loss.backward()  # backpropagation
            optimizer.step()  # weights updated
            losses.append(loss.item())

        if epoc % 100 == 0:
            # print the loss per iteration
            print('[%d/%d] Loss: %.3f' % (epoc + 1, epochs, loss.item()))
    return classifier

def training(classifier,epochs,optimizer,x_train,y_train,x_val,y_val, anti_number,patience):

    print('with early stop setting..')
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    if anti_number==1:
        n_critical = 500
    else:
        n_critical=500 #for multi-anti model.


    for epoc in range(epochs):
        if epoc >( n_critical-1 )and epoc % 100 == 0:#July 16: change 1000 to 500. So so far, all the single species model are at lest 1000 epoches.

            if epoc==n_critical:
                print('starting earlystop monitor...')
                early_stopping = EarlyStopping(patience=patience, verbose=False) #starting checking after a certain time.
                early_stopping(valid_loss, classifier)
            else:

                # early_stopping needs the validation loss to check if it has decresed,
                # and if it has, it will make a checkpoint of the current model
                early_stopping(valid_loss, classifier)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

        ###################
        #1. train the model #
        ###################
        classifier.train()
        x_train=x_train.to(device)
        y_train = y_train.to(device)
        x_train_new = torch.utils.data.TensorDataset(x_train)
        y_train_new = torch.utils.data.TensorDataset(y_train)

        all_data = list(zip(x_train_new, y_train_new))

        # the model is trained for 100 batches
        data_loader = torch.utils.data.DataLoader(
            all_data, batch_size=100, drop_last=False)
        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []

        # losses = []  # save the error for each iteration
        for i, (sample_x, sample_y) in enumerate(data_loader):
            optimizer.zero_grad()  # zero gradients #previous gradients do not keep accumulating
            inputv = sample_x[0]
            inputv=inputv.to(device)

            # inputv = torch.FloatTensor(inputv)
            inputv = Variable(inputv).view(len(inputv), -1)
            # print(inputv.size())

            if anti_number == 1:
                labelsv = sample_y[0].view(len(sample_y[0]), -1)
            else:
                labelsv = sample_y[0][:, :]


            weights = labelsv.data.clone().view(len(sample_y[0]), -1)
            # That step is added to handle missing outputs.
            # Weights are not updated in the presence of missing values.
            weights[weights == 1.0] = 1
            weights[weights == 0.0] = 1
            weights[weights < 0] = 0
            weights.to(device)


            # Calculate the loss/error using Binary Cross Entropy Loss

            criterion = nn.BCELoss(weight=weights, reduction="none")
            output = classifier(inputv)
            # print('------------',output.is_cuda)
            # print(labelsv.size())
            loss = criterion(output, labelsv)
            loss = loss.mean()  # compute loss
            loss.backward()  # backpropagation
            optimizer.step()  # weights updated
            train_losses.append(loss.item())

        ######################
        # 2. validate the model #
        ######################
        if epoc>(n_critical-2) and (epoc+1) % 100 == 0:

            classifier.eval()  # prep model for evaluation
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            x_val_new = torch.utils.data.TensorDataset(x_val)
            y_val_new = torch.utils.data.TensorDataset(y_val)
            all_data = list(zip(x_val_new, y_val_new))

            # the model is trained for 100 batches
            data_loader = torch.utils.data.DataLoader(
                all_data, batch_size=100, drop_last=False)

            for i, (sample_x, sample_y) in enumerate(data_loader):
                inputv = sample_x[0]
                inputv = inputv.to(device)

                # inputv = torch.FloatTensor(inputv)
                inputv = Variable(inputv).view(len(inputv), -1)
                # print(inputv.size())

                if anti_number == 1:
                    labelsv = sample_y[0].view(len(sample_y[0]), -1)
                else:
                    labelsv = sample_y[0][:, :]

                weights = labelsv.data.clone().view(len(sample_y[0]), -1)
                # That step is added to handle missing outputs.
                # Weights are not updated in the presence of missing values.
                weights[weights == 1.0] = 1
                weights[weights == 0.0] = 1
                weights[weights < 0] = 0
                weights.to(device)

                # Calculate the loss/error using Binary Cross Entropy Loss

                criterion = nn.BCELoss(weight=weights, reduction="none")
                output = classifier(inputv)
                # calculate the loss
                loss = criterion(output, labelsv)
                # record validation loss
                loss = loss.mean()
                # print(loss.size())
                valid_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        if (epoc+1) % 100 == 0:
            # print the loss per iteration
            # print('[%d/%d] Loss: %.3f' % (epoc + 1, epochs, loss.item()))
            epoch_len = len(str(epochs))
            print_msg = (f'[{epoc:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
            print(print_msg)

    return classifier,epoc


def hyper_range(anti_number,f_no_early_stop,antibiotics):
    if f_no_early_stop==True: #Aytan-Aktug default
        print('please do not use this option, because no patience is included in the hyper=para selection. ',
              'If you really want to use it, we use the default hyper-para in the article Aytan-Aktug et al. 2020.')
        if anti_number==1:
            hyper_space={'n_hidden': [200], 'epochs': [1000],'lr':[0.001],'classifier':[4],'dropout': [None],'patience': [None]} #default setting in Derya Aytan's work.
        else:
            hyper_space = {'n_hidden': [200], 'epochs': [5000],'lr': [0.001],'classifier':[4],'dropout': [None],'patience': [None]} #default setting in Derya Aytan's work.
    else:
        if anti_number==1:
            hyper_space = {'n_hidden': [200], 'epochs': [10000], 'lr': [0.001, 0.0005],
                           'classifier': [1], 'dropout': [0, 0.2], 'patience': [2]}

        elif type(antibiotics)==list:# multi-anti model.
            hyper_space = {'n_hidden': [200], 'epochs': [50000], 'lr': [0.001],
                           'classifier': [1], 'dropout': [0, 0.2], 'patience': [2]}
        else: #discrete multi-model and concat model for comparison.
            hyper_space = {'n_hidden': [200,400], 'epochs': [30000], 'lr': [0.0005,0.0001],
                           'classifier': [1,2],'dropout':[0,0.2],'patience':[2]}

    return hyper_space

def eval(species, antibiotics, level, xdata, ydata,p_names, cv,f_scaler,f_fixed_threshold,f_no_early_stop,f_phylotree,
         f_kma,f_optimize_score,save_name_weights):

    #data
    data_x = np.loadtxt(xdata, dtype="float")
    data_y =np.loadtxt(ydata)
    print('dataset shape',data_x.shape)

    # output results list
    pred_test = []  # probabilities are for the validation set
    # pred_test_binary = [] #binary based on selected
    thresholds_selected_test=[]# cv len, each is the selected threshold.
    weight_test=[]# cv len, each is the index of the selected model in inner CV,
    mcc_test = []  # MCC results for the test data
    f1_test = []
    score_report_test = []
    aucs_test = []  # all AUC values for the test data
    predictY_test=[] #predicted Y
    true_Y=[] #corresponding true Y
    sampleNames_test=[] #sample names
    # aucs_test_all=[]# multi-output and plotting used
    # tprs_test = []  # all True Positives for the test data
    mean_fpr = np.linspace(0, 1, 100)
    hyperparameters_test=[]
    actual_epoc_test = []
    actual_epoc_test_std = []
    # -------------------------------------------------------------------------------------------------------------------
    # model varibales
    anti_number = data_y[0].size  # number of antibiotics
    nlabel = data_y[0].size  # number of neurons in the output layer
    D_input = len(data_x[0])  # number of neurons in the input layer
    N_sample = len(data_x)  # number of input samples #should be equal to len(names)

    #fodls
    folds_txt=name_utility.GETname_folds(species,antibiotics,level,f_kma,f_phylotree)
    folders_sample_name = json.load(open(folds_txt, "rb"))
    folders_sample=name2index.Get_index(folders_sample_name,p_names)


    #hyper parameter range
    hyper_space = hyper_range(anti_number, f_no_early_stop, antibiotics)
    for out_cv in range(cv):

        train_val_samples= folders_sample[:out_cv] + folders_sample[out_cv+1 :]#list
        Validation_mcc_thresholds = []  #  inner CV *11 thresholds value
        Validation_f1_thresholds = []  #  inner CV *11 thresholds value
        Validation_auc = []  # inner CV
        Validation_actul_epoc=[]

        #later choose the inner loop and relevant thresholds with the highest f1 score
        # ---------------------------
        # select testing folder
        test_samples = folders_sample[out_cv]
        x_test = data_x[test_samples]
        y_test = data_y[test_samples]
        print('x_test shape',x_test.shape)
        for innerCV in range(cv - 1):  # e.g. 1,2,3,4
            print('Starting outer: ', str(out_cv), '; inner: ', str(innerCV), ' inner loop...')

            val_samples=train_val_samples[innerCV]
            train_samples=train_val_samples[:innerCV] + train_val_samples[innerCV+1 :]#only works for list, not np
            train_samples=list(itertools.chain.from_iterable(train_samples))
            x_train, x_val = data_x[train_samples], data_x[val_samples]#only np
            y_train, y_val = data_y[train_samples], data_y[val_samples]
            print('sample length and feature length for inner CV(train & val):',len(x_train), len(x_train[0]),len(x_val), len(x_val[0]))
            # vary for each CV

            # normalize the data
            if f_scaler==True:
                scaler = preprocessing.StandardScaler().fit(x_train)
                x_train = scaler.transform(x_train)
                # scale the val data based on the training data
                x_val = scaler.transform(x_val)

            # In regards of the predicted response values they dont need to be in the range of -1,1.
            x_train = torch.from_numpy(x_train).float()
            y_train = torch.from_numpy(y_train).float()
            x_val = torch.from_numpy(x_val).float()
            y_val = torch.from_numpy(y_val).float()
            pred_val_inner = []  # predicted results on validation set.
            ###################
            # 1. train the model #
            ###################
            # print(hyper_space)
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
                # print('Hyper_parameters:',n_cl)
                # print(epochs,n_hidden,learning)
                classifier.to(device)
                # generate the optimizer - Stochastic Gradient Descent
                optimizer = optim.SGD(classifier.parameters(), lr=learning)
                optimizer.zero_grad()  # Clears existing gradients from previous epoch
                #loop:
                print(species, antibiotics, out_cv, innerCV)
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
                            # tprs.append(interp(mean_fpr, fpr, tpr))
                            # tprs[-1][0] = 0.0
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
                                mcc = matthews_corrcoef(y_val_multi_sub[:, i], y_val_pred_multi_sub[:, i])
                                f1 = f1_score(y_val_multi_sub[:, i], y_val_pred_multi_sub[:, i], average='macro')
                                mcc_sub_anti.append(mcc)
                                f1_sub_anti.append(f1)
                        if anti_number > 1:  # multi-out, scores based on mean of all the involved antibotics
                            mcc_sub.append(statistics.mean(mcc_sub_anti))  # mcc_sub_anti dimension: n_anti
                            f1_sub.append(statistics.mean(f1_sub_anti))


                    Validation_mcc_thresholds_split.append(mcc_sub)
                    Validation_f1_thresholds_split.append(f1_sub)
                    # print(Validation_f1_thresholds)
                    # print(Validation_mcc_thresholds)

            Validation_mcc_thresholds.append(Validation_mcc_thresholds_split)  # inner CV * hyperpara_combination
            Validation_f1_thresholds.append(Validation_f1_thresholds_split)  # inner CV * hyperpara_combination
            Validation_auc.append(Validation_auc_split)  # inner CV * hyperpara_combination
            Validation_actul_epoc.append(Validation_actul_epoc_split) # inner CV * hyperpara_combination

        # finish evaluation in the inner loop.
        if f_fixed_threshold==True and f_optimize_score=='f1_macro':#finished.May 30. 7am
            thresholds_selected=0.5
            Validation_f1_thresholds = np.array(Validation_f1_thresholds)
            Validation_f1_thresholds = Validation_f1_thresholds.mean(axis=0)
            Validation_actul_epoc = np.array(Validation_actul_epoc)
            Validation_actul_epoc_std = Validation_actul_epoc.std(axis=0)
            Validation_actul_epoc = Validation_actul_epoc.mean(axis=0)

            #select the inner loop index with the highest f1 score in the column w.r.t. 0.5
            aim_column=np.where(np.arange(0, 1.1, 0.1) == 0.5)
            # aim_column=aim_column[0][0]
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
            weights_selected=ind[0]
            hyperparameters_test.append(list(ParameterGrid(hyper_space))[ind[0]])
            actual_epoc_test.append(Validation_actul_epoc[ind[0]])
            actual_epoc_test_std.append(Validation_actul_epoc_std[ind[0]])
        elif f_optimize_score=='auc':
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
        # epochs_testing = hyperparameters_test[out_cv]['epochs']
        n_hidden = hyperparameters_test[out_cv]['n_hidden']
        learning = hyperparameters_test[out_cv]['lr']
        n_cl = hyperparameters_test[out_cv]['classifier']
        dropout=hyperparameters_test[out_cv]['dropout']
        # generate a NN model
        if n_cl == 1:
            classifier = _classifier(nlabel, D_input, n_hidden,dropout)  # reload, after testing(there is fine tune traning!)
        elif n_cl == 2:
            classifier = _classifier2(nlabel, D_input, n_hidden,dropout)
        elif n_cl == 3:
            classifier = _classifier3(nlabel, D_input, n_hidden,dropout)
        # print( 'testing hyper-parameter: ',hyperparameters_test[out_cv])
        classifier.to(device)
        # generate the optimizer - Stochastic Gradient Descent
        optimizer = optim.SGD(classifier.parameters(), lr=learning)
        optimizer.zero_grad()  # Clears existing gradients from previous epoch
        classifier.train()
        # print('actual_epoc_test[out_cv]',actual_epoc_test[out_cv])
        classifier = OuterLoop_training(classifier, int(actual_epoc_test[out_cv]), optimizer, x_train_val, y_train_val, anti_number)

        torch.save(classifier.state_dict(), save_name_weights+'_weights_cv'+str(out_cv)) #save the learned model.
        classifier.eval()


        pred_test_sub = []
        for a, a_sample in enumerate(x_test):
            tested = Variable(a_sample).view(1, -1)
            # tested = Variable(torch.FloatTensor(a_sample)).view(1, -1)
            output_test = classifier(tested)
            out = output_test
            temp = []
            for h in out[0]:
                temp.append(float(h))
            pred_test_sub.append(temp)
        pred_test_sub = np.array(pred_test_sub)
        pred_test.append(pred_test_sub)#len= y_test
        print('x_test',x_test.shape)
        print('pred_test_sub',pred_test_sub.shape)
        print('y_test',y_test.shape)

        #turn probability to binary.
        threshold_matrix=np.full(pred_test_sub.shape, thresholds_selected)
        pred_test_binary_sub = (pred_test_sub > threshold_matrix)
        pred_test_binary_sub = 1 * pred_test_binary_sub

        y_test = np.array(y_test)
        # pred_test_binary.append(pred_test_binary_sub)
        f1_test_anti=[]# multi-out
        score_report_test_anti=[]# multi-out
        tprs = []# multi-out
        aucs_test_anti=[]# multi-out
        mcc_test_anti=[]# multi-out

        for i in range(anti_number):
            comp = []
            if anti_number == 1:
                mcc = matthews_corrcoef(y_test, pred_test_binary_sub)
                # report = classification_report(y_val, y_val_pred, labels=[0, 1], output_dict=True)
                f1 = f1_score(y_test, pred_test_binary_sub, average='macro')
                report = classification_report(y_test, pred_test_binary_sub, labels=[0, 1],output_dict=True)
                fpr, tpr, _ = roc_curve(y_test, pred_test_binary_sub, pos_label=1)
                # tprs.append(interp(mean_fpr, fpr, tpr))
                # tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)

            else:  # multi-out
                for t in range(len(y_test)):
                    if -1 != y_test[t][i]:
                        comp.append(t)
                y_test_anti=y_test[comp]
                pred_test_binary_sub_anti=pred_test_binary_sub[comp]
                pred_test_sub_anti=pred_test_sub[comp]#June 21st, auc bugs

                mcc = matthews_corrcoef(y_test_anti[:, i], pred_test_binary_sub_anti[:, i])
                f1 = f1_score(y_test_anti[:, i], pred_test_binary_sub_anti[:, i], average='macro')
                fpr, tpr, _ = roc_curve(y_test_anti[:, i], pred_test_sub_anti[:, i], pos_label=1)#June 21st, auc bugs
                roc_auc = auc(fpr, tpr)
                # tprs.append(interp(mean_fpr, fpr, tpr))
                # tprs[-1][0] = 0.0
                # aucs.append(roc_auc)
                report = classification_report(y_test_anti[:, i], pred_test_binary_sub_anti[:, i], labels=[0, 1], output_dict=True)

                f1_test_anti.append(f1)
                score_report_test_anti.append(report)
                aucs_test_anti.append(roc_auc)
                mcc_test_anti.append(mcc)


        #summerise each outer loop's results.
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
        # tprs_test.append(tprs)  ## multi-out

    actual_epoc_test=Num2l(actual_epoc_test)
    actual_epoc_test_std=Num2l(actual_epoc_test_std)


    score = {'f1_test':f1_test,'score_report_test':score_report_test,'aucs_test':aucs_test,'mcc_test':mcc_test,
             'predictY_test':predictY_test,'ture_Y':true_Y,'samples':sampleNames_test,'thresholds_selected_test':thresholds_selected_test,
             'hyperparameters_test':hyperparameters_test,'actual_epoc_test':actual_epoc_test,'actual_epoc_test_std':actual_epoc_test_std}

    torch.cuda.empty_cache()
    return score



