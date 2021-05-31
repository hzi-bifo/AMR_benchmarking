#!/usr/bin/python
#Python 3.6
#nested CV, modified by khu based on Derya Aytan's work.
#https://bitbucket.org/deaytan/neural_networks/src/master/Neural_networks.py
#Note: this scritp should be used as a module, otherwise the storage will be disrupted.
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import sys
sys.path.append('../')
sys.path.insert(0, os.getcwd())
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
import getopt
import sys
import warnings
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import ParameterGrid
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,confusion_matrix,classification_report,f1_score
from scipy import interp
import collections
import random
from sklearn import utils
from sklearn import preprocessing
import argparse
import ast
import amr_utility.name_utility
import amr_utility.file_utility
import itertools
import statistics
from pytorchtools import EarlyStopping
import pickle
import copy
import neural_networks.cluster_folders
'''
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''
use_cuda = torch.cuda.is_available()
# # use_cuda=False
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
# print(torch.cuda.device_count(),torch.cuda.get_device_name(0))
# print('torch.cuda.current_device()', torch.cuda.current_device())
# print(device)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
class _classifier3(nn.Module):
    def __init__(self, nlabel,D_in,H):
        super(_classifier3, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, nlabel),
        )

    def forward(self, input):
        out=self.main(input)
        out=nn.Sigmoid()(out)
        return out
class _classifier2(nn.Module):
    def __init__(self, nlabel,D_in,H):
        super(_classifier2, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Linear(H, nlabel),
        )

    def forward(self, input):
        out=self.main(input)
        out=nn.Sigmoid()(out)
        return out

class _classifier(nn.Module):
    def __init__(self, nlabel,D_in,H):
        super(_classifier, self).__init__()
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
            # print('------------',output.is_cuda)
            # print(labelsv.size())
            loss = criterion(output, labelsv)
            loss = loss.mean()  # compute loss

            loss.backward()  # backpropagation
            optimizer.step()  # weights updated
            losses.append(loss.item())

        if epoc % 100 == 0:
            # print the loss per iteration
            print('[%d/%d] Loss: %.3f' % (epoc + 1, epochs, loss.item()))
    return classifier,epoc

def training(classifier,epochs,optimizer,x_train,y_train,x_val,y_val, anti_number):
    patience = 200
    print('with early stop setting..')
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    for epoc in range(epochs):
        if epoc>1000:
            if epoc==1001:
                early_stopping = EarlyStopping(patience=patience, verbose=False) #starting checking after a certain time.
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
        if epoc % 100 == 0:
            # print the loss per iteration
            # print('[%d/%d] Loss: %.3f' % (epoc + 1, epochs, loss.item()))
            epoch_len = len(str(epochs))
            print_msg = (f'[{epoc:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
            print(print_msg)

    return classifier,epoc

def fine_tune_training(classifier,epochs,optimizer,x_train,y_train,anti_number):
    #for each outer CV, use the best estimator selected from inner CVs. The selected estimator are fine tuned, using both
    # validation and training data.

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

# def eval(species, antibiotics, level, xdata, ydata, p_names, p_clusters, cv, Random_State, hidden, epochs,re_epochs,
#          learning,f_scaler,f_fixed_threshold,f_no_early_stop,f_optimize_score,save_name_score,concat_merge_name,threshold_point,min_cov_point):
def eval(species, antibiotics, level, xdata, ydata, p_names, p_clusters, cv, Random_State,
         re_epochs, f_scaler,f_fixed_threshold,f_no_early_stop, f_optimize_score, save_name_score,concat_merge_name,threshold_point,min_cov_point):

    #data
    data_x = np.loadtxt(xdata, dtype="float")
    data_y =np.loadtxt(ydata)

    ##prepare data stores for testing scores##

    pred_test = []  # probabilities are for the validation set
    pred_test_binary = [] #binary based on selected
    thresholds_selected_test=[]# cv len, each is the selected threshold.
    weight_test=[]# cv len, each is the index of the selected model in inner CV,
    # only this weights are preseved in /log/temp/

    mcc_test = []  # MCC results for the test data
    f1_test = []
    score_report_test = []

    aucs_test = []  # all AUC values for the test data
    aucs_test_all=[]# multi-output and plotting used
    tprs_test = []  # all True Positives for the test data
    mean_fpr = np.linspace(0, 1, 100)
    hyperparameters_test=[]
    actual_epoc_test = []
    actual_epoc_test_std = []
    # -------------------------------------------------------------------------------------------------------------------
    ###construct the Artificial Neural Networks Model###
    # The feed forward NN has only one hidden layer
    # The activation function used in the input and hidden layer is ReLU, in the output layer the sigmoid function.
    # -------------------------------------------------------------------------------------------------------------------


    # n_hidden = hidden# number of neurons in the hidden layer
    # learning_rate = learning

    anti_number = data_y[0].size  # number of antibiotics
    nlabel = data_y[0].size  # number of neurons in the output layer

    D_input = len(data_x[0])  # number of neurons in the input layer
    N_sample = len(data_x)  # number of input samples #should be equal to len(names)


    # cross validation loop where the training and testing performed.
    #khu:nested CV.
    # =====================================
    # training
    # =====================================
    # dimension: cv*(sample_n in each split(it varies))
    # element: index of sampels w.r.t. data_x, data_y
    folders_sample,_,_ = neural_networks.cluster_folders.prepare_folders(cv, Random_State, p_names, p_clusters, 'new')
    if f_no_early_stop==True:
        if anti_number==1:
            hyper_space={'n_hidden': [200,300], 'epochs': [1000,2000, 3000,4000],'lr':[0.001,0.0005],'classifier':[1]}

        else:
            hyper_space = {'n_hidden': [200,300,400], 'epochs': [2000, 3000,4000,5000], 'lr': [ 0.001,0.0005,0.0001],
                           'classifier': [1,2,3]}

    else:
        if anti_number==1:
            hyper_space = {'n_hidden': [200,300], 'epochs': [20000], 'lr': [ 0.001,0.0005,0.0001],
                           'classifier': [1]}
            # hyper_space = {'n_hidden': [200, 300], 'epochs': [200], 'lr': [0.001, 0.0005],
            #                'classifier': [1]}
        else:
            hyper_space = {'n_hidden': [200,300,400], 'epochs': [30000], 'lr': [ 0.001,0.0005,0.0001],
                           'classifier': [1,2,3]}
            # hyper_space = {'n_hidden': [200, 300], 'epochs': [200], 'lr': [0.001, 0.0005],
            #                'classifier': [1, 2]}




    for out_cv in range(cv):
        #select testing folder
        test_samples=folders_sample[out_cv]
        x_test=data_x[test_samples]
        y_test = data_y[test_samples]
        x_test = torch.from_numpy(x_test).float()
        y_test = torch.from_numpy(y_test).float()
        x_test = x_test.to(device)
        # remain= list(set(range(cv)) - set([out_cv])).sort()#first round: set(0,1,2,3,4)-set(0)=set(1,2,3,4)
        train_val_samples= folders_sample[:out_cv] + folders_sample[out_cv+1 :]#list
        Validation_mcc_thresholds = []  #  inner CV *11 thresholds value
        Validation_f1_thresholds = []  #  inner CV *11 thresholds value
        Validation_auc = []  # inner CV
        Validation_actul_epoc=[]
        # Validation_mcc = []  # len=inner CV
        # Validation_f1 = []  # len=inner CV
        #later choose the inner loop and relevant thresholds with the highest f1 score


        for innerCV in range(cv - 1):  # e.g. 1,2,3,4
            print('Starting outer: ', str(out_cv), '; inner: ', str(innerCV), ' inner loop...')

            val_samples=train_val_samples[innerCV]
            train_samples=train_val_samples[:innerCV] + train_val_samples[innerCV+1 :]#only works for list, not np
            train_samples=list(itertools.chain.from_iterable(train_samples))
            # training and val samples
            # select by order

            x_train, x_val = data_x[train_samples], data_x[val_samples]#only np
            y_train, y_val = data_y[train_samples], data_y[val_samples]
            print('sample length and feature length for inner CV(train & val):',len(x_train), len(x_train[0]),len(x_val), len(x_val[0]))
            # vary for each CV


            # x_train = x_train.to(device)
            # y_train = y_train.to(device)

            # normalize the data
            if f_scaler==True:
                scaler = preprocessing.StandardScaler().fit(x_train)
                x_train = scaler.transform(x_train)
                # scale the val data based on the training data
                scaler = preprocessing.StandardScaler().fit(x_train)
                x_val = scaler.transform(x_val)

            # In regards of the predicted response values they dont need to be in the range of -1,1.
            x_train = torch.from_numpy(x_train).float()
            y_train = torch.from_numpy(y_train).float()
            x_val = torch.from_numpy(x_val).float()

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
                y_val = torch.from_numpy(y_val).float()
                # --------------------------------------------------------------------
                epochs=list(ParameterGrid(hyper_space))[grid_iteration]['epochs']
                n_hidden=list(ParameterGrid(hyper_space))[grid_iteration]['n_hidden']
                learning=list(ParameterGrid(hyper_space))[grid_iteration]['lr']
                n_cl=list(ParameterGrid(hyper_space))[grid_iteration]['classifier']
                # generate a NN model
                if n_cl==1:
                    classifier = _classifier(nlabel, D_input, n_hidden)#reload, after testing(there is fine tune traning!)
                elif n_cl==2:
                    classifier = _classifier2(nlabel, D_input, n_hidden)
                elif n_cl==3:
                    classifier = _classifier3(nlabel, D_input, n_hidden)
                print(list(ParameterGrid(hyper_space))[grid_iteration])
                # print('Hyper_parameters:',n_cl)
                # print(epochs,n_hidden,learning)
                classifier.to(device)
                # generate the optimizer - Stochastic Gradient Descent
                optimizer = optim.SGD(classifier.parameters(), lr=learning)
                optimizer.zero_grad()  # Clears existing gradients from previous epoch
                #loop:
                if f_no_early_stop==True:
                    classifier,actual_epoc = training_original(classifier,epochs,optimizer,x_train,y_train,anti_number)
                else:
                    classifier,actual_epoc =training(classifier, epochs, optimizer, x_train, y_train,x_val,y_val, anti_number)
                Validation_actul_epoc_split.append(actual_epoc)

                #==================================================

                print(species, antibiotics,level, out_cv, innerCV)
                #if hyperparameter selectio mode, set learning and epoch as 0 for naming the output.
                # name_weights = amr_utility.name_utility.GETname_multi_bench_weight(concat_merge_name,species, antibiotics,level, out_cv, innerCV,0.0,0,f_fixed_threshold,f_no_early_stop,f_optimize_score,threshold_point,min_cov_point)
                # print(name_weights)
                # amr_utility.file_utility.make_dir(os.path.dirname(name_weights))#make folders for storing weights
                # torch.save(classifier.state_dict(), name_weights)

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
                # pred_val_inner.append(pred_val_sub)#for all innerCV at this out_cv. #No further use so far. April,30.
                print('pred_val_sub shape:', pred_val_sub.shape)#(cv-1)*n_sample
                print('y_val',np.array(y_val).shape)

                #--------------------------------------------------
                #auc score, threshould operation is contained in itself definition.
                #khu add: 13May
                print('f_optimize_score:',f_optimize_score)
                if f_optimize_score=='auc':

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
                            # todo check
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
                    #todo for multi-species!!!
                    for threshold in np.arange(0, 1.1, 0.1):
                        # predictions for the test data
                        #turn probabilty to binary
                        threshold_matrix = np.full(pred_val_sub.shape, threshold)
                        y_val_pred = (pred_val_sub > threshold_matrix)
                        y_val_pred = 1*y_val_pred

                        y_val = np.array(y_val)  # ground truth

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
            aim_f1=Validation_f1_thresholds[:,aim_column]
            weights_selected=np.argmax(aim_f1)
            ind=np.unravel_index(np.argmax(aim_f1, axis=None), aim_f1.shape)
            print('ind',ind)
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
            print('weights--------------',weights_selected)
            print('ind',ind)
            hyperparameters_test.append(list(ParameterGrid(hyper_space))[ind[0]])
            print('Validation_f1_thresholds.shape, should be related to the hyper-space?',Validation_f1_thresholds.shape)
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
        # name_weights = amr_utility.name_utility.GETname_multi_bench_weight(concat_merge_name,species, antibiotics, level,out_cv, weights_selected,0.0,0,f_fixed_threshold,f_no_early_stop,f_optimize_score,threshold_point,min_cov_point)
        # classifier.load_state_dict(torch.load(name_weights))




        #finish innner loop
        #=======================================================
        #3. Re-train on both train and val data with the weights

        print('Outer loop, testing.')
        train_val_samples = list(itertools.chain.from_iterable(train_val_samples))
        x_train_val= data_x[train_val_samples] # only np
        y_train_val= data_y[train_val_samples]

        if f_scaler==True:
            scaler = preprocessing.StandardScaler().fit(x_train_val)
            x_train_val = scaler.transform(x_train_val)
            # scale the val data based on the training data
            scaler = preprocessing.StandardScaler().fit(x_train_val)
            x_test = scaler.transform(x_test)

        x_train_val = torch.from_numpy(x_train_val).float()
        y_train_val = torch.from_numpy(y_train_val).float()

        #-
        # epochs_testing = hyperparameters_test[out_cv]['epochs']
        n_hidden = hyperparameters_test[out_cv]['n_hidden']
        learning = hyperparameters_test[out_cv]['lr']
        n_cl = hyperparameters_test[out_cv]['classifier']
        # generate a NN model
        if n_cl == 1:
            classifier = _classifier(nlabel, D_input, n_hidden)  # reload, after testing(there is fine tune traning!)
        elif n_cl == 2:
            classifier = _classifier2(nlabel, D_input, n_hidden)
        elif n_cl == 3:
            classifier = _classifier3(nlabel, D_input, n_hidden)
        print( 'testing hyper-parameter: ',hyperparameters_test[out_cv])
        # print('Hyper_parameters:',n_cl)
        # print(epochs,n_hidden,learning)
        classifier.to(device)
        # generate the optimizer - Stochastic Gradient Descent
        optimizer = optim.SGD(classifier.parameters(), lr=learning)
        optimizer.zero_grad()  # Clears existing gradients from previous epoch
        classifier.train()
        # optimizer = optim.SGD(classifier.parameters(), lr=0.0001)
        # classifier = fine_tune_training(classifier, re_epochs, optimizer, x_train_val, y_train_val, anti_number)
        print('actual_epoc_test[out_cv]',actual_epoc_test[out_cv])
        classifier = fine_tune_training(classifier, int(actual_epoc_test[out_cv]), optimizer, x_train_val, y_train_val, anti_number)
        name_weights = amr_utility.name_utility.GETname_multi_bench_weight(concat_merge_name,species, antibiotics,level, out_cv,'',0.0,0,f_fixed_threshold,f_no_early_stop,f_optimize_score,threshold_point,min_cov_point)

        torch.save(classifier.state_dict(), name_weights)

        # # rm inner loop models' weight in the log
        # for i in np.arange(cv-1):
        #     n = amr_utility.name_utility.GETname_multi_bench_weight(concat_merge_name,species, antibiotics,level, out_cv, i,0.0,0,f_fixed_threshold,f_no_early_stop,f_optimize_score,threshold_point,min_cov_point)
        #     os.system("rm " + n)

        # apply the trained model to the test data
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
        # 4. get measurement scores on testing set.
        #Positive predictive value (PPV), Precision; Accuracy (ACC); f1-score here,
        # y: True positive rate (TPR), aka. sensitivity, hit rate, and recall,
        # X: False positive rate (FPR), aka. fall-out,

        #turn probability to binary.
        threshold_matrix=np.full(pred_test_sub.shape, thresholds_selected)
        pred_test_binary_sub = (pred_test_sub > threshold_matrix)
        pred_test_binary_sub = 1 * pred_test_binary_sub

        y_test = np.array(y_test)
        # pred_test_binary_sub = np.array(pred_test_binary_sub)
        pred_test_binary.append(pred_test_binary_sub)


        # print(y_test)
        # print(pred_test_binary_sub)
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
                print(f1)
                print(mcc)
                report = classification_report(y_test, pred_test_binary_sub, labels=[0, 1],output_dict=True)
                print(report)
                fpr, tpr, _ = roc_curve(y_test, pred_test_binary_sub, pos_label=1)

                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                # aucs.append(roc_auc)


            else:  # multi-out
                #todo check
                for t in range(len(y_test)):
                    if -1 != y_test[t][i]:
                        comp.append(t)
                y_test_anti=y_test[comp]
                pred_test_binary_sub_anti=pred_test_binary_sub[comp]

                mcc = matthews_corrcoef(y_test_anti[:, i], pred_test_binary_sub_anti[:, i])
                f1 = f1_score(y_test_anti[:, i], pred_test_binary_sub_anti[:, i], average='macro')
                fpr, tpr, _ = roc_curve(y_test_anti[:, i], pred_test_binary_sub_anti[:, i], pos_label=1)
                report = classification_report(y_test_anti[:, i], pred_test_binary_sub_anti[:, i], labels=[0, 1], output_dict=True)
                roc_auc = auc(fpr, tpr)
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                # aucs.append(roc_auc)
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

        # aucs_test_all.append(aucs)  ## multi-out
        tprs_test.append(tprs)  ## multi-out





    # plot(anti_number, mcc_test, cv, validation, pred_val_all, validation_y, tprs_test, aucs_test_all, mean_fpr)


    print('thresholds_selected_test',thresholds_selected_test)
    print('f1_test',f1_test)
    print('mcc_test',mcc_test)
    print('hyperparameters_test',hyperparameters_test)
    # score_summary(cv, score_report_test, aucs_test, mcc_test, save_name_score,thresholds_selected_test)#save mean and std of each 6 score
    score = [thresholds_selected_test, f1_test, mcc_test, score_report_test, aucs_test, tprs_test,hyperparameters_test,actual_epoc_test,actual_epoc_test_std]
    with open(save_name_score + '_all_score.pickle', 'wb') as f:  # overwrite
        pickle.dump(score, f)


    torch.cuda.empty_cache()
    return score


def test(hyperparameters,species, antibiotics, weights,threshold_select,level, xdata, ydata, p_names, p_clusters, cv, Random_State, n_hidden, epochs,re_epochs, learning,f_scaler,f_fixed_threshold,f_no_early_stop,f_optimize_score):
    x_test = np.loadtxt(xdata, dtype="float")
    y_test = np.loadtxt(ydata)
    anti_number = y_test[0].size  # number of neurons in the output layer
    D_input = len(x_test[0])  # number of neurons in the input layer
    if f_scaler == True:
        scaler = preprocessing.StandardScaler().fit(x_test)
        x_test = scaler.transform(x_test)
    data_x = torch.from_numpy(x_test).float()
    data_y = torch.from_numpy(y_test).float()
    data_x=data_x.to(device)

    #-load in learned hyper-parameters-----------
    # epochs = hyperparameters['epochs']
    n_hidden = hyperparameters['n_hidden']
    # learning = hyperparameters['lr']
    n_cl = hyperparameters['classifier']
    # generate a NN model
    if n_cl == 1:
        classifier = _classifier(anti_number, D_input, n_hidden)  # reload, after testing(there is fine tune traning!)
    elif n_cl == 2:
        classifier = _classifier2(anti_number, D_input, n_hidden)
    elif n_cl == 3:
        classifier = _classifier3(anti_number, D_input, n_hidden)

    # classifier = _classifier(anti_number, D_input, n_hidden)  # reload, after testing(there is fine tune traning!)
    classifier.to(device)
    classifier.load_state_dict(torch.load(weights))
    classifier.eval()

    pred_test = []
    for a, a_sample in enumerate(data_x):
        tested = Variable(a_sample).view(1, -1)
        # tested = Variable(torch.FloatTensor(a_sample)).view(1, -1)
        output_test = classifier(tested)
        out = output_test
        temp = []
        for h in out[0]:
            temp.append(float(h))
        pred_test.append(temp)
    pred_test = np.array(pred_test)
    # pred_test.append(pred_test_sub)  # len= y_test
    print('x_test', data_x.shape)
    print('pred_test_sub', pred_test.shape)
    print('y_test', data_y.shape)

    # turn probability to binary.
    threshold_matrix = np.full(pred_test.shape, threshold_select)
    pred_test_binary_sub = (pred_test > threshold_matrix)
    pred_test_binary_sub = 1 * pred_test_binary_sub

    y_test = np.array(y_test)
    # pred_test_binary_sub = np.array(pred_test_binary_sub)
    # pred_test_binary.append(pred_test_binary_sub)

    # print(y_test)
    # print(pred_test_binary_sub)
    f1_test_anti = []  # multi-out
    score_report_test_anti = []  # multi-out
    tprs = []  # multi-out
    aucs_test_anti = []  # multi-out
    mcc_test_anti = []  # multi-out
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(anti_number):
        comp = []
        if anti_number == 1:
            mcc = matthews_corrcoef(y_test, pred_test_binary_sub)
            # report = classification_report(y_val, y_val_pred, labels=[0, 1], output_dict=True)
            f1 = f1_score(y_test, pred_test_binary_sub, average='macro')
            print(f1)
            print(mcc)
            report = classification_report(y_test, pred_test_binary_sub, labels=[0, 1], output_dict=True)
            print(report)
            fpr, tpr, _ = roc_curve(y_test, pred_test_binary_sub, pos_label=1)

            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            # aucs.append(roc_auc)


        else:  # multi-out
            # todo check
            for t in range(len(y_test)):
                if -1 != y_test[t][i]:
                    comp.append(t)
            y_test_anti = y_test[comp]
            pred_test_binary_sub_anti = pred_test_binary_sub[comp]
            # print('prey_test', pred_test_binary_sub.shape)
            # print('precomy_test', pred_test_binary_sub_anti.shape)
            # print(comp)
            if comp!=[]:
                mcc = matthews_corrcoef(y_test_anti[:, i], pred_test_binary_sub_anti[:, i])
                f1 = f1_score(y_test_anti[:, i], pred_test_binary_sub_anti[:, i], average='macro')
                fpr, tpr, _ = roc_curve(y_test_anti[:, i], pred_test_binary_sub_anti[:, i], pos_label=1)
                report = classification_report(y_test_anti[:, i], pred_test_binary_sub_anti[:, i], labels=[0, 1],
                                               output_dict=True)
                roc_auc = auc(fpr, tpr)
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                # aucs.append(roc_auc)
                f1_test_anti.append(f1)
                score_report_test_anti.append(report)
                aucs_test_anti.append(roc_auc)
                mcc_test_anti.append(mcc)
            else:
                f1_test_anti.append(None)
                score_report_test_anti.append(None)
                aucs_test_anti.append(None)
                mcc_test_anti.append(None)

    return [threshold_select,f1_test_anti,mcc_test_anti,score_report_test_anti,aucs_test_anti,tprs]
