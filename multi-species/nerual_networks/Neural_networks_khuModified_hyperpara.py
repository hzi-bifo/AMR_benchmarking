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
import time
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
            # print(losses)
            # print(np.average(losses))
            print('[%d/%d] Loss: %.3f' % (epoc + 1, epochs,  np.average(losses)))
    return classifier,epoc

def training(classifier,epochs,optimizer,x_train,y_train,x_val,y_val, anti_number,patience):
    # patience = 200
    print('with early stop setting..')
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    if anti_number==1:
        n_critical = 1000
    else:
        n_critical=500

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

def fine_tune_training(classifier,epochs,optimizer,x_train,y_train,anti_number):
    #for each outer CV, use the best estimator selected from inner CVs. The selected estimator are fine tuned, using both
    # validation and training data.
    # July 17: seems not based on previous weights

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

def hyper_range(anti_number,f_no_early_stop,antibiotics):
    if f_no_early_stop==True:
        print('please do not use this option, because no patience is included in the hyper=para selection. ',
              'If you really want to use it, we use the default hyper-para in the article Aytan-Aktug et al. 2020.')
        if anti_number==1:
            hyper_space={'n_hidden': [200], 'epochs': [1000],'lr':[0.001],'classifier':[1],
                         'dropout':[0],'patience':[1000]}

        else:
            pass
            # hyper_space = {'n_hidden': [200,300,400], 'epochs': [2000,3000,4000,5000,6000], 'lr': [ 0.001,0.0005,0.0001],
            #                'classifier': [1,2,3],'dropout':[0,0.2,0.5],'patience':[200,600,1000,30000]}

    else:
        if anti_number==1:
            hyper_space = {'n_hidden': [200], 'epochs': [10000], 'lr': [0.001, 0.0005],
                           'classifier': [1], 'dropout': [0, 0.2], 'patience': [2]}  # June.13 th. July 16. delete patience 600
            # hyper_space = {'n_hidden': [200], 'epochs': [1000], 'lr': [0.01],
            #                'classifier': [1], 'dropout': [0]}#g2p manuscrpit. Dec 10
            #                'patience': [1]}  # June.13 th. July 16. delete patience 600
            # hyper_space = {'n_hidden': [200], 'epochs': [10000], 'lr': [ 0.001,0.0005],
            #                'classifier': [1],'dropout':[0,0.2],'patience':[200,600]}#June.3rd.
            # hyper_space = {'n_hidden': [200, 300], 'epochs': [200], 'lr': [0.001, 0.0005],
            #                'classifier': [1],'dropout':[0,0.2,0.5]}
        # elif antibiotics=='all_possible_anti_concat' and anti_number>1:
        #     hyper_space = {'n_hidden': [200, 400], 'epochs': [30000], 'lr': [0.0005, 0.0001],
        #                    'classifier': [1, 2], 'dropout': [0, 0.2], 'patience': [200,600]}  # June.3rd.
        #     # hyper_space = {'n_hidden': [200, 300], 'epochs': [200], 'lr': [0.001, 0.0005],
        #     #                'classifier': [1, 2], 'dropout': [0, 0.2],'patience': [200]}
        else: #discrete multi-model and concat model for comparison.
            hyper_space = {'n_hidden': [200,400], 'epochs': [30000], 'lr': [0.0005,0.0001],
                           'classifier': [1,2],'dropout':[0,0.2],'patience':[2]}#June.12th. New. July 16. delete patience 600
            # hyper_space = {'n_hidden': [200, 400], 'epochs': [30000], 'lr': [0.0005, 0.0001],
            #                'classifier': [1, 2], 'dropout': [0, 0.2], 'patience': [200]}  # June.3rd.old
            # hyper_space = {'n_hidden': [200], 'epochs': [800], 'lr': [0.001],
            #                'classifier': [1],'dropout':[0,0.2],'patience': [200]}
            # if anti_number == 1:  June 3rd
            #     hyper_space = {'n_hidden': [200, 300], 'epochs': [20000], 'lr': [0.001, 0.0005, 0.0001],
            #                    'classifier': [1], 'dropout': [0, 0.2, 0.5], 'patience': [200, 600, 1000, 30000]}
            #     # hyper_space = {'n_hidden': [200, 300], 'epochs': [200], 'lr': [0.001, 0.0005],
            #     #                'classifier': [1],'dropout':[0,0.2,0.5]}
            # else:
            #     hyper_space = {'n_hidden': [200, 300, 400], 'epochs': [30000], 'lr': [0.001, 0.0005, 0.0001],
            #                    'classifier': [1, 2, 3], 'dropout': [0, 0.2, 0.5], 'patience': [200, 600, 1000, 30000]}
    return hyper_space
def hyper_range_concat(anti_number,f_no_early_stop,antibiotics):
    if f_no_early_stop==True:
        print('please do not use this option, because no patience is included in the hyper=para selection.')
    else:

        hyper_space = {'n_hidden': [200, 400], 'epochs': [30000], 'lr': [0.0005, 0.0001],
                       'classifier': [1, 2], 'dropout': [0, 0.2], 'patience': [200,600]}  # June.3rd.
        # hyper_space = {'n_hidden': [200], 'epochs': [200], 'lr': [0.001],
        #                'classifier': [1], 'dropout': [0, 0.2], 'patience': [200]}


    return hyper_space

# def eval(species, antibiotics, level, xdata, ydata, p_names, p_clusters, cv, Random_State, hidden, epochs,re_epochs,
#          learning,f_scaler,f_fixed_threshold,f_no_early_stop,f_optimize_score,save_name_score,concat_merge_name,threshold_point,min_cov_point):
def eval(species, antibiotics, level, xdata, ydata, p_names, p_clusters, cv, Random_State,
         re_epochs, f_scaler,f_fixed_threshold,f_no_early_stop,f_phylotree, f_random,f_optimize_score, save_name_score,f_learning,
         f_epochs,concat_merge_name,threshold_point,min_cov_point,feature):

    #data
    data_x = np.loadtxt(xdata, dtype="float")
    data_y =np.loadtxt(ydata)
    print('dataset shape',data_x.shape)
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
    if f_phylotree:#phylo-tree based cv folders
        folders_sample = neural_networks.cluster_folders.prepare_folders_tree(cv,species,antibiotics,p_names,False)
    elif f_random:
        folders_sample = neural_networks.cluster_folders.prepare_folders_random(cv,species,antibiotics,p_names,False)
    else:#kma cluster based cv folders
        folders_sample,_,_ = neural_networks.cluster_folders.prepare_folders(cv, Random_State, p_names, p_clusters,
                                                                               'new')

    hyper_space = hyper_range(anti_number, f_no_early_stop, antibiotics)
    for out_cv in range(cv):

        # remain= list(set(range(cv)) - set([out_cv])).sort()#first round: set(0,1,2,3,4)-set(0)=set(1,2,3,4)
        train_val_samples= folders_sample[:out_cv] + folders_sample[out_cv+1 :]#list
        Validation_mcc_thresholds = []  #  inner CV *11 thresholds value
        Validation_f1_thresholds = []  #  inner CV *11 thresholds value
        Validation_auc = []  # inner CV
        Validation_actul_epoc=[]
        # Validation_mcc = []  # len=inner CV
        # Validation_f1 = []  # len=inner CV
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
            print(len(val_samples))
            print(len(train_samples))
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
                print(list(ParameterGrid(hyper_space))[grid_iteration])
                # print('Hyper_parameters:',n_cl)
                # print(epochs,n_hidden,learning)
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
                #==================================================


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

                        # y_val = np.array(y_val)  # ground truth

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

                                # report= classification_report(y_val, y_val_pred,
                                #                            labels=[0, 1],
                                #                            output_dict=True)
                                # #todo check
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
            # x_test = torch.from_numpy(x_test).float()

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
        if feature =='res':#todo 1) need to change names for resfeature.
            name_weights = amr_utility.name_utility.GETname_multi_bench_weight(concat_merge_name,species, antibiotics,level,
                                                                           out_cv,'',f_learning,f_epochs,f_fixed_threshold,
                                                                           f_no_early_stop,f_phylotree,f_optimize_score,
                                                                           threshold_point,min_cov_point)
        else:#'kmer','s2g'. Nov 16.2021. so far only for g2p_Manu.
            name_weights = amr_utility.name_utility.g2pManu_weight(concat_merge_name,species, antibiotics,level,
                                                                           out_cv,'',f_learning,f_epochs,f_fixed_threshold,
                                                                           f_no_early_stop,f_phylotree,f_optimize_score,
                                                                           threshold_point,min_cov_point,feature)

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
                pred_test_sub_anti=pred_test_sub[comp]#June 21st, auc bugs

                mcc = matthews_corrcoef(y_test_anti[:, i], pred_test_binary_sub_anti[:, i])
                f1 = f1_score(y_test_anti[:, i], pred_test_binary_sub_anti[:, i], average='macro')
                fpr, tpr, _ = roc_curve(y_test_anti[:, i], pred_test_sub_anti[:, i], pos_label=1)#June 21st, auc bugs
                roc_auc = auc(fpr, tpr)
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
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

        # aucs_test_all.append(aucs)  ## multi-out
        tprs_test.append(tprs)  ## multi-out





    # plot(anti_number, mcc_test, cv, validation, pred_val_all, validation_y, tprs_test, aucs_test_all, mean_fpr)


    print('thresholds_selected_test',thresholds_selected_test)
    print('f1_test',f1_test)
    print('mcc_test',mcc_test)
    print('hyperparameters_test',hyperparameters_test)
    # score_summary(cv, score_report_test, aucs_test, mcc_test, save_name_score,thresholds_selected_test)#save mean and std of each 6 score
    score = [thresholds_selected_test, f1_test, mcc_test, score_report_test, aucs_test, tprs_test,hyperparameters_test,actual_epoc_test,actual_epoc_test_std]
    # if f_phylotree:
    #     with open(save_name_score + '_all_score_Tree.pickle', 'wb') as f:  # overwrite
    #         pickle.dump(score, f)
    # else: #todo 2) need to change names for res feature
    #     with open(save_name_score + '_all_score.pickle', 'wb') as f:  # overwrite
    #         pickle.dump(score, f)
    torch.cuda.empty_cache()
    return score


def multi_eval(species, antibiotics, level, xdata, ydata, p_names, p_clusters, cv, Random_State,
         re_epochs, f_scaler,f_fixed_threshold,f_no_early_stop, f_optimize_score, save_name_score,concat_merge_name,threshold_point,min_cov_point):
    '''Normal CV'''

    # data
    data_x = np.loadtxt(xdata, dtype="float")
    data_y = np.loadtxt(ydata)

    ##prepare data stores for testing scores##

    pred_test = []  # probabilities are for the validation set
    pred_test_binary = []  # binary based on selected
    thresholds_selected_test = []  # cv len, each is the selected threshold.
    weight_test = []  # cv len, each is the index of the selected model in inner CV,
    # only this weights are preseved in /log/temp/

    mcc_test = []  # MCC results for the test data
    f1_test = []
    score_report_test = []

    aucs_test = []  # all AUC values for the test data
    aucs_test_all = []  # multi-output and plotting used
    tprs_test = []  # all True Positives for the test data
    mean_fpr = np.linspace(0, 1, 100)
    hyperparameters_test = []#outCV_N, i.e. one-element list
    actual_epoc_test = []#outCV_N, i.e. one-element list. mean of the selected hyper-para among n_innerCV
    actual_epoc_test_std = []#outCV_N, i.e. one-element list. std of the selected hyper-para among n_innerCV
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
    # khu:nested CV.
    # =====================================
    # training
    # =====================================
    # dimension: cv*(sample_n in each split(it varies))
    # element: index of sampels w.r.t. data_x, data_y
    folders_sample, _, _ = neural_networks.cluster_folders.prepare_folders(cv, Random_State, p_names, p_clusters, 'new')

    hyper_space = hyper_range(anti_number, f_no_early_stop, antibiotics)


    # select testing folder
    out_cv=0 #always use the first folder as testing folder
    test_samples = folders_sample[out_cv]
    x_test = data_x[test_samples]
    y_test = data_y[test_samples]
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()
    x_test = x_test.to(device)
    # remain= list(set(range(cv)) - set([out_cv])).sort()#first round: set(0,1,2,3,4)-set(0)=set(1,2,3,4)
    train_val_samples = folders_sample[:out_cv] + folders_sample[out_cv + 1:]  # list
    Validation_mcc_thresholds = []  # inner CV *11 thresholds value
    Validation_f1_thresholds = []  # inner CV *11 thresholds value
    Validation_auc = []  # inner CV
    Validation_actul_epoc = []
    # Validation_mcc = []  # len=inner CV
    # Validation_f1 = []  # len=inner CV
    # later choose the inner loop and relevant thresholds with the highest f1 score

    # ---------------------------------------------Only for validation scores output
    # f1_val = []#only for validation score saving
    mcc_val = []  # only for validation score saving
    score_report_val = []  # only for validation score saving
    aucs_val = []  # only for validation score saving
    # tprs_val = []#only for validation score saving
    # --------------------------------------------



    for innerCV in range(cv - 1):  # e.g. 1,2,3,4
        print('Starting outer: ', str(out_cv), '; inner: ', str(innerCV), ' inner loop...')

        val_samples = train_val_samples[innerCV]
        train_samples = train_val_samples[:innerCV] + train_val_samples[innerCV + 1:]  # only works for list, not np
        train_samples = list(itertools.chain.from_iterable(train_samples))
        # training and val samples
        # select by order

        x_train, x_val = data_x[train_samples], data_x[val_samples]  # only np
        y_train, y_val = data_y[train_samples], data_y[val_samples]
        print('sample length and feature length for inner CV(train & val):', len(x_train), len(x_train[0]), len(x_val),
              len(x_val[0]))
        # vary for each CV

        # x_train = x_train.to(device)
        # y_train = y_train.to(device)

        # normalize the data
        if f_scaler == True:
            scaler = preprocessing.StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            # scale the val data based on the training data
            # scaler = preprocessing.StandardScaler().fit(x_train)
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
        # Validation_mcc_thresholds_split = []  # inner CV *11 thresholds value.June 21: grid_number
        Validation_f1_thresholds_split = []  # inner CV *11 thresholds value.June 21: grid_number
        Validation_auc_split = []  # inner CV
        Validation_actul_epoc_split = []
        # for validation scores output. Oct 21.2021
        score_report_val_split = []
        mcc_val_split = []
        aucs_val_split = []

        for grid_iteration in np.arange(len(list(ParameterGrid(hyper_space)))):

            # --------------------------------------------------------------------
            epochs = list(ParameterGrid(hyper_space))[grid_iteration]['epochs']
            n_hidden = list(ParameterGrid(hyper_space))[grid_iteration]['n_hidden']
            learning = list(ParameterGrid(hyper_space))[grid_iteration]['lr']
            n_cl = list(ParameterGrid(hyper_space))[grid_iteration]['classifier']
            dropout = list(ParameterGrid(hyper_space))[grid_iteration]['dropout']
            patience = list(ParameterGrid(hyper_space))[grid_iteration]['patience']
            # generate a NN model
            if n_cl == 1:
                classifier = _classifier(nlabel, D_input, n_hidden,
                                         dropout)  # reload, after testing(there is fine tune traning!)
            elif n_cl == 2:
                classifier = _classifier2(nlabel, D_input, n_hidden, dropout)
            elif n_cl == 3:
                classifier = _classifier3(nlabel, D_input, n_hidden, dropout)
            print(list(ParameterGrid(hyper_space))[grid_iteration])
            # print('Hyper_parameters:',n_cl)
            # print(epochs,n_hidden,learning)
            classifier.to(device)
            # generate the optimizer - Stochastic Gradient Descent
            optimizer = optim.SGD(classifier.parameters(), lr=learning)
            optimizer.zero_grad()  # Clears existing gradients from previous epoch
            # loop:
            print(species, antibiotics, level, out_cv, innerCV)
            start = time.time()
            if f_no_early_stop == True:
                classifier, actual_epoc = training_original(classifier, epochs, optimizer, x_train, y_train,
                                                            anti_number)
            else:
                classifier, actual_epoc = training(classifier, epochs, optimizer, x_train, y_train, x_val, y_val,
                                                   anti_number, patience)
            Validation_actul_epoc_split.append(actual_epoc)
            end = time.time()
            print('Time used: ', end - start)
            # ==================================================

            # if hyperparameter selectio mode, set learning and epoch as 0 for naming the output.
            # name_weights = amr_utility.name_utility.GETname_multi_bench_weight(concat_merge_name,species, antibiotics,level, out_cv, innerCV,0.0,0,f_fixed_threshold,f_no_early_stop,f_optimize_score,threshold_point,min_cov_point)
            # print(name_weights)
            # amr_utility.file_utility.make_dir(os.path.dirname(name_weights))#make folders for storing weights
            # torch.save(classifier.state_dict(), name_weights)

            ######################
            # 2. validate the model #
            ######################

            classifier.eval()  # eval mode
            pred_val_sub = []
            x_val = x_val.to(device)
            for v, v_sample in enumerate(x_val):

                # val = Variable(torch.FloatTensor(v_sample)).view(1, -1)
                val = Variable(v_sample).view(1, -1)
                output_test = classifier(val)
                out = output_test
                temp = []
                for h in out[0]:
                    temp.append(float(h))
                pred_val_sub.append(temp)

            pred_val_sub = np.array(pred_val_sub)  # for this innerCV at this out_cv
            # pred_val_inner.append(pred_val_sub)#for all innerCV at this out_cv. #No further use so far. April,30.
            print('pred_val_sub shape:', pred_val_sub.shape)  # (cv-1)*n_sample
            print('y_val', np.array(y_val).shape)

            # --------------------------------------------------
            # auc score, threshould operation is contained in itself definition.
            # khu add: 13May
            print('f_optimize_score:', f_optimize_score)



            if f_optimize_score == 'auc':
                # for validation scores output
                score_report_val_sub_anti = []
                mcc_val_sub_anti = []
                aucs_val_sub_anti = []
                # y_val=np.array(y_val)
                # aucs_val_sub_anti = []
                for i in range(anti_number):
                    comp = []
                    if anti_number == 1:#no use.
                        print('please check your antibiotic number')
                        exit()
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

                        y_val_anti = y_val[comp]
                        pred_val_sub_anti = pred_val_sub[comp]

                        fpr, tpr, _ = roc_curve(y_val_anti[:, i], pred_val_sub_anti[:, i], pos_label=1)

                        roc_auc = auc(fpr, tpr)
                        # tprs.append(interp(mean_fpr, fpr, tpr))
                        # tprs[-1][0] = 0.0
                        aucs_val_sub_anti.append(roc_auc)
                        # June 14th
                        # turn probability to binary.
                        threshold_matrix = np.full(pred_val_sub_anti.shape, 0.5)#use 0.5 as the threshold
                        pred_val_sub_anti_binary = (pred_val_sub_anti > threshold_matrix)
                        pred_val_sub_anti_binary = 1 * pred_val_sub_anti_binary

                        report = classification_report(y_val_anti[:, i], pred_val_sub_anti_binary[:, i], labels=[0, 1],
                                                       output_dict=True)
                        score_report_val_sub_anti.append(report)
                        mcc = matthews_corrcoef(y_val_anti[:, i], pred_val_sub_anti_binary[:, i])

                        mcc_val_sub_anti.append(mcc)


                if anti_number > 1:  # multi-out, scores based on mean of all the involved antibotics
                    aucs_val_sub_mean = statistics.mean(aucs_val_sub_anti)  # dimension: n_anti to 1
                    Validation_auc_split.append(aucs_val_sub_mean)  # D: n_innerCV
                    # June 14th
                    score_report_val_split.append(score_report_val_sub_anti)
                    aucs_val_split.append(aucs_val_sub_anti)
                    mcc_val_split.append(mcc_val_sub_anti)


            # ====================================================
            elif f_optimize_score == 'f1_macro':
                # Calculate macro f1. for thresholds from 0 to 1.
                mcc_sub = []
                f1_sub = []
                # for validation scores output.Oct 21.2021
                score_report_val_sub = []
                mcc_val_sub = []
                aucs_val_sub = []
                for threshold in np.arange(0, 1.1, 0.1):
                    # predictions for the test data
                    # turn probabilty to binary
                    threshold_matrix = np.full(pred_val_sub.shape, threshold)
                    y_val_pred = (pred_val_sub > threshold_matrix)
                    y_val_pred = 1 * y_val_pred

                    # y_val = np.array(y_val)  # ground truth

                    mcc_sub_anti = []
                    f1_sub_anti = []

                    # for validation scores output. Oct 21.2021
                    score_report_val_sub_anti = []
                    mcc_val_sub_anti = []
                    aucs_val_sub_anti = []

                    for i in range(anti_number):

                        comp = []  # becasue in the multi-species model, some species,anti combination are missing data
                        # so those won't be counted when evaluating scores.

                        if anti_number == 1:
                            print('please check your antibiotic number')
                            exit()
                            # mcc = matthews_corrcoef(y_val, y_val_pred)
                            # # report = classification_report(y_val, y_val_pred, labels=[0, 1], output_dict=True)
                            # f1 = f1_score(y_val, y_val_pred, average='macro')
                            # mcc_sub.append(mcc)
                            # f1_sub.append(f1)

                        else:  # multi-out
                            for t in range(len(y_val)):
                                if -1 != y_val[t][i]:
                                    comp.append(t)
                            y_val_multi_sub = y_val[comp]
                            y_val_pred_multi_sub = y_val_pred[comp]
                            mcc = matthews_corrcoef(y_val_multi_sub[:, i], y_val_pred_multi_sub[:, i])
                            f1 = f1_score(y_val_multi_sub[:, i], y_val_pred_multi_sub[:, i], average='macro')
                            # mcc_sub_anti.append(mcc)
                            f1_sub_anti.append(f1)
                            # June 16th
                            fpr, tpr, _ = roc_curve(y_val_multi_sub[:, i], y_val_pred_multi_sub[:, i], pos_label=1)
                            roc_auc = auc(fpr, tpr)
                            report = classification_report(y_val_multi_sub[:, i], y_val_pred_multi_sub[:, i],
                                                           labels=[0, 1],
                                                           output_dict=True)

                            # print(report)

                            mcc_val_sub_anti.append(mcc)
                            score_report_val_sub_anti.append(report)
                            aucs_val_sub_anti.append(roc_auc)
                    if anti_number > 1:  # multi-out, scores based on mean of all the involved antibotics
                        # mcc_sub.append(statistics.mean(mcc_sub_anti))  # mcc_sub_anti dimension: n_anti
                        f1_sub.append(statistics.mean(f1_sub_anti))
                        # --for validation scores output
                        score_report_val_sub.append(score_report_val_sub_anti)
                        aucs_val_sub.append(aucs_val_sub_anti)
                        mcc_val_sub.append(mcc_val_sub_anti)
                # Validation_mcc_thresholds_split.append(mcc_sub)
                Validation_f1_thresholds_split.append(f1_sub)
                # --for validation scores output
                score_report_val_split.append(score_report_val_sub)
                aucs_val_split.append(aucs_val_sub)
                mcc_val_split.append(mcc_val_sub)


                # print(Validation_f1_thresholds)
                # print(Validation_mcc_thresholds)


        # Validation_mcc_thresholds.append(Validation_mcc_thresholds_split)  # inner CV * hyperpara_combination
        Validation_f1_thresholds.append(Validation_f1_thresholds_split)  # inner CV * hyperpara_combination
        Validation_auc.append(Validation_auc_split)  # inner CV * hyperpara_combination
        Validation_actul_epoc.append(Validation_actul_epoc_split)  # inner CV * hyperpara_combination
        # -------------------------------------------------
        # --for validation scores output. June 21st. Oct 21
        score_report_val.append(score_report_val_split)
        aucs_val.append(aucs_val_split)
        mcc_val.append(mcc_val_split)

    # finish evaluation in the inner loop.
    if f_fixed_threshold == True and f_optimize_score == 'f1_macro':  # finished.May 30. 7am
        thresholds_selected = 0.5
        Validation_f1_thresholds = np.array(Validation_f1_thresholds)
        Validation_f1_thresholds = Validation_f1_thresholds.mean(axis=0)
        Validation_actul_epoc = np.array(Validation_actul_epoc)
        Validation_actul_epoc_std = Validation_actul_epoc.std(axis=0)
        Validation_actul_epoc = Validation_actul_epoc.mean(axis=0)

        # select the inner loop index with the highest f1 score in the column w.r.t. 0.5
        aim_column = np.where(np.arange(0, 1.1, 0.1) == 0.5)
        # aim_column = aim_column[0][0]
        aim_f1 = Validation_f1_thresholds[:, aim_column]
        weights_selected = np.argmax(aim_f1)
        print(Validation_f1_thresholds.shape)
        print(aim_f1)
        print(aim_f1.shape)
        print()
        ind = np.unravel_index(np.argmax(aim_f1, axis=None), aim_f1.shape)
        print('ind', ind)
        print(ind[0])
        print(len(score_report_val))
        hyperparameters_test.append(list(ParameterGrid(hyper_space))[ind[0]])
        actual_epoc_test.append(Validation_actul_epoc[ind[0]])
        actual_epoc_test_std.append(Validation_actul_epoc_std[ind[0]])

        # only for validation score saving.June 16th
        score_report_val = score_report_val[:,ind[0],aim_column[0][0],:]
        aucs_val = aucs_val[:,ind[0],aim_column[0][0],:]
        mcc_val = mcc_val[:,ind[0],aim_column[0][0],:]

    elif f_fixed_threshold == False and f_optimize_score == 'f1_macro':  # finished.May 30. 7am
        # select the inner loop index,and threshold with the highest f1 score in the matrix
        Validation_f1_thresholds = np.array(Validation_f1_thresholds)
        Validation_f1_thresholds = Validation_f1_thresholds.mean(axis=0)
        Validation_actul_epoc = np.array(Validation_actul_epoc)

        Validation_actul_epoc_std = Validation_actul_epoc.std(axis=0)
        Validation_actul_epoc = Validation_actul_epoc.mean(axis=0)
        ind = np.unravel_index(np.argmax(Validation_f1_thresholds, axis=None), Validation_f1_thresholds.shape)
        thresholds_selected = np.arange(0, 1.1, 0.1)[ind[1]]
        weights_selected = ind[0]  # the order of innerCV# bug ? seems no 13May.
        print(Validation_f1_thresholds.shape)#(16, 11)
        print('ind', ind)# (0,4)
        print(len(score_report_val))
        hyperparameters_test.append(list(ParameterGrid(hyper_space))[ind[0]])

        actual_epoc_test.append(Validation_actul_epoc[ind[0]])

        actual_epoc_test_std.append(Validation_actul_epoc_std[ind[0]])

        # only for validation score saving.June 16th
        score_report_val = score_report_val[:,ind[0],ind[1],:]#[hyperparameter][thresholds]
        aucs_val = aucs_val[:,ind[0],ind[1],:]
        mcc_val = mcc_val[:,ind[0],ind[1],:]

    elif f_optimize_score == 'auc':  # finished.May 30. 7am

        thresholds_selected = 0.5
        Validation_auc = np.array(Validation_auc)
        Validation_auc = Validation_auc.mean(axis=0)
        Validation_actul_epoc = np.array(Validation_actul_epoc)#innerCV*hyper_number
        Validation_actul_epoc_std = Validation_actul_epoc.std(axis=0)
        Validation_actul_epoc = Validation_actul_epoc.mean(axis=0)#hyper_number

        weights_selected = np.argmax(Validation_auc)
        ind = np.unravel_index(np.argmax(Validation_auc, axis=None), Validation_auc.shape)
        hyperparameters_test.append(list(ParameterGrid(hyper_space))[ind[0]])
        actual_epoc_test.append(Validation_actul_epoc[ind[0]])  # actually it's mean epoch for that hyperpara.
        actual_epoc_test_std.append(Validation_actul_epoc_std[ind[0]])
        # only for validation score saving
        score_report_val = score_report_val[:,ind[0]]
        aucs_val = aucs_val[:,ind[0]]
        mcc_val = mcc_val[:,ind[0]]

    print('hyper_space selected: ', list(ParameterGrid(hyper_space))[ind[0]])
    print('weights_selected', weights_selected)
    print('thresholds_selected', thresholds_selected)
    score_val = [score_report_val, aucs_val, mcc_val]
    weight_test.append(weights_selected)
    thresholds_selected_test.append(thresholds_selected)
    # name_weights = amr_utility.name_utility.GETname_multi_bench_weight(concat_merge_name,species, antibiotics, level,out_cv, weights_selected,0.0,0,f_fixed_threshold,f_no_early_stop,f_optimize_score,threshold_point,min_cov_point)
    # classifier.load_state_dict(torch.load(name_weights))

    # finish innner loop
    # =======================================================
    # 3. Re-train on both train and val data with the selected hyper-para

    print('Outer loop, testing.')
    train_val_samples = list(itertools.chain.from_iterable(train_val_samples))
    x_train_val = data_x[train_val_samples]  # only np
    y_train_val = data_y[train_val_samples]

    if f_scaler == True:
        scaler = preprocessing.StandardScaler().fit(x_train_val)
        x_train_val = scaler.transform(x_train_val)
        # scale the val data based on the training data
        # scaler = preprocessing.StandardScaler().fit(x_train_val)
        x_test = scaler.transform(x_test)

    x_train_val = torch.from_numpy(x_train_val).float()
    y_train_val = torch.from_numpy(y_train_val).float()

    # -
    # epochs_testing = hyperparameters_test[out_cv]['epochs']
    n_hidden = hyperparameters_test[out_cv]['n_hidden']
    learning = hyperparameters_test[out_cv]['lr']
    n_cl = hyperparameters_test[out_cv]['classifier']
    dropout = hyperparameters_test[out_cv]['dropout']
    # generate a NN model
    if n_cl == 1:
        classifier = _classifier(nlabel, D_input, n_hidden,
                                 dropout)  # reload, after testing(there is fine tune traning!)
    elif n_cl == 2:
        classifier = _classifier2(nlabel, D_input, n_hidden, dropout)
    elif n_cl == 3:
        classifier = _classifier3(nlabel, D_input, n_hidden, dropout)
    print('testing hyper-parameter: ', hyperparameters_test[out_cv])
    # print('Hyper_parameters:',n_cl)
    # print(epochs,n_hidden,learning)
    classifier.to(device)
    # generate the optimizer - Stochastic Gradient Descent
    optimizer = optim.SGD(classifier.parameters(), lr=learning)
    optimizer.zero_grad()  # Clears existing gradients from previous epoch
    classifier.train()
    # optimizer = optim.SGD(classifier.parameters(), lr=0.0001)
    # classifier = fine_tune_training(classifier, re_epochs, optimizer, x_train_val, y_train_val, anti_number)
    print('actual_epoc_test[out_cv]', actual_epoc_test[out_cv])
    classifier = fine_tune_training(classifier, int(actual_epoc_test[out_cv]), optimizer, x_train_val, y_train_val,
                                    anti_number)
    name_weights = amr_utility.name_utility.GETname_multi_bench_weight(concat_merge_name, species, antibiotics, level,
                                                                       out_cv, '', 0.0, 0, f_fixed_threshold,
                                                                       f_no_early_stop, False,f_optimize_score,
                                                                       threshold_point, min_cov_point)#todo , if tree-based need change.

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
    pred_test.append(pred_test_sub)  # len= y_test
    print('x_test', x_test.shape)
    print('pred_test_sub', pred_test_sub.shape)
    print('y_test', y_test.shape)
    # 4. get measurement scores on testing set.
    # Positive predictive value (PPV), Precision; Accuracy (ACC); f1-score here,
    # y: True positive rate (TPR), aka. sensitivity, hit rate, and recall,
    # X: False positive rate (FPR), aka. fall-out,

    # turn probability to binary.
    threshold_matrix = np.full(pred_test_sub.shape, thresholds_selected)
    pred_test_binary_sub = (pred_test_sub > threshold_matrix)
    pred_test_binary_sub = 1 * pred_test_binary_sub

    y_test = np.array(y_test)
    # pred_test_binary_sub = np.array(pred_test_binary_sub)
    pred_test_binary.append(pred_test_binary_sub)

    # print(y_test)
    # print(pred_test_binary_sub)
    f1_test_anti = []  # multi-out
    score_report_test_anti = []  # multi-out
    tprs = []  # multi-out
    aucs_test_anti = []  # multi-out
    mcc_test_anti = []  # multi-out

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
            pred_test_sub_anti = pred_test_sub[comp]  # June 21st, auc bugs

            mcc = matthews_corrcoef(y_test_anti[:, i], pred_test_binary_sub_anti[:, i])
            f1 = f1_score(y_test_anti[:, i], pred_test_binary_sub_anti[:, i], average='macro')
            fpr, tpr, _ = roc_curve(y_test_anti[:, i], pred_test_sub_anti[:, i], pos_label=1)# June 21st, auc bugs
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

    # summerise each outer loop's results.
    if anti_number > 1:
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


    print('thresholds_selected_test', thresholds_selected_test)
    print('f1_test', f1_test)
    print('mcc_test', mcc_test)
    print('hyperparameters_test', hyperparameters_test)
    # score_summary(cv, score_report_test, aucs_test, mcc_test, save_name_score,thresholds_selected_test)#save mean and std of each 6 score
    score = [thresholds_selected_test, f1_test, mcc_test, score_report_test, aucs_test, tprs_test, hyperparameters_test,
             actual_epoc_test, actual_epoc_test_std,score_val]
    with open(save_name_score + '_all_score.pickle', 'wb') as f:  # overwrite
        pickle.dump(score, f)

    torch.cuda.empty_cache()
    return score


def concat_eval(species, antibiotics, level, xdata, ydata, p_names, p_clusters,path_x_test, path_y_test, cv, Random_State,
         re_epochs, f_scaler,f_fixed_threshold,f_no_early_stop, f_optimize_score, save_name_score,concat_merge_name,threshold_point,min_cov_point):
    '''Normal CV'''

    # data
    data_x = np.loadtxt(xdata, dtype="float")
    data_y = np.loadtxt(ydata)

    ##prepare data stores for testing scores##

    pred_test = []  # probabilities are for the validation set
    pred_test_binary = []  # binary based on selected
    thresholds_selected_test = []  # cv len, each is the selected threshold.
    weight_test = []  # cv len, each is the index of the selected model in inner CV,
    # only this weights are preseved in /log/temp/

    mcc_test = []  # MCC results for the test data
    f1_test = []
    score_report_test = []

    aucs_test = []  # all AUC values for the test data
    aucs_test_all = []  # multi-output and plotting used
    tprs_test = []  # all True Positives for the test data
    mean_fpr = np.linspace(0, 1, 100)
    hyperparameters_test = []
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



    x_test = np.loadtxt(path_x_test, dtype="float")
    y_test = np.loadtxt(path_y_test)
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()
    x_test = x_test.to(device)

    # =====================================
    # training
    # =====================================
    # dimension: cv*(sample_n in each split(it varies))
    # element: index of sampels w.r.t. data_x, data_y
    folders_sample, _, _ = neural_networks.cluster_folders.prepare_folders(cv, Random_State, p_names, p_clusters, 'new')

    hyper_space = hyper_range(anti_number, f_no_early_stop, antibiotics)


    # select testing folder
    out_cv=0 #always use the first folder as testing folder


    train_val_samples = folders_sample


    Validation_mcc_thresholds = []  # inner CV *11 thresholds value
    Validation_f1_thresholds = []  # inner CV *11 thresholds value
    Validation_auc = []  # inner CV
    Validation_actul_epoc = []
    # Validation_mcc = []  # len=inner CV
    # Validation_f1 = []  # len=inner CV
    # later choose the inner loop and relevant thresholds with the highest f1 score

    # ---------------------------------------------Only for validation scores output
    # f1_val = []#only for validation score saving
    mcc_val = []  # only for validation score saving
    score_report_val = []  # only for validation score saving
    aucs_val = []  # only for validation score saving
    # tprs_val = []#only for validation score saving
    # --------------------------------------------



    for innerCV in range(cv - 1):  # e.g. 1,2,3,4
        print('Starting outer: ', str(out_cv), '; inner: ', str(innerCV), ' inner loop...')

        val_samples = train_val_samples[innerCV]
        train_samples = train_val_samples[:innerCV] + train_val_samples[innerCV + 1:]  # only works for list, not np
        train_samples = list(itertools.chain.from_iterable(train_samples))
        # training and val samples
        # select by order

        x_train, x_val = data_x[train_samples], data_x[val_samples]  # only np
        y_train, y_val = data_y[train_samples], data_y[val_samples]
        print('sample length and feature length for inner CV(train & val):', len(x_train), len(x_train[0]), len(x_val),
              len(x_val[0]))
        # vary for each CV

        # x_train = x_train.to(device)
        # y_train = y_train.to(device)

        # normalize the data
        if f_scaler == True:
            scaler = preprocessing.StandardScaler().fit(x_train)
            x_train = scaler.transform(x_train)
            # scale the val data based on the training data
            # scaler = preprocessing.StandardScaler().fit(x_train)
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
        Validation_mcc_thresholds_split = []  # inner CV *11 thresholds value.June 21: grid_number
        Validation_f1_thresholds_split = []  # inner CV *11 thresholds value.June 21: grid_number
        Validation_auc_split = []  # inner CV
        Validation_actul_epoc_split = []
        # only for validation score saving
        score_report_val_sub = []
        aucs_val_sub = []
        mcc_val_sub = []
        for grid_iteration in np.arange(len(list(ParameterGrid(hyper_space)))):

            # --------------------------------------------------------------------
            epochs = list(ParameterGrid(hyper_space))[grid_iteration]['epochs']
            n_hidden = list(ParameterGrid(hyper_space))[grid_iteration]['n_hidden']
            learning = list(ParameterGrid(hyper_space))[grid_iteration]['lr']
            n_cl = list(ParameterGrid(hyper_space))[grid_iteration]['classifier']
            dropout = list(ParameterGrid(hyper_space))[grid_iteration]['dropout']
            patience = list(ParameterGrid(hyper_space))[grid_iteration]['patience']
            # generate a NN model
            if n_cl == 1:
                classifier = _classifier(nlabel, D_input, n_hidden,
                                         dropout)  # reload, after testing(there is fine tune traning!)
            elif n_cl == 2:
                classifier = _classifier2(nlabel, D_input, n_hidden, dropout)
            elif n_cl == 3:
                classifier = _classifier3(nlabel, D_input, n_hidden, dropout)
            print(list(ParameterGrid(hyper_space))[grid_iteration])
            # print('Hyper_parameters:',n_cl)
            # print(epochs,n_hidden,learning)
            classifier.to(device)
            # generate the optimizer - Stochastic Gradient Descent
            optimizer = optim.SGD(classifier.parameters(), lr=learning)
            optimizer.zero_grad()  # Clears existing gradients from previous epoch
            # loop:
            print(species, antibiotics, level, out_cv, innerCV)
            start = time.time()
            if f_no_early_stop == True:
                classifier, actual_epoc = training_original(classifier, epochs, optimizer, x_train, y_train,
                                                            anti_number)
            else:
                classifier, actual_epoc = training(classifier, epochs, optimizer, x_train, y_train, x_val, y_val,
                                                   anti_number, patience)
            Validation_actul_epoc_split.append(actual_epoc)
            end = time.time()
            print('Time used: ', end - start)
            # ==================================================

            # if hyperparameter selectio mode, set learning and epoch as 0 for naming the output.
            # name_weights = amr_utility.name_utility.GETname_multi_bench_weight(concat_merge_name,species, antibiotics,level, out_cv, innerCV,0.0,0,f_fixed_threshold,f_no_early_stop,f_optimize_score,threshold_point,min_cov_point)
            # print(name_weights)
            # amr_utility.file_utility.make_dir(os.path.dirname(name_weights))#make folders for storing weights
            # torch.save(classifier.state_dict(), name_weights)

            ######################
            # 2. validate the model #
            ######################

            classifier.eval()  # eval mode
            pred_val_sub = []
            x_val = x_val.to(device)
            for v, v_sample in enumerate(x_val):

                # val = Variable(torch.FloatTensor(v_sample)).view(1, -1)
                val = Variable(v_sample).view(1, -1)
                output_test = classifier(val)
                out = output_test
                temp = []
                for h in out[0]:
                    temp.append(float(h))
                pred_val_sub.append(temp)

            pred_val_sub = np.array(pred_val_sub)  # for this innerCV at this out_cv
            # pred_val_inner.append(pred_val_sub)#for all innerCV at this out_cv. #No further use so far. April,30.
            print('pred_val_sub shape:', pred_val_sub.shape)  # (cv-1)*n_sample
            print('y_val', np.array(y_val).shape)

            # --------------------------------------------------
            # auc score, threshould operation is contained in itself definition.
            # khu add: 13May
            print('f_optimize_score:', f_optimize_score)
            # for validation scores output
            score_report_val_sub_anti = []
            mcc_val_sub_anti = []
            aucs_val_sub_anti = []


            if f_optimize_score == 'auc':
                # y_val=np.array(y_val)
                # aucs_val_sub_anti = []
                for i in range(anti_number):
                    comp = []
                    if anti_number == 1:#no use.
                        print('please check your antibiotic number')
                        exit()
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

                        y_val_anti = y_val[comp]
                        pred_val_sub_anti = pred_val_sub[comp]

                        fpr, tpr, _ = roc_curve(y_val_anti[:, i], pred_val_sub_anti[:, i], pos_label=1)

                        roc_auc = auc(fpr, tpr)
                        # tprs.append(interp(mean_fpr, fpr, tpr))
                        # tprs[-1][0] = 0.0
                        aucs_val_sub_anti.append(roc_auc)
                        # June 14th
                        # turn probability to binary.
                        threshold_matrix = np.full(pred_val_sub_anti.shape, 0.5)#use 0.5 as the threshold
                        pred_val_sub_anti_binary = (pred_val_sub_anti > threshold_matrix)
                        pred_val_sub_anti_binary = 1 * pred_val_sub_anti_binary

                        report = classification_report(y_val_anti[:, i], pred_val_sub_anti_binary[:, i], labels=[0, 1],
                                                       output_dict=True)
                        score_report_val_sub_anti.append(report)
                        mcc = matthews_corrcoef(y_val_anti[:, i], pred_val_sub_anti_binary[:, i])

                        mcc_val_sub_anti.append(mcc)


                if anti_number > 1:  # multi-out, scores based on mean of all the involved antibotics
                    aucs_val_sub_mean = statistics.mean(aucs_val_sub_anti)  # dimension: n_anti to 1
                    Validation_auc_split.append(aucs_val_sub_mean)  # D: n_innerCV
                    # June 14th
                    score_report_val_sub.append(score_report_val_sub_anti)
                    aucs_val_sub.append(aucs_val_sub_anti)
                    mcc_val_sub.append(mcc_val_sub_anti)

            # ====================================================
            elif f_optimize_score == 'f1_macro':
                # Calculate macro f1. for thresholds from 0 to 1.
                mcc_sub = []
                f1_sub = []

                for threshold in np.arange(0, 1.1, 0.1):
                    # predictions for the test data
                    # turn probabilty to binary
                    threshold_matrix = np.full(pred_val_sub.shape, threshold)
                    y_val_pred = (pred_val_sub > threshold_matrix)
                    y_val_pred = 1 * y_val_pred

                    # y_val = np.array(y_val)  # ground truth

                    mcc_sub_anti = []
                    f1_sub_anti = []
                    for i in range(anti_number):

                        comp = []  # becasue in the multi-species model, some species,anti combination are missing data
                        # so those won't be counted when evaluating scores.

                        if anti_number == 1:
                            print('please check your antibiotic number')
                            exit()
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
                            # June 16th
                            fpr, tpr, _ = roc_curve(y_val_multi_sub[:, i], y_val_pred_multi_sub[:, i], pos_label=1)
                            roc_auc = auc(fpr, tpr)
                            report = classification_report(y_val_multi_sub[:, i], y_val_pred_multi_sub[:, i],
                                                           labels=[0, 1],
                                                           output_dict=True)
                            mcc_val_sub_anti.append(mcc)
                            score_report_val_sub_anti.append(report)
                            aucs_val_sub_anti.append(roc_auc)
                    if anti_number > 1:  # multi-out, scores based on mean of all the involved antibotics
                        mcc_sub.append(statistics.mean(mcc_sub_anti))  # mcc_sub_anti dimension: n_anti
                        f1_sub.append(statistics.mean(f1_sub_anti))
                        # --for validation scores output
                        score_report_val_sub.append(score_report_val_sub_anti)
                        aucs_val_sub.append(aucs_val_sub_anti)
                        mcc_val_sub.append(mcc_val_sub_anti)
                #finish one grid combination.
                Validation_mcc_thresholds_split.append(mcc_sub)
                Validation_f1_thresholds_split.append(f1_sub)
                # print(Validation_f1_thresholds)
                # print(Validation_mcc_thresholds)



        #finish grid search in one inner loop
        Validation_mcc_thresholds.append(Validation_mcc_thresholds_split)  # inner CV * hyperpara_combination
        Validation_f1_thresholds.append(Validation_f1_thresholds_split)  # inner CV * hyperpara_combination
        Validation_auc.append(Validation_auc_split)  # inner CV * hyperpara_combination
        Validation_actul_epoc.append(Validation_actul_epoc_split)  # inner CV * hyperpara_combination
        # --for validation scores output.June 21st.
        score_report_val.append(score_report_val_sub)
        aucs_val.append(aucs_val_sub)
        mcc_val.append(mcc_val_sub)



    # finish evaluation in the inner loop.
    if f_fixed_threshold == True and f_optimize_score == 'f1_macro':  # finished.May 30. 7am
        thresholds_selected = 0.5
        Validation_f1_thresholds = np.array(Validation_f1_thresholds)
        Validation_f1_thresholds = Validation_f1_thresholds.mean(axis=0)
        Validation_actul_epoc = np.array(Validation_actul_epoc)
        Validation_actul_epoc_std = Validation_actul_epoc.std(axis=0)
        Validation_actul_epoc = Validation_actul_epoc.mean(axis=0)

        # select the inner loop index with the highest f1 score in the column w.r.t. 0.5
        aim_column = np.where(np.arange(0, 1.1, 0.1) == 0.5)
        # aim_column = aim_column[0][0]
        aim_f1 = Validation_f1_thresholds[:, aim_column]
        weights_selected = np.argmax(aim_f1)
        print(Validation_f1_thresholds.shape)
        print(aim_f1)
        print(aim_f1.shape)
        print()
        ind = np.unravel_index(np.argmax(aim_f1, axis=None), aim_f1.shape)
        print('ind', ind)
        print(ind[0])
        print()
        hyperparameters_test.append(list(ParameterGrid(hyper_space))[ind[0]])
        actual_epoc_test.append(Validation_actul_epoc[ind[0]])
        actual_epoc_test_std.append(Validation_actul_epoc_std[ind[0]])

        # only for validation score saving.June 16th
        score_report_val = score_report_val[ind[0]][aim_column[0][0]]
        aucs_val = aucs_val[ind[0]][aim_column[0][0]]
        mcc_val = mcc_val[ind[0]][aim_column[0][0]]

    elif f_fixed_threshold == False and f_optimize_score == 'f1_macro':  # finished.May 30. 7am
        # select the inner loop index,and threshold with the highest f1 score in the matrix
        Validation_f1_thresholds = np.array(Validation_f1_thresholds)
        Validation_f1_thresholds = Validation_f1_thresholds.mean(axis=0)
        Validation_actul_epoc = np.array(Validation_actul_epoc)

        Validation_actul_epoc_std = Validation_actul_epoc.std(axis=0)
        Validation_actul_epoc = Validation_actul_epoc.mean(axis=0)
        ind = np.unravel_index(np.argmax(Validation_f1_thresholds, axis=None), Validation_f1_thresholds.shape)
        thresholds_selected = np.arange(0, 1.1, 0.1)[ind[1]]
        weights_selected = ind[0]  # the order of innerCV# bug ? seems no 13May.
        print(Validation_f1_thresholds.shape)
        print('ind', ind)
        hyperparameters_test.append(list(ParameterGrid(hyper_space))[ind[0]])

        actual_epoc_test.append(Validation_actul_epoc[ind[0]])

        actual_epoc_test_std.append(Validation_actul_epoc_std[ind[0]])

        # only for validation score saving.June 16th
        score_report_val = score_report_val[ind[0]][ind[1]]#[hyperparameter][thresholds]
        aucs_val = aucs_val[ind[0]][ind[1]]
        mcc_val = mcc_val[ind[0]][ind[1]]

    elif f_optimize_score == 'auc':  # finished.May 30. 7am

        thresholds_selected = 0.5
        Validation_auc = np.array(Validation_auc)
        Validation_auc = Validation_auc.mean(axis=0)
        Validation_actul_epoc = np.array(Validation_actul_epoc)#innerCV*hyper_number
        Validation_actul_epoc_std = Validation_actul_epoc.std(axis=0)
        Validation_actul_epoc = Validation_actul_epoc.mean(axis=0)#hyper_number

        weights_selected = np.argmax(Validation_auc)
        ind = np.unravel_index(np.argmax(Validation_auc, axis=None), Validation_auc.shape)
        hyperparameters_test.append(list(ParameterGrid(hyper_space))[ind[0]])
        actual_epoc_test.append(Validation_actul_epoc[ind[0]])  # actually it's mean epoch for that hyperpara.
        actual_epoc_test_std.append(Validation_actul_epoc_std[ind[0]])
        # only for validation score saving
        score_report_val = score_report_val[ind[0]]
        aucs_val = aucs_val[ind[0]]
        mcc_val = mcc_val[ind[0]]

    print('hyper_space selected: ', list(ParameterGrid(hyper_space))[ind[0]])
    print('weights_selected', weights_selected)
    print('thresholds_selected', thresholds_selected)
    score_val = [score_report_val, aucs_val, mcc_val]
    weight_test.append(weights_selected)
    thresholds_selected_test.append(thresholds_selected)
    # name_weights = amr_utility.name_utility.GETname_multi_bench_weight(concat_merge_name,species, antibiotics, level,out_cv, weights_selected,0.0,0,f_fixed_threshold,f_no_early_stop,f_optimize_score,threshold_point,min_cov_point)
    # classifier.load_state_dict(torch.load(name_weights))

    # finish innner loop
    # =======================================================
    # 3. Re-train on both train and val data with the selected hyper-para

    print('Outer loop, testing.')
    train_val_samples = list(itertools.chain.from_iterable(train_val_samples))
    x_train_val = data_x[train_val_samples]  # only np
    y_train_val = data_y[train_val_samples]

    if f_scaler == True:
        scaler = preprocessing.StandardScaler().fit(x_train_val)
        x_train_val = scaler.transform(x_train_val)
        # scale the val data based on the training data
        # scaler = preprocessing.StandardScaler().fit(x_train_val)
        x_test = scaler.transform(x_test)

    x_train_val = torch.from_numpy(x_train_val).float()
    y_train_val = torch.from_numpy(y_train_val).float()

    # -
    # epochs_testing = hyperparameters_test[out_cv]['epochs']
    n_hidden = hyperparameters_test[out_cv]['n_hidden']
    learning = hyperparameters_test[out_cv]['lr']
    n_cl = hyperparameters_test[out_cv]['classifier']
    dropout = hyperparameters_test[out_cv]['dropout']
    # generate a NN model
    if n_cl == 1:
        classifier = _classifier(nlabel, D_input, n_hidden,
                                 dropout)  # reload, after testing(there is fine tune traning!)
    elif n_cl == 2:
        classifier = _classifier2(nlabel, D_input, n_hidden, dropout)
    elif n_cl == 3:
        classifier = _classifier3(nlabel, D_input, n_hidden, dropout)
    print('testing hyper-parameter: ', hyperparameters_test[out_cv])
    # print('Hyper_parameters:',n_cl)
    # print(epochs,n_hidden,learning)
    classifier.to(device)
    # generate the optimizer - Stochastic Gradient Descent
    optimizer = optim.SGD(classifier.parameters(), lr=learning)
    optimizer.zero_grad()  # Clears existing gradients from previous epoch
    classifier.train()
    # optimizer = optim.SGD(classifier.parameters(), lr=0.0001)
    # classifier = fine_tune_training(classifier, re_epochs, optimizer, x_train_val, y_train_val, anti_number)
    print('actual_epoc_test[out_cv]', actual_epoc_test[out_cv])
    classifier = fine_tune_training(classifier, int(actual_epoc_test[out_cv]), optimizer, x_train_val, y_train_val,
                                    anti_number)
    name_weights = amr_utility.name_utility.GETname_multi_bench_weight(concat_merge_name, species, antibiotics, level,
                                                                       out_cv, '', 0.0, 0, f_fixed_threshold,
                                                                       f_no_early_stop, False,f_optimize_score,
                                                                       threshold_point, min_cov_point)#todo, if tree based need change.

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
    pred_test.append(pred_test_sub)  # len= y_test
    print('x_test', x_test.shape)
    print('pred_test_sub', pred_test_sub.shape)
    print('y_test', y_test.shape)
    # 4. get measurement scores on testing set.
    # Positive predictive value (PPV), Precision; Accuracy (ACC); f1-score here,
    # y: True positive rate (TPR), aka. sensitivity, hit rate, and recall,
    # X: False positive rate (FPR), aka. fall-out,

    # turn probability to binary.
    threshold_matrix = np.full(pred_test_sub.shape, thresholds_selected)
    pred_test_binary_sub = (pred_test_sub > threshold_matrix)
    pred_test_binary_sub = 1 * pred_test_binary_sub

    y_test = np.array(y_test)
    # pred_test_binary_sub = np.array(pred_test_binary_sub)
    pred_test_binary.append(pred_test_binary_sub)

    # print(y_test)
    # print(pred_test_binary_sub)
    f1_test_anti = []  # multi-out
    score_report_test_anti = []  # multi-out
    tprs = []  # multi-out
    aucs_test_anti = []  # multi-out
    mcc_test_anti = []  # multi-out

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
            pred_test_sub_anti = pred_test_sub[comp]  # June 21st, auc bugs
            if comp != []:
                mcc = matthews_corrcoef(y_test_anti[:, i], pred_test_binary_sub_anti[:, i])
                f1 = f1_score(y_test_anti[:, i], pred_test_binary_sub_anti[:, i], average='macro')
                fpr, tpr, _ = roc_curve(y_test_anti[:, i], pred_test_sub_anti[:, i], pos_label=1)# June 21st, auc bugs
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

    # summerise each outer loop's results.
    if anti_number > 1:
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


    print('thresholds_selected_test', thresholds_selected_test)
    print('f1_test', f1_test)
    print('mcc_test', mcc_test)
    print('hyperparameters_test', hyperparameters_test)
    # score_summary(cv, score_report_test, aucs_test, mcc_test, save_name_score,thresholds_selected_test)#save mean and std of each 6 score
    score = [thresholds_selected_test, f1_test, mcc_test, score_report_test, aucs_test, tprs_test, hyperparameters_test,
             actual_epoc_test, actual_epoc_test_std,score_val]
    # with open(save_name_score + '_all_score.pickle', 'wb') as f:  # overwrite
    #     pickle.dump(score, f)

    torch.cuda.empty_cache()
    return score

# def concat_eval(species, antibiotics, level, path_x_train, path_y_train,
#                                     path_name_train,  path_x_val, path_y_val,
#                                     path_name_val, path_x_test, path_y_test,
#                                     path_name_test,
#                                     Random_State, f_scaler, f_fixed_threshold,f_no_early_stop,
#                                     f_optimize_score, save_name_score, concat_merge_name, threshold_point,
#                                     min_cov_point):
#     '''
#     concatenated multi-species model.
#     Evaluation on selected species via fastANI, training on rest species
#     '''
#     # data
#     data_x_train = np.loadtxt(path_x_train, dtype="float")
#     data_y_train = np.loadtxt(path_y_train)
#     data_x_val = np.loadtxt(path_x_val, dtype="float")
#     data_y_val =np.loadtxt(path_y_val)
#
#
#     # -------------------------------------------------------------------------------------------------------------------
#     # -------------------------------------------For hyper-para selection
#     Validation_mcc_thresholds = []  # N_gridsearch*11 thresholds value
#     Validation_f1_thresholds = []  # N_gridsearch*11 thresholds value
#     Validation_auc = []  # N_gridsearch
#     Validation_actul_epoc = [] #N_gridsearch
#
#     #---------------------------------------------Only for validation scores output
#     # f1_val = []#only for validation score saving
#     mcc_val = []#only for validation score saving
#     score_report_val = []#only for validation score saving
#     aucs_val = []#only for validation score saving
#     # tprs_val = []#only for validation score saving
#     # --------------------------------------------
#     # ------------------------------------------------------------------------------------
#     anti_number = data_y_train[0].size  # number of antibiotics
#     nlabel = data_y_train[0].size  # number of neurons in the output layer
#
#     D_input = len(data_x_train[0])  # number of neurons in the input layer
#     N_sample = len(data_x_train)  # number of input samples #should be equal to len(names)
#
#
#
#     data_x_train = torch.from_numpy(data_x_train).float()
#     data_y_train = torch.from_numpy(data_y_train).float()
#     data_x_val = torch.from_numpy(data_x_val).float()
#     data_y_val = torch.from_numpy(data_y_val).float()
#     data_x_train=data_x_train.to(device)
#     data_x_val = data_x_val.to(device)
#     hyper_space=hyper_range_concat(anti_number, False, antibiotics)
#
#     for grid_iteration in np.arange(len(list(ParameterGrid(hyper_space)))):
#         # =====================================
#         # 1. training
#         # =====================================
#         # --------------------------------------------------------------------
#         epochs = list(ParameterGrid(hyper_space))[grid_iteration]['epochs']
#         n_hidden = list(ParameterGrid(hyper_space))[grid_iteration]['n_hidden']
#         learning = list(ParameterGrid(hyper_space))[grid_iteration]['lr']
#         n_cl = list(ParameterGrid(hyper_space))[grid_iteration]['classifier']
#         dropout = list(ParameterGrid(hyper_space))[grid_iteration]['dropout']
#         patience = list(ParameterGrid(hyper_space))[grid_iteration]['patience']
#         # generate a NN model
#         if n_cl == 1:
#             classifier = _classifier(nlabel, D_input, n_hidden,
#                                      dropout)  # reload, after testing(there is fine tune traning!)
#         elif n_cl == 2:
#             classifier = _classifier2(nlabel, D_input, n_hidden, dropout)
#         elif n_cl == 3:
#             classifier = _classifier3(nlabel, D_input, n_hidden, dropout)
#         print(list(ParameterGrid(hyper_space))[grid_iteration])
#         # print('Hyper_parameters:',n_cl)
#         # print(epochs,n_hidden,learning)
#         classifier.to(device)
#         # generate the optimizer - Stochastic Gradient Descent
#         optimizer = optim.SGD(classifier.parameters(), lr=learning)
#         optimizer.zero_grad()  # Clears existing gradients from previous epoch
#         # loop:
#         print(species, antibiotics, level)
#         start = time.time()
#         if f_no_early_stop == True:
#             classifier, actual_epoc = training_original(classifier, epochs, optimizer, data_x_train, data_y_train, anti_number)
#         else:
#             classifier, actual_epoc = training(classifier, epochs, optimizer, data_x_train, data_y_train, data_x_val, data_y_val,
#                                                anti_number, patience)
#         Validation_actul_epoc.append(actual_epoc)
#         end = time.time()
#         print('Time used: ', end - start)
#         # ==================================================
#         #2. evaluate
#         # ==================================================
#         classifier.eval()  # eval mode
#         pred_val_sub = []
#         # data_x_val = data_x_val.to(device)
#         for v, v_sample in enumerate(data_x_val):
#
#             # val = Variable(torch.FloatTensor(v_sample)).view(1, -1)
#             val = Variable(v_sample).view(1, -1)
#             output_test = classifier(val)
#             out = output_test
#             temp = []
#             for h in out[0]:
#                 temp.append(float(h))
#             pred_val_sub.append(temp)
#
#         pred_val_sub = np.array(pred_val_sub)  # for this innerCV at this out_cv
#         # pred_val_inner.append(pred_val_sub)#for all innerCV at this out_cv. #No further use so far. April,30.
#         print('pred_val_sub shape:', pred_val_sub.shape)  # (cv-1)*n_sample
#         print('y_val', np.array(data_y_val).shape)
#
#         # --------------------------------------------------
#         # auc score, threshould operation is contained in itself definition.
#         # khu add: 13May
#         print('f_optimize_score:', f_optimize_score)
#         # for validation scores output
#         score_report_val_sub_anti = []
#         mcc_val_sub_anti = []
#         aucs_val_sub_anti = []
#         if f_optimize_score == 'auc':
#             # y_val=np.array(y_val)
#
#             for i in range(anti_number):
#                 comp = []
#                 if anti_number == 1:# Not in use for cacatenated case.
#                     fpr, tpr, _ = roc_curve(data_y_val, pred_val_sub, pos_label=1)
#                     # tprs.append(interp(mean_fpr, fpr, tpr))
#                     # tprs[-1][0] = 0.0
#                     roc_auc = auc(fpr, tpr)
#                     Validation_auc.append(roc_auc)
#                 else:  # multi-out
#                     # todo check
#                     for t in range(len(data_y_val)):
#                         if -1 != data_y_val[t][i]:
#                             comp.append(t)
#
#                     y_val_anti = data_y_val[comp]
#                     pred_val_sub_anti = pred_val_sub[comp]
#                     fpr, tpr, _ = roc_curve(y_val_anti[:, i], pred_val_sub_anti[:, i], pos_label=1)
#                     roc_auc = auc(fpr, tpr)
#                     # tprs.append(interp(mean_fpr, fpr, tpr))
#                     # tprs[-1][0] = 0.0
#                     aucs_val_sub_anti.append(roc_auc)
#                     #June 14th
#                     report = classification_report(y_val_anti[:, i], pred_val_sub_anti[:, i], labels=[0, 1],
#                                                    output_dict=True)
#                     score_report_val_sub_anti.apend(report)
#                     mcc = matthews_corrcoef(y_val_anti[:, i], pred_val_sub_anti[:, i])
#
#                     mcc_val_sub_anti.append(mcc)
#             if anti_number > 1:  # multi-out, scores based on mean of all the involved antibotics
#                 aucs_val_sub = statistics.mean(aucs_val_sub_anti)  # dimension: n_anti to 1
#                 Validation_auc.append(aucs_val_sub)  # D: n_innerCV
#                 # June 14th
#                 score_report_val.append(score_report_val_sub_anti)
#                 aucs_val.append(aucs_val_sub_anti)
#                 mcc_val.append(mcc_val_sub_anti)
#         # ====================================================
#         elif f_optimize_score == 'f1_macro':
#             # Calculate macro f1. for thresholds from 0 to 1.
#             mcc_sub = []
#             f1_sub = []
#             #only for validation score saving
#             score_report_val_sub=[]
#             aucs_val_sub=[]
#             mcc_val_sub=[]
#             for threshold in np.arange(0, 1.1, 0.1):
#                 # predictions for the test data
#                 # turn probabilty to binary
#                 threshold_matrix = np.full(pred_val_sub.shape, threshold)
#                 y_val_pred = (pred_val_sub > threshold_matrix)
#                 y_val_pred = 1 * y_val_pred
#
#                 # y_val = np.array(y_val)  # ground truth
#
#                 mcc_sub_anti = []
#                 f1_sub_anti = []
#                 for i in range(anti_number):
#
#                     comp = []  # becasue in the multi-species model, some species,anti combination are missing data
#                     # so those won't be counted when evaluating scores.
#
#                     if anti_number == 1:
#                         mcc = matthews_corrcoef(data_y_val, y_val_pred)
#                         # report = classification_report(y_val, y_val_pred, labels=[0, 1], output_dict=True)
#                         f1 = f1_score(data_y_val, y_val_pred, average='macro')
#                         mcc_sub.append(mcc)
#                         f1_sub.append(f1)
#
#                     else:  # multi-out
#                         for t in range(len(data_y_val)):
#                             if -1 != data_y_val[t][i]:
#                                 comp.append(t)
#                         y_val_multi_sub = data_y_val[comp]
#                         y_val_pred_multi_sub = y_val_pred[comp]
#                         mcc = matthews_corrcoef(y_val_multi_sub[:, i], y_val_pred_multi_sub[:, i])
#                         f1 = f1_score(y_val_multi_sub[:, i], y_val_pred_multi_sub[:, i], average='macro')
#                         mcc_sub_anti.append(mcc)
#                         f1_sub_anti.append(f1)
#                         #June 16th
#                         fpr, tpr, _ = roc_curve(y_val_multi_sub[:, i], y_val_pred_multi_sub[:, i], pos_label=1)
#                         roc_auc = auc(fpr, tpr)
#                         report = classification_report(y_val_multi_sub[:, i], y_val_pred_multi_sub[:, i], labels=[0, 1],
#                                                        output_dict=True)
#                         mcc_val_sub_anti.append(mcc)
#                         score_report_val_sub_anti.append(report)
#                         aucs_val_sub_anti.append(roc_auc)
#                 if anti_number > 1:  # multi-out, scores based on mean of all the involved antibotics
#                     mcc_sub.append(statistics.mean(mcc_sub_anti))  # mcc_sub_anti dimension: n_anti
#                     f1_sub.append(statistics.mean(f1_sub_anti))
#                     # --for validation scores output
#                     score_report_val_sub.append(score_report_val_sub_anti)
#                     aucs_val_sub.append(aucs_val_sub_anti)
#                     mcc_val_sub.append(mcc_val_sub_anti)
#
#
#             Validation_mcc_thresholds.append(mcc_sub)
#             Validation_f1_thresholds.append(f1_sub)
#             # print(Validation_f1_thresholds)
#             # print(Validation_mcc_thresholds)
#             # --for validation scores output
#             score_report_val.append(score_report_val_sub)
#             aucs_val.append(aucs_val_sub)
#             mcc_val.append(mcc_val_sub)
#
#
#     #finish grid search
#     # finish evaluation with all hyper-para combination.
#     if f_fixed_threshold == True and f_optimize_score == 'f1_macro':
#         thresholds_selected = 0.5
#         Validation_f1_thresholds = np.array(Validation_f1_thresholds)
#         # select the inner loop index with the highest f1 score in the column w.r.t. 0.5
#         aim_column = np.where(np.arange(0, 1.1, 0.1) == 0.5)
#         # aim_column = aim_column[0][0]
#         aim_f1 = Validation_f1_thresholds[:, aim_column]
#         weights_selected = np.argmax(aim_f1)
#         # score_val=aim_f1[weights_selected]#foe save
#
#         ind = np.unravel_index(np.argmax(aim_f1, axis=None), aim_f1.shape)
#         hyperparameters_test = list(ParameterGrid(hyper_space))[ind[0]]
#         actual_epoc_test = Validation_actul_epoc[ind[0]]
#         actual_epoc_test_std = Validation_actul_epoc.std(axis=0)
#
#         # only for validation score saving.June 16th
#         score_report_val=score_report_val[ind[0]][aim_column[0][0]]
#         aucs_val=aucs_val[ind[0]][aim_column[0][0]]
#         mcc_val=mcc_val[ind[0]][aim_column[0][0]]
#
#
#     elif f_fixed_threshold == False and f_optimize_score == 'f1_macro':
#         # select the inner loop index,and threshold with the highest f1 score in the matrix
#         Validation_f1_thresholds = np.array(Validation_f1_thresholds)
#         ind = np.unravel_index(np.argmax(Validation_f1_thresholds, axis=None), Validation_f1_thresholds.shape)
#         thresholds_selected = np.arange(0, 1.1, 0.1)[ind[1]]
#         weights_selected = ind[0]  # the order of innerCV# bug ? seems no 13May.
#         # score_val = Validation_f1_thresholds[weights_selected]#for save
#         hyperparameters_test = list(ParameterGrid(hyper_space))[ind[0]]
#         actual_epoc_test = Validation_actul_epoc[ind[0]]
#         actual_epoc_test_std = Validation_actul_epoc.std(axis=0)
#
#         # only for validation score saving.June 16th
#         score_report_val = score_report_val[ind[0]][ind[0]][ind[1]]
#         aucs_val = aucs_val[ind[0]][ind[0]][ind[1]]
#         mcc_val = mcc_val[ind[0]][ind[0]][ind[1]]
#
#     elif f_optimize_score == 'auc':
#
#         thresholds_selected = 0.5
#
#         Validation_auc = np.array(Validation_auc)
#         weights_selected = np.argmax(Validation_auc)
#         # score_val = Validation_auc[weights_selected]#for save
#         ind = np.unravel_index(np.argmax(Validation_auc, axis=None), Validation_auc.shape)
#         print('ind',ind)
#         hyperparameters_test=list(ParameterGrid(hyper_space))[ind[0]]
#         actual_epoc_test=Validation_actul_epoc[ind[0]]
#         actual_epoc_test_std=Validation_actul_epoc.std(axis=0)
#
#         # only for validation score saving
#         score_report_val = score_report_val[ind[0]]
#         aucs_val = aucs_val[ind[0]]
#         mcc_val = mcc_val[ind[0]]
#
#     # print('weights_selected', weights_selected)
#     print('thresholds_selected', thresholds_selected)
#     print('hyperparameters_test',hyperparameters_test)
#     # weight_test.append(weights_selected)
#     # thresholds_selected_test.append(thresholds_selected)
#
#     # save training and val scores w.r.t. selected hyperpara. June 16th
#     score_val = [score_report_val,aucs_val,mcc_val]
#     # with open(save_name_score + '_all_score.pickle', 'wb') as f:  # overwrite
#     #     pickle.dump(score_val, f)
#
#
#     # =======================================================
#     # 3. Re-train on both train and val data with the selected hyper-para
#     data_x_train=np.array(data_x_train)
#     data_y_train = np.array(data_y_train)
#     data_x_val = np.array(data_x_val)
#     data_y_val = np.array(data_y_val)
#     x_train_val = np.concatenate((data_x_train, data_x_val), axis=0)
#     y_train_val =np.concatenate((data_y_train, data_y_val), axis=0)
#     x_train_val = torch.from_numpy(x_train_val).float()
#     y_train_val = torch.from_numpy(y_train_val).float()
#
#     # -
#     # epochs_testing = hyperparameters_test[out_cv]['epochs']
#     n_hidden = hyperparameters_test['n_hidden']
#     learning = hyperparameters_test['lr']
#     n_cl = hyperparameters_test['classifier']
#     dropout = hyperparameters_test['dropout']
#     # generate a NN model
#     if n_cl == 1:
#         classifier = _classifier(nlabel, D_input, n_hidden,
#                                  dropout)  # reload, after testing(there is fine tune traning!)
#     elif n_cl == 2:
#         classifier = _classifier2(nlabel, D_input, n_hidden, dropout)
#     elif n_cl == 3:
#         classifier = _classifier3(nlabel, D_input, n_hidden, dropout)
#     print('testing hyper-parameter: ', hyperparameters_test)
#
#     classifier.to(device)
#     # generate the optimizer - Stochastic Gradient Descent
#     optimizer = optim.SGD(classifier.parameters(), lr=learning)
#     optimizer.zero_grad()  # Clears existing gradients from previous epoch
#     classifier.train()
#
#     print('actual_epoc_test[out_cv]', actual_epoc_test)
#     classifier = fine_tune_training(classifier, int(actual_epoc_test), optimizer, x_train_val, y_train_val,
#                                     anti_number)
#     name_weights = amr_utility.name_utility.GETname_multi_bench_weight(concat_merge_name, species, antibiotics, level,
#                                                                        '', '', 0.0, 0, f_fixed_threshold,
#                                                                        f_no_early_stop, f_optimize_score,
#                                                                        threshold_point, min_cov_point)
#
#     torch.save(classifier.state_dict(), name_weights)
#
#
#     # ===============================================
#     # 4. apply the trained model to the test data
#     score=test(hyperparameters_test,species, antibiotics, name_weights,thresholds_selected,level,path_x_test, path_y_test, f_scaler)
#     score.append([score_val,hyperparameters_test,actual_epoc_test])#the scored used to select best estimator in the first training phase.
#     torch.cuda.empty_cache()
#     return score

def test(hyperparameters,species, antibiotics, weights,threshold_select,level, xdata, ydata, f_scaler):
    '''
     return: scores based on given weights, hyper-parameters, testing set.
    '''
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
    dropout=hyperparameters['dropout']
    # generate a NN model
    if n_cl == 1:
        classifier = _classifier(anti_number, D_input, n_hidden,dropout)  # reload, after testing(there is fine tune traning!)
    elif n_cl == 2:
        classifier = _classifier2(anti_number, D_input, n_hidden,dropout)
    elif n_cl == 3:
        classifier = _classifier3(anti_number, D_input, n_hidden,dropout)

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
            pred_test_sub_anti = pred_test[comp]  # June 21st, auc bugs
            # print('prey_test', pred_test_binary_sub.shape)
            # print('precomy_test', pred_test_binary_sub_anti.shape)
            # print(comp)
            if comp!=[]:
                mcc = matthews_corrcoef(y_test_anti[:, i], pred_test_binary_sub_anti[:, i])
                f1 = f1_score(y_test_anti[:, i], pred_test_binary_sub_anti[:, i], average='macro')
                fpr, tpr, _ = roc_curve(y_test_anti[:, i], pred_test_sub_anti[:, i], pos_label=1)# June 21st, auc bugs
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

