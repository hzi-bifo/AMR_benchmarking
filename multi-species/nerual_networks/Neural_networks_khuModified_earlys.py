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
import itertools
import statistics
from pytorchtools import EarlyStopping


'''
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''
use_cuda = torch.cuda.is_available()
# # use_cuda=False
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
# print('torch.cuda.current_device()', torch.cuda.current_device())
# print(device)

if not sys.warnoptions:
    warnings.simplefilter("ignore")


def plot(anti_number, all_mcc_values, cv, pred_val_all, validation_y, tprs_all, aucs_all, mean_fpr):
    #####Generate MCC plots#####
    # '''
    # Plot the average of the MCC results.

    all_ant1 = []
    for i in range(anti_number):
        cv_list = []
        for c in all_mcc_values:
            all_ant = []
            for each in c:
                each = np.array(each)
                if anti_number == 1:
                    all_ant.append(each)
                else:
                    all_ant.append(each[:, i])
            cv_list.append(all_ant)
        all_ant1.append((np.sum(cv_list, axis=0))[0])

    colors = ["darkblue", "darkred", "darkgreen",
              "orange", "purple", "magenta"] * 100
    legends = range(1, 501)

    for i in range(anti_number):
        aver_all_mcc = []
        for val in all_ant1[i]:
            aver_all_mcc.append(val / cv)  # average of the MCC results
        plt.plot(np.arange(0, 1.1, 0.1), aver_all_mcc,
                 color=colors[i], alpha=1, label=str(legends[i]))
    plt.xlim([0, 1])
    plt.xlabel("Thresholds")
    plt.ylabel("MCC average values for %s fold CV" % str(cv))
    plt.title("Thresholds vs MCC values")
    plt.legend(loc="best")
    plt.savefig("log/results/MCC_output_test.png")
    plt.close()



    # plot the AUC values
    plt.figure(figsize=(8, 8))  # figure size
    # take the average of the validation predictions
    pred_val_all = np.sum(pred_val_all, axis=0) / float(cv)

    # calculate the AUCs for the validation data
    # do not take into consideration missing outputs
    for i in range(anti_number):
        comp = []
        for t in range(len(validation_y)):
            if anti_number == 1:
                if -1 != validation_y[t] or -1.0 != validation_y[t]:
                    comp.append(t)
            else:
                if -1 != validation_y[t][i] or -1.0 != validation_y[t][i]:
                    comp.append(t)
        y_val_sub = validation_y[comp]
        pred_sub_val = pred_val_all[comp]
        tprs = []
        aucs = []

        for t in tprs_all:
            tprs.append(t[i])
        for z in aucs_all:
            aucs.append(z[i])
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color=colors[i],
                 label=r'%s-Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (legends[i], mean_auc, std_auc), lw=2, alpha=.8)
        if anti_number == 1:
            fpr1, tpr1, _ = roc_curve(y_val_sub, pred_sub_val, pos_label=1)
        else:
            fpr1, tpr1, _ = roc_curve(
                y_val_sub[:, i], pred_sub_val[:, i], pos_label=1)
        roc_auc1 = auc(fpr1, tpr1)
        plt.plot(fpr1, tpr1, color=colors[i], alpha=1, lw=1,
                 label='ROC curve for validation (area = %0.2f)' % roc_auc1)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2,
             color='r', label='Chance', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig("./results/ROC_curve.png")
    plt.close()
    # '''

def cluster_split(dict_cluster, Random_State, cv):
    # Custom k fold cross validation
    # cross validation method divides the clusters and adds to the partitions.
    # Samples are not divided directly due to sample similarity issue
    all_data_splits_pre = []  # cluster order
    all_data_splits = []  # cluster order
    all_available_data = range(0, len(dict_cluster))  # all the clusters had
    clusters_n = len(dict_cluster)  # number of clusters
    all_samples = []  # all the samples had in the clusters
    for i in dict_cluster:
        for each in dict_cluster[i]:
            all_samples.append(each)

    # Shuffle the clusters and divide them
    shuffled = list(utils.shuffle(list(dict_cluster.keys()),
                                  random_state=Random_State))  # shuffled cluster names

    # Divide the clusters equally
    r = int(len(shuffled) / cv)  # batches size,e.g. 105/5=21

    a = 0
    b = r
    for i in range(cv):  # 5

        all_data_splits_pre.append(shuffled[a:b])

        if i != cv - 2:  # 1 =0,1,2,4
            a = b
            b = b + r

        else:
            a = b
            b = len(shuffled)

    # Extract the samples inside the clusters
    # If the number of samples are lower than the expected number of samples per partition, get an extra cluster

    totals = []
    all_extra = []
    for i in range(len(all_data_splits_pre)):  # 5.#cluster order
        if i == 0:
            m_fromprevious = i + 1  # 1
        else:
            m_fromprevious = np.argmax(totals)  # the most samples CV index
        tem_Nsamples = []  # number of samples in the selected clusters
        extracted = list(set(all_data_splits_pre[i]) - set(all_extra))  # order of cluster, w.r.t dict_cluster

        for e in extracted:
            elements = dict_cluster[str(e)]
            tem_Nsamples.append(len(elements))
        sum_tem = sum(tem_Nsamples)


        a = 0
        while sum_tem + 100 < len(all_samples) / float(cv):  # all_samples: val,train,test
            extra = list(utils.shuffle(
                all_data_splits_pre[m_fromprevious], random_state=Random_State))[a]  # cluster order
            extracted = extracted + [extra]  # cluster order
            all_extra.append(extra)
            a = a + 1
            tem_Nsamples = []
            print('extracted', extracted)
            for e in extracted:  # every time add one cluster order
                elements = dict_cluster[str(e)]
                tem_Nsamples.append(len(elements))  # sample number
            sum_tem = sum(tem_Nsamples)
            for item in range(len(all_data_splits)):
                all_data_splits[item] = list(
                    set(all_data_splits[item]) - set([extra]))  # rm previous cluster order, because it's moved to this fold.
        totals.append(sum_tem)  # sample number for each CV. #no use afterwards.
        all_data_splits.append(extracted)  ##cluster order for each CV
        print('sum_tem', sum_tem)  # all_samples in that folder: val,train,test.
        print('average', len(all_samples) / float(cv))
        print('len(all_samples)', len(all_samples))
    return all_data_splits
def prepare_cluster(fileDir, p_clusters):
    # prepare a dictionary for clusters, the keys are cluster numbers, items are sample names.
    # cluster index collection.
    # cluster order list
    filename = os.path.join(fileDir, p_clusters)
    filename = os.path.abspath(os.path.realpath(filename))
    output_file_open = open(filename, "r")
    cluster_summary = output_file_open.readlines()
    dict_cluster = collections.defaultdict(list)  # key: starting from 0

    for each_cluster in cluster_summary:
        splitted = each_cluster.split()
        if splitted != []:
            if splitted[0].isdigit():
                dict_cluster[str(int(splitted[0]) - 1)].append(splitted[3])
            if splitted[0] == "Similar":
                splitted = each_cluster.split()
                splitted_2 = each_cluster.split(":")
                dict_cluster[str(int(splitted_2[1].split()[0]) - 1)
                ].append(splitted[6])
    # print("dict_cluster: ", dict_cluster)
    return dict_cluster
def prepare_sample_name(fileDir, p_names):
    # sample name list

    filename = os.path.join(fileDir, p_names)
    filename = os.path.abspath(os.path.realpath(filename))
    names_open = open(filename, "r")
    names_read = names_open.readlines()
    # sample names collection.
    names = []
    for each in range(len(names_read)):
        names.append(names_read[each].replace("\n", ""))  # correct the sample names #no need w.r.t. Patric data
    return names
def prepare_folders(cv, Random_State, dict_cluster, names):
    all_data_splits = cluster_split(dict_cluster, Random_State,
                                    cv)  # split cluster into cv Folds. len(all_data_splits)=5
    folders_sample = []  # collection of samples for each split
    for out_cv in range(cv):
        folders_sample_sub = []
        iter_clusters = all_data_splits[out_cv]  # clusters included in that split
        for cl_ID in iter_clusters:
            for element in dict_cluster[cl_ID]:
                folders_sample_sub.append(names.index(element))  # extract cluster ID from the rest folders. 4*(cluster_N)
        folders_sample.append(folders_sample_sub)

    return folders_sample
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


def training(classifier,epochs,optimizer,x_train,y_train,x_val,y_val, anti_number):
    patience = 200

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

    return classifier

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



def score_summary(cv,score_report_test,aucs_test,mcc_test,save_name_score,thresholds_selected_test):

    summary = pd.DataFrame(index=['mean','std'], columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy_macro',
                                                          'auc','mcc','threshold'])
    #
    f1=[]
    precision=[]
    recall=[]
    accuracy=[]
    for i in np.arange(cv):
        report=score_report_test[i]
        report=pd.DataFrame(report).transpose()
        print(report)
        print('--------')
        f1.append(report.loc['macro avg','f1-score'])
        precision.append(report.loc['macro avg','precision'])
        recall.append(report.loc['macro avg','recall'])
        accuracy.append(report.loc['accuracy','f1-score'])
    summary.loc['mean','f1_macro']=statistics.mean(f1)
    summary.loc['std','f1_macro']=statistics.stdev(f1)
    summary.loc['mean','precision_macro'] = statistics.mean(precision)
    summary.loc['std','precision_macro'] = statistics.stdev(precision)
    summary.loc['mean','recall_macro']  = statistics.mean(recall)
    summary.loc['std','recall_macro'] = statistics.stdev(recall)
    summary.loc['mean','accuracy_macro'] = statistics.mean(accuracy)
    summary.loc['std','accuracy_macro'] = statistics.stdev(accuracy)
    summary.loc['mean','auc'] = statistics.mean(aucs_test)
    summary.loc['std','auc'] = statistics.stdev(aucs_test)
    summary.loc['mean','mcc'] = statistics.mean(mcc_test)
    summary.loc['std','mcc'] = statistics.stdev(mcc_test)
    summary.loc['mean', 'threshold'] = statistics.mean(thresholds_selected_test)
    summary.loc['std', 'threshold'] = statistics.stdev(thresholds_selected_test)
    summary.to_csv('log/temp/'+save_name_score+'_score.txt', sep="\t")
    print(summary)





def eval(species, antibiotics, level, xdata, ydata, p_names, p_clusters, cv, random, hidden, epochs,re_epochs, learning,f_scaler,f_fixed_threshold):

    fileDir = os.path.dirname(os.path.realpath('__file__'))
    # sample name list
    names=prepare_sample_name(fileDir, p_names)
    # prepare a dictionary for clusters, the keys are cluster numbers, items are sample names.
    dict_cluster = prepare_cluster(fileDir, p_clusters)

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
    Random_State = random

    # -------------------------------------------------------------------------------------------------------------------
    ###construct the Artificial Neural Networks Model###
    # The feed forward NN has only one hidden layer
    # The activation function used in the input and hidden layer is ReLU, in the output layer the sigmoid function.
    # -------------------------------------------------------------------------------------------------------------------


    n_hidden = hidden# number of neurons in the hidden layer
    learning_rate = learning

    anti_number = data_y[0].size  # number of antibiotics
    D_input = len(data_x[0])  # number of neurons in the input layer
    N_sample = len(data_x)  # number of input samples #should be equal to len(names)
    nlabel = data_y[0].size  # number of neurons in the output layer

    # cross validation loop where the training and testing performed.
    #khu:nested CV.


    # =====================================
    # training
    # =====================================
    folders_sample=prepare_folders(cv, Random_State, dict_cluster, names)

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
            print('sample lens for inner CV:',len(x_train), len(x_train[0]),len(x_val), len(x_val[0]))
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
            y_val = torch.from_numpy(y_val).float()
            pred_val_inner=[]#predicted results on validation set.
            ###################
            # 1. train the model #
            ###################


            # generate a NN model
            classifier = _classifier(nlabel, D_input, n_hidden)#reload, after testing(there is fine tune traning!)
            classifier.to(device)

            # generate the optimizer - Stochastic Gradient Descent
            optimizer = optim.SGD(classifier.parameters(), lr=learning_rate)
            optimizer.zero_grad()  # Clears existing gradients from previous epoch
            #loop:
            classifier=training(classifier, epochs, optimizer, x_train, y_train,x_val,y_val, anti_number)


            #==================================================

            print(species, antibiotics,level, out_cv, innerCV)
            name_weights = amr_utility.name_utility.name_multi_bench(species, antibiotics,level, out_cv, innerCV)
            torch.save(classifier.state_dict(), name_weights)

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
            pred_val_inner.append(pred_val_sub)#for all innerCV at this out_cv. #No further use so far. April,30.
            print('pred_val_inner shape:', np.array(pred_val_inner).shape)#(cv-1)*n_sample

            ##
            # Calculate macro f1. for thresholds from 0 to 1.

            mcc_sub = []
            f1_sub = []
            for threshold in np.arange(0, 1.1, 0.1):
                # predictions for the test data
                # all_thresholds = []

                pred = []
                for probability in pred_val_sub:# for this innerCV at this out_cv
                    if probability > threshold:
                        pred.append(1)
                    else:
                        pred.append(0)
                y_val = np.array(y_val)#ground truth
                y_val_pred = np.array(pred)


                for i in range(anti_number):
                    if anti_number == 1:
                        mcc=matthews_corrcoef(y_val, y_val_pred)
                        # report = classification_report(y_val, y_val_pred, labels=[0, 1], output_dict=True)
                        f1=f1_score(y_val, y_val_pred, average='macro')
                    else:# multi-out

                        mcc = matthews_corrcoef(y_val[:,i], y_val_pred[:,i])
                        f1 = f1_score(y_val[:,i], y_val_pred[:,i], average='macro')
                    mcc_sub.append(mcc)
                    f1_sub.append(f1)
                # todo
                # all_thresholds.append(mcc_sub)#for multi-output & visualization

            Validation_mcc_thresholds.append(mcc_sub)
            Validation_f1_thresholds.append(f1_sub)
        #====================================================
        if f_fixed_threshold==True:
            thresholds_selected=0.5
            Validation_f1_thresholds = np.array(Validation_f1_thresholds)
            #select the inner loop index with the highest f1 score in the column w.r.t. 0.5
            aim_column=np.where(np.arange(0, 1.1, 0.1) == 0.5)
            aim_f1=Validation_f1_thresholds[:,aim_column]
            weights_selected=np.argmax(aim_f1)

        else:
            # select the inner loop index,and threshold with the highest f1 score in the matrix
            Validation_f1_thresholds=np.array(Validation_f1_thresholds)
            ind = np.unravel_index(np.argmax(Validation_f1_thresholds, axis=None), Validation_f1_thresholds.shape)
            thresholds_selected=np.arange(0, 1.1, 0.1)[ind[1]]
            weights_selected=ind[0]#the order of innerCV

        weight_test.append(weights_selected)
        thresholds_selected_test.append(thresholds_selected)

        print(Validation_f1_thresholds)
        print(Validation_mcc_thresholds)
        print('weights_selected',weights_selected)
        print('thresholds_selected', thresholds_selected)



        name_weights = amr_utility.name_utility.name_multi_bench(species, antibiotics, level,out_cv, weights_selected)
        classifier.load_state_dict(torch.load(name_weights))

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
        classifier.train()
        optimizer = optim.SGD(classifier.parameters(), lr=0.0001)
        classifier = fine_tune_training(classifier, re_epochs, optimizer, x_train_val, y_train_val, anti_number)
        name_weights = amr_utility.name_utility.name_multi_bench(species, antibiotics,level, out_cv,'')

        torch.save(classifier.state_dict(), name_weights)

        # rm inner loop models' weight in the log
        for i in np.arange(cv-1):
            n = amr_utility.name_utility.name_multi_bench(species, antibiotics,level, out_cv, i)
            os.system("rm " + n)

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
        pred_test_binary_sub = []
        for probability in pred_test_sub:

            if probability > thresholds_selected:
                pred_test_binary_sub.append(1)
            else:
                pred_test_binary_sub.append(0)
        y_test = np.array(y_test)
        pred_test_binary_sub = np.array(pred_test_binary_sub)
        pred_test_binary.append(pred_test_binary_sub)
        aucs = []
        tprs = []
        # print(y_test)
        # print(pred_test_binary_sub)

        for i in range(anti_number):
            if anti_number == 1:
                mcc = matthews_corrcoef(y_test, pred_test_binary_sub)
                # report = classification_report(y_val, y_val_pred, labels=[0, 1], output_dict=True)
                f1 = f1_score(y_test, pred_test_binary_sub, average='macro')
                print(f1)
                print(mcc)
                report = classification_report(y_test, pred_test_binary_sub, labels=[0, 1],output_dict=True)
                print(report)
                fpr, tpr, _ = roc_curve(y_test, pred_test_binary_sub, pos_label=1)

            else:  # multi-out
                #todo
                mcc = matthews_corrcoef(y_test[:, i], pred_test_binary_sub[:, i])
                f1 = f1_score(y_test[:, i], pred_test_binary_sub[:, i], average='macro')
                fpr, tpr, _ = roc_curve(y_test[:, i], pred_test_binary_sub[:, i], pos_label=1)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

        #note,here single output also matrixed, for use of the plot codes by the original author.
        f1_test.append(f1)
        score_report_test.append(report)
        aucs_test_all.append(aucs)## multi-out
        tprs_test.append(tprs)## multi-out
        aucs_test.append(roc_auc) #single-out

        mcc_test.append(mcc)


    # plot(anti_number, mcc_test, cv, validation, pred_val_all, validation_y, tprs_test, aucs_test_all, mean_fpr)

    save_name_score=amr_utility.name_utility.name_multi_bench_save_name_score(species, antibiotics,level)
    # if f_fixed_threshold==True:
    save_name_score=save_name_score+'_fixed_threshold_'+str(f_fixed_threshold)+'e_'+str(epochs)+'lr_'+str(learning)

    print('thresholds_selected_test',thresholds_selected_test)
    print('f1_test',f1_test)
    print('mcc_test',mcc_test)
    score_summary(cv, score_report_test, aucs_test, mcc_test, save_name_score,thresholds_selected_test)#save mean and std of each 6 score

    torch.cuda.empty_cache()


