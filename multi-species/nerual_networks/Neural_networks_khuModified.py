#!/usr/bin/python
#Python 3.6
#nested CV, modified by khu based on Derya Aytan's work.
#https://bitbucket.org/deaytan/neural_networks/src/master/Neural_networks.py
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
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
# import nltk
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
import collections
import random
from sklearn import utils
from sklearn import preprocessing
import argparse
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

# if not sys.warnoptions:
#     warnings.simplefilter("ignore")


def plot(anti_number, all_mcc_values, cv, validation, pred_val_all, validation_y, tprs_all, aucs_all, mean_fpr):
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
    plt.savefig("./results/MCC_output_test.png")
    plt.close()

    # Calculate the MCC for the validation data, for the same thresholds
    all_thresholds_valid = []
    for threshold in np.arange(0, 1.1, 0.1):
        pred_validation = []
        for r in range(0, len(validation)):
            all_predictions = []
            for item in pred_val_all:
                all_predictions.append(item[r])
            all_predictions_all = np.array(all_predictions).sum(
                axis=0) / float(cv)  # take the average of the probabilities
            temp = []
            for c in all_predictions_all:
                if c > threshold:
                    temp.append(1)
                else:
                    temp.append(0)

            pred_validation.append(temp)
        pred_validation = np.array(pred_validation)

        # mcc values for the validation data
        # do not take into consideration missing outputs.
        mcc_all = []
        for i in range(anti_number):
            comp = []
            for t in range(len(validation_y)):
                if anti_number == 1:
                    if -1 != validation_y[t]:
                        comp.append(t)
                else:
                    if -1 != validation_y[t][i]:
                        comp.append(t)
            y_val_sub = validation_y[comp]
            pred_sub_val = pred_validation[comp]
            if anti_number == 1:
                mcc = matthews_corrcoef(y_val_sub, pred_sub_val)
            else:
                mcc = matthews_corrcoef(y_val_sub[:, i], pred_sub_val[:, i])
            mcc_all.append(mcc)
        all_thresholds_valid.append(mcc_all)

    # plot MCC values per output
    for i in range(anti_number):
        aver_all_mcc = []
        for val in all_thresholds_valid:
            if anti_number == 1:
                aver_all_mcc.append(val)
            else:
                aver_all_mcc.append(val[i])
        plt.plot(np.arange(0, 1.1, 0.1), aver_all_mcc,
                 color=colors[i], alpha=1, label=legends[i])
    plt.xlim([0, 1])
    plt.xlabel("Thresholds")
    plt.ylabel("MCC average values for %s fold CV" % str(cv))
    plt.title("Thresholds vs MCC values")
    plt.legend(loc="best")
    plt.savefig("./results/MCC_output_validation.png")
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

        print('sum_tem', sum_tem)  # all_samples: val,train,test.
        print(len(all_samples) / float(cv))
        print(len(all_samples))
        a = 0
        while sum_tem + 200 < len(all_samples) / float(cv):  # all_samples: val,train,test
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
                    set(all_data_splits[item]) - set(
                        [extra]))  # rm previous cluster order, because it's moved to this fold.
        totals.append(sum_tem)  # sample number for each CV. #no use afterwards.
        all_data_splits.append(extracted)  ##cluster order for each CV
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
    print("dict_cluster: ", dict_cluster)
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

class _classifier(nn.Module):
    def __init__(self, nlabel,D_in,H):
        super(_classifier, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, nlabel),
        )

    def forward(self, input):
        input.to(device)
        return self.main(input).to(device)



def eval(species, xdata, ydata, p_names, p_clusters, cv, random, hidden, epochs, learning, level, output):
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    # sample name list
    names=prepare_sample_name(fileDir, p_names)
    # prepare a dictionary for clusters, the keys are cluster numbers, items are sample names.
    dict_cluster = prepare_cluster(fileDir, p_clusters)

    #data
    data_x = np.loadtxt(xdata, dtype="float")
    data_y =np.loadtxt(ydata)

    ##prepare data stores##
    all_mcc_values = []  # MCC results for the test data
    all_mcc_values_2 = []  # MCC results for the validation data

    pred_val_all = []  # probabilities are for the validation set
    pred_test_res_all = []  # probabilities for the test set

    aucs_all = []  # all AUC values for the test data
    tprs_all = []  # all True Positives for the test data
    mean_fpr = np.linspace(0, 1, 100)
    Random_State = random
    all_data_splits = cluster_split(dict_cluster,Random_State,cv) # split cluster into cv Folds. len(all_data_splits)=5
    # -------------------------------------------------------------------------------------------------------------------
    ###construct the Artificial Neural Networks Model###
    # The feed forward NN has only one hidden layer
    # The activation function used in the input and hidden layer is ReLU, in the output layer the sigmoid function.
    # -------------------------------------------------------------------------------------------------------------------
    m_sigmoid = nn.Sigmoid()  # sigmoid function for the ooutput layer
    n_epochs = epochs
    n_hidden = hidden# number of neurons in the hidden layer
    learning_rate = learning

    anti_number = data_y[0].size  # number of antibiotics
    D_input = len(data_x[0])  # number of neurons in the input layer
    N_sample = len(data_x)  # number of input samples #should be equal to len(names)
    nlabel = data_y[0].size  # number of neurons in the output layer
    epochs = n_epochs  # number of iterations the each model will be trained
    # cross validation loop where the training and testing performed.
    #khu:nested CV.
    # generate a NN model
    classifier = _classifier(nlabel, D_input, n_hidden)
    classifier.to(device)

    # generate the optimizer - Stochastic Gradient Descent
    optimizer = optim.SGD(classifier.parameters(), lr=learning_rate)

    # =====================================
    # training
    # =====================================
    # iterate the loop for number of epochs
    for iter_cv in range(cv):
        test_clusters = all_data_splits[iter_cv]  # clusters including test data
        # clusters including validation data
        remain = list(set(range(cv)) - set([iter_cv])).sort() #first round: set(0,1,2,3,4)-set(0)=set(1,2,3,4)


        #inner CV : innerCV=cv-1
        for
            training_clusters = []
            for cv_ID in remain:
                for cl_ID in all_data_splits[cv_ID]:
                    training_clusters.append(cl_ID)#extract cluster ID from the rest folders

            # extract the training data indexes:
            #khu: note, inside the
            train_samples = []
            for cl_ID in training_clusters:
                for element in dict_cluster["%s" % cl_ID]:
                    train_samples.append(names.index(element))# index in names list
                    # if element in names_all:
                    #     ind = names_all.index(element)






        # extract the test data indexes:
        test_samples = []
        for cl_ID in test_clusters:
            for element in dict_cluster["%s" % cl_ID]:
                if element in names:#names:samples for training and testing.khu: can delete for nested CV
                    test_samples.append(names.index(element))
                # if element in names_all:
                #     ind = names_all.index(element)

        # training and test samples
        # select by order
        x_train, x_test = data_x[train_samples], data_x[test_samples]
        y_train, y_test = data_y[train_samples], data_y[test_samples]
        print(len(x_train),len(x_train[0]))#vary for each CV
        print(len(x_test), len(x_test[0]))
        x_train=torch.from_numpy(x_train).float()
        y_train=torch.from_numpy(y_train).float()
        x_train=x_train.to(device)
        y_train=y_train.to(device)



        #todo chcek
        # if user chose to normalize the data
        # for currentArgument, currentValue in arguments:
        #     if currentArgument in ("-n", "--normal"):
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        # scale the test data based on the training data
        x_test = scaler.transform(x_test)
        # # scale the validation data based on the training data
        # validation_x = scaler.transform(validation_x_raw)

        classifier.train()
        for epoc in range(epochs):
            optimizer.zero_grad()  # Clears existing gradients from previous epoch

            x_train_new = torch.utils.data.TensorDataset(x_train)
            y_train_new = torch.utils.data.TensorDataset(y_train)

            all_data = list(zip(x_train_new, y_train_new))


            # the model is trained for 100 batches
            data_loader = torch.utils.data.DataLoader(
                all_data, batch_size=100, drop_last=False)

            losses = []  # save the error for each iteration
            for i, (sample_x, sample_y) in enumerate(data_loader):
                inputv = sample_x[0]
                # print(we.size())
                inputv =torch.FloatTensor(inputv)
                inputv=Variable(inputv).view(len(inputv), -1)
                # print(inputv.size())

                if anti_number == 1:
                    labelsv = sample_y[0].view(len(sample_y[0]), -1)
                else:
                    labelsv = sample_y[0][:, :]
                weights = labelsv.data.clone().view(len(sample_y[0]), -1)
                print(weights)
                # That step is added to handle missing outputs.
                # Weights are not updated in the presence of missing values.
                weights[weights == 1.0] = 1
                weights[weights == 0.0] = 1
                # weights[weights < 0] = 0
                weights.to(device)
                print(weights)

                # Calculate the loss/error using Binary Cross Entropy Loss

                criterion = nn.BCELoss(weight=weights, reduction="none")
                output = classifier(inputv)
                # print(output.size())
                # print(labelsv.size())
                loss = criterion(m_sigmoid(output), labelsv)
                loss = loss.mean()  # compute loss
                optimizer.zero_grad()  # zero gradients #previous gradients do not keep accumulating
                loss.backward()  # backpropagation
                optimizer.step()  # weights updated
                losses.append(loss.data.mean())
            if epoc % 100 == 0:
                # print the loss per iteration
                print('[%d/%d] Loss: %.3f' % (epoc+1, epochs, np.mean(losses)))

        # apply the trained model to the validation data
        classifier.eval()  # eval mode
        pred_val = []
        for v, v_sample in enumerate(validation_x):
            val = Variable(torch.FloatTensor(v_sample)).view(1, -1)
            output_test = classifier(val)
            out = m_sigmoid(output_test)
            temp = []
            for h in out[0]:
                temp.append(float(h))
            pred_val.append(temp)
        pred_val = np.array(pred_val)
        pred_val_all.append(pred_val)

        # apply the trained model to the test data
        pred_test_res = []
        for a, a_sample in enumerate(x_test):
            tested = Variable(torch.FloatTensor(a_sample)).view(1, -1)
            output_test = classifier(tested)
            out = m_sigmoid(output_test)
            temp = []
            for h in out[0]:
                temp.append(float(h))
            pred_test_res.append(temp)
        pred_test_res = np.array(pred_test_res)
        pred_test_res_all.append(pred_test_res)

        # '''
        # ================================================================================================================
        # ===============================================================================================================
        # Calculate MCC for thresholds from 0 to 1.
        # correlation coefficient between the actual and predicted series
        all_thresholds = []  # MCC values for test data
        all_thresholds_2 = []  # MCC values for training data
        for threshold in np.arange(0, 1.1, 0.1):
            # predictions for the test data
            pred = []
            mcc_all = []

            for x, x_sample in enumerate(x_test):
                test = Variable(torch.FloatTensor(x_sample)).view(1, -1)
                output_test = classifier(test)
                out = m_sigmoid(output_test)
                temp = []
                for c in out[0]:
                    if c > threshold:
                        temp.append(1)
                    else:
                        temp.append(0)

                pred.append(temp)
            y_test = np.array(y_test)
            pred = np.array(pred)

            # predictions for the training data
            pred_2 = []
            mcc_all_2 = []
            for x, x_sample in enumerate(x_train):
                train = Variable(torch.FloatTensor(x_sample)).view(1, -1)
                output_test = classifier(train)
                out = m_sigmoid(output_test)
                temp = []
                for c in out[0]:
                    if c > threshold:
                        temp.append(1)
                    else:
                        temp.append(0)

                pred_2.append(temp)
            y_train = np.array(y_train)
            pred_2 = np.array(pred_2)

            # mcc values for test and train data
            # exclude missing outputs
            for i in range(anti_number):
                comp = []
                for t in range(len(y_train)):
                    if anti_number == 1:
                        if -1 != y_train[t]:
                            comp.append(t)
                    else:
                        if -1 != y_train[t][i]:
                            comp.append(t)
                y_train_sub = y_train[comp]
                pred_sub_2 = pred_2[comp]

                comp2 = []
                for t in range(len(y_test)):
                    if anti_number == 1:
                        if -1 != y_test[t]:
                            comp2.append(t)
                    else:
                        if -1 != y_test[t][i]:
                            comp2.append(t)
                y_test_sub = y_test[comp2]
                pred_sub = pred[comp2]
                if anti_number == 1:
                    mcc = matthews_corrcoef(y_test_sub, pred_sub)
                    mcc_2 = matthews_corrcoef(y_train_sub, pred_sub_2)
                else:
                    mcc = matthews_corrcoef(y_test_sub[:, i], pred_sub[:, i])
                    mcc_2 = matthews_corrcoef(
                        y_train_sub[:, i], pred_sub_2[:, i])
                # print(mcc)
                mcc_all.append(mcc)
                mcc_all_2.append(mcc_2)
            all_thresholds.append(mcc_all)
            all_thresholds_2.append(mcc_all_2)
        all_mcc_values.append([all_thresholds])
        all_mcc_values_2.append(all_thresholds_2)

        # '''
        # TODO add Positive predictive value (PPV), Precision; Accuracy (ACC); f1-score here
        # Calculate the AUC values, Area Under the Curve.
        # y: True positive rate (TPR), aka. sensitivity, hit rate, and recall,
        # X: False positive rate (FPR), aka. fall-out,
        # Exclude missing values
        aucs = []
        tprs = []
        for a in range(anti_number):
            comp_auc = []
            for t in range(len(y_test)):  # sample order
                if anti_number == 1:  # only one antibiotic
                    if -1 != y_test[t]:
                        comp_auc.append(t)

                else:  # multi-output
                    if -1 != y_test[t][a]:
                        comp_auc.append(t)

            y_test_auc = y_test[comp_auc]
            pred_auc = pred_test_res[comp_auc]

            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            # Compute micro-average ROC curve and ROC area
            if anti_number == 1:
                fpr, tpr, _ = roc_curve(y_test_auc, pred_auc, pos_label=1)
                # fpr: Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i].
            else:
                fpr, tpr, _ = roc_curve(
                    y_test_auc[:, a], pred_auc[:, a], pos_label=1)
            # mean_fpr = np.linspace(0, 1, 100)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            #plt.plot(fpr, tpr, color='darkorange', alpha = 0.3,lw=1)
        aucs_all.append(aucs)
        tprs_all.append(tprs)
    plot(anti_number, all_mcc_values, cv, validation, pred_val_all, validation_y, tprs_all, aucs_all, mean_fpr)


    torch.cuda.empty_cache()

def extract_info(s,xdata,ydata,p_names,p_clusters,cv_number, random, hidden, epochs, learning, level,output,n_jobs):
    data = pd.read_csv('metadata/loose_Species_antibiotic_FineQuality.csv', index_col=0, dtype={'genome_id': object},
                       sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    # for training model on part of the dataset:-------------
    # data=data.loc[['Escherichia coli'],:]
    data = data.loc[s, :]
    # data.at['Mycobacterium tuberculosis', 'modelling antibiotics']=['capreomycin', 'ciprofloxacin']
    # --------------------------------------------------------
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    print(data)
    # pool = mp.Pool(processes=5)
    # pool.starmap(determination, zip(df_species,repeat(l),repeat(n_jobs)))
    for species in df_species:
        eval(species, xdata,ydata,p_names,p_clusters,cv_number, random, hidden, epochs, learning, level,output)




if __name__== '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-x", "--xdata", default=None, type=float, required=True,
						help='input x data')
    parser.add_argument("-y", "--ydata", default=None, type=int, required=True,
                        help='output y data')# todo check type
    parser.add_argument("-names", "--p_names", default=None, type=str, required=True,
						help='path to list of sample names')
    parser.add_argument("-c", "--p_clusters", default=None, type=str, required=True,
                        help='path to the sample clusters')
    parser.add_argument("-cv", "--cv_number", default=10, type=int, required=True,
                        help='CV splits number')
    parser.add_argument("-r", "--random", default=42, type=int, required=True,
                        help='random state related to shuffle cluster order')
    parser.add_argument("-d", "--hidden", default=200, type=int, required=True,
                        help='dimension of hidden layer')
    parser.add_argument("-e", "--epochs", default=1000, type=int, required=True,
                        help='epochs')
    parser.add_argument("-learing", "--learning", default=0.001, type=int, required=True,
                        help='learning rate')
    parser.add_argument('--l', '--level', default=None, type=str, required=True,
                        help='Quality control: strict or loose')
    parser.add_argument("-o","--output", default=None, type=str, required=True,
						help='Output file names')
    parser.add_argument('--s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
            \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
            \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parsedArgs = parser.parse_args()
    # parser.print_help()
    # print(parsedArgs)
    extract_info(parsedArgs.s,parsedArgs.xdata,parsedArgs.ydata,parsedArgs.p_names,parsedArgs.p_clusters,parsedArgs.cv_number,
                 parsedArgs.random,parsedArgs.hidden,parsedArgs.epochs,parsedArgs.learning,parsedArgs.level,parsedArgs.output,parsedArgs.n_jobs)

