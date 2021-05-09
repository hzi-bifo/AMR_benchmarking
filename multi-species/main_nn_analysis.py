import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import numpy as np
import ast
import amr_utility.name_utility
import amr_utility.graph_utility
import argparse
import amr_utility.load_data
import pandas as pd
import neural_networks.Neural_networks_khuModified as nn_module
import pickle
import statistics
import math

def make_visualization(species,antibiotics,level,f_fixed_threshold,epochs,learning):
    #todo need re work

    print(species)
    # antibiotics_selected = ast.literal_eval(antibiotics)
    print('====> Select_antibiotic:', len(antibiotics), antibiotics)
    final=pd.DataFrame(index=antibiotics, columns=['weighted-f1_macro', 'weighted-precision_macro', 'weighted-recall_macro', 'weighted-accuracy_macro',
                                                          'weighted-auc','weighted-mcc','weighted-threshold'] )
    print(final)
    for anti in antibiotics:
        save_name_score = amr_utility.name_utility.name_multi_bench_save_name_score(species, anti,level,learning,epochs,f_fixed_threshold)

        print(anti, '--------------------------------------------------------------------')

        data = pd.read_csv('log/temp/'+save_name_score+'_score.txt', sep="\t",index_col=0, header=0)
        print(data)
        data=data.loc[['weighted-mean','weighted-std'],:]
        print(data)
        data = data.astype(float).round(2)
        print(data)
        m= data.loc['weighted-mean',:].apply(lambda x: "{:.2f}".format(x))
        print(m)
        n=data.loc['weighted-std',:].apply(lambda x: "{:.2f}".format(x))
        # print(data.dtypes)

        final.loc[anti,:]=m.str.cat(n, sep='±').values

        print(final)


    # final=final.astype(float).round(2)
    final.to_csv('log/results/'+save_name_score+'_score_final.txt', sep="\t")

def weithgted_var(values,average,weights):
    n=len(values)
    p_variance = np.average((values - average) ** 2, weights=weights)#np-int, element-wise
    s_variance=p_variance * n/(n-1)
    return s_variance

def score_summary(cv,score_report_test,aucs_test,mcc_test,save_name_score,thresholds_selected_test):

    summary = pd.DataFrame(index=['mean','std','weighted-mean','weighted-std'], columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy_macro',
                                                          'auc','mcc','threshold'])

    #
    f1=[]
    precision=[]
    recall=[]
    accuracy=[]
    support=[]
    f_noAccu = []
    for i in np.arange(cv):
        report=score_report_test[i]
        report=pd.DataFrame(report).transpose()
        print(report)
        print('--------')

        if 'accuracy' not in report.index.to_list():# no resitance pheno in test folder
            accuracy.append('-')
            f_noAccu.append(i)
        else:
            accuracy.append(report.loc['accuracy', 'f1-score'])

        f1.append(report.loc['macro avg','f1-score'])
        precision.append(report.loc['macro avg','precision'])
        recall.append(report.loc['macro avg','recall'])
        support.append(report.loc['macro avg','support'])



    if f_noAccu != []:
        #rm the iteration's results, where no resistance phenotype in the test folder.
        f1 = [i for j, i in enumerate(f1) if j not in f_noAccu]
        precision = [i for j, i in enumerate(precision) if j not in f_noAccu]
        recall = [i for j, i in enumerate(recall) if j not in f_noAccu]
        accuracy = [i for j, i in enumerate(accuracy) if j not in f_noAccu]
        support = [i for j, i in enumerate(support) if j not in f_noAccu]
        mcc_test = [i for j, i in enumerate(mcc_test) if j not in f_noAccu]
        aucs_test = [i for j, i in enumerate(aucs_test) if j not in f_noAccu]
        thresholds_selected_test = [i for j, i in enumerate(thresholds_selected_test) if j not in f_noAccu]


    summary.loc['mean','accuracy_macro'] = statistics.mean(accuracy)
    summary.loc['std','accuracy_macro'] = statistics.stdev(accuracy)
    summary.loc['mean', 'f1_macro'] = statistics.mean(f1)
    summary.loc['std', 'f1_macro'] = statistics.stdev(f1)
    summary.loc['mean', 'precision_macro'] = statistics.mean(precision)
    summary.loc['std', 'precision_macro'] = statistics.stdev(precision)
    summary.loc['mean', 'recall_macro'] = statistics.mean(recall)
    summary.loc['std', 'recall_macro'] = statistics.stdev(recall)
    summary.loc['mean', 'auc'] = statistics.mean(aucs_test)
    summary.loc['std', 'auc'] = statistics.stdev(aucs_test)
    summary.loc['mean', 'mcc'] = statistics.mean(mcc_test)
    summary.loc['std', 'mcc'] = statistics.stdev(mcc_test)
    summary.loc['mean', 'threshold'] = statistics.mean(thresholds_selected_test)
    summary.loc['std', 'threshold'] = statistics.stdev(thresholds_selected_test)
    f1_average = np.average(f1, weights=support)
    precision_average = np.average(precision, weights=support)
    recall_average = np.average(recall, weights=support)
    aucs_average = np.average(aucs_test, weights=support)
    mcc_average = np.average(mcc_test, weights=support)
    thr_average = np.average(thresholds_selected_test, weights=support)
    accuracy_average = np.average(accuracy, weights=support)
    summary.loc['weighted-mean', :] = [f1_average, precision_average, recall_average, accuracy_average,
                                       aucs_average, mcc_average, thr_average]
    summary.loc['weighted-std', :] = [math.sqrt(weithgted_var(f1, f1_average, support)),
                                      math.sqrt(weithgted_var(precision, precision_average, support)),
                                      math.sqrt(weithgted_var(recall, recall_average, support)),
                                      math.sqrt(weithgted_var(accuracy, accuracy_average, support)),
                                      math.sqrt(weithgted_var(aucs_test, aucs_average, support)),
                                      math.sqrt(weithgted_var(mcc_test, mcc_average, support)),
                                      math.sqrt(weithgted_var(thresholds_selected_test, thr_average, support))]



    summary.to_csv('log/temp/'+save_name_score+'_score.txt', sep="\t")
    print(summary)



def extract_info(s,level,cv,hidden, epochs, re_epochs, learning,f_fixed_threshold):
    data = pd.read_csv('metadata/'+str(level)+'_Species_antibiotic_FineQuality.csv', index_col=0, dtype={'genome_id': object},
                       sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    data = data.loc[s, :]
    df_species = data.index.tolist()
    # antibiotics = data['modelling antibiotics'].tolist()
    print(data)

    for species in df_species:
        # logDir = os.path.join('log/results/'+str(level)+'/'+ str(species.replace(" ", "_")))
        # if not os.path.exists(logDir):
        #     try:
        #         os.makedirs(logDir)
        #     except OSError:
        #         print("Can't create logging directory:", logDir)


        antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
        for anti in antibiotics:
            save_name_score = amr_utility.name_utility.name_multi_bench_save_name_score(species, anti, level,
                                                                                        learning, epochs, f_fixed_threshold)

            score = pickle.load(open('log/temp/' + save_name_score + '_all_score.pickle', "rb"))
            aucs_test=score[4]
            score_report_test=score[3]
            mcc_test=score[2]
            thresholds_selected_test=score[0]
            score_summary(cv, score_report_test, aucs_test, mcc_test, save_name_score, thresholds_selected_test)
        #put out final table with scores:'f1-score','precision', 'recall','accuracy'
        make_visualization(species, antibiotics,level,f_fixed_threshold, epochs,learning)



if __name__== '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-cv", "--cv_number", default=10, type=int,
                        help='CV splits number')
    parser.add_argument("-d", "--hidden", default=200, type=int,
                        help='dimension of hidden layer')
    parser.add_argument("-e", "--epochs", default=1000, type=int,
                        help='epochs')
    parser.add_argument("-re_e", "--re_epochs", default=500, type=int,
                        help='epochs')
    parser.add_argument("-learning", "--learning", default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('-l', '--level', default='loose', type=str,
                        help='Quality control: strict or loose')
    parser.add_argument('-f_fixed_threshold', '--f_fixed_threshold', dest='f_fixed_threshold', action='store_true',
                        help='set a fixed threshod:0.5.')

    # parser.add_argument("-o","--output", default=None, type=str, required=True,
	# 					help='Output file names')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
            \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
            \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    # parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parsedArgs = parser.parse_args()
    # parser.print_help()
    # print(parsedArgs)
    extract_info(parsedArgs.species,parsedArgs.level,parsedArgs.cv_number,parsedArgs.hidden,parsedArgs.epochs,
                 parsedArgs.re_epochs,parsedArgs.learning,parsedArgs.f_fixed_threshold)

