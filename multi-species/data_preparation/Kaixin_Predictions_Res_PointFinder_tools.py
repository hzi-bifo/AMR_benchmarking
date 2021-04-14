
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
from subprocess import PIPE, run
import pandas as pd
import numpy as np
import ast
from os import walk
from itertools import repeat
from collections import defaultdict
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import argparse

def determination(species,antibiotics,level,tool):
    path_to_pointfinder="/net/flashtest/scratch/khu/benchmarking/Results/Point_results_"+str(species.replace(" ", "_"))
    path_to_resfinder="/net/flashtest/scratch/khu/benchmarking/Results/Res_results_"+str(species.replace(" ", "_"))
    path_to_pr="/net/flashtest/scratch/khu/benchmarking/Results/" + str(species.replace(" ", "_"))
    print(species)
    antibiotics_selected = ast.literal_eval(antibiotics)

    print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)

    for anti in antibiotics_selected:
        print(anti,'--------------------------------------------------------------------')
    # for anti in ['aztreonam']:
        # save_name_meta, save_name_modelID = amr_utility.name_utility.save_name_modelID(level, species, anti)
        save_name_modelID = 'metadata/model/' + str(level) + '/Data_' + str(species.replace(" ", "_")) + '_' + str(
            anti.translate(str.maketrans({'/': '_', ' ': '_'})))

        data_sub_anti = pd.read_csv(save_name_modelID + '.txt', index_col=0, dtype={'genome_id': object}, sep="\t")
        # save_name_speciesID = 'metadata/model/id_' + str(species.replace(" ", "_")) + '.list'
        # data_sub_anti = pd.read_csv(data_sub_anti , index_col=0, dtype={'genome_id': object}, sep="\t")
        data_sub_anti = data_sub_anti.drop_duplicates()
        y_pre =list()
        samples=data_sub_anti['genome_id'].to_list()

        for strain_ID in samples:

            lines_start_read = 16
            if tool=="point":
                point = open("%s/%s/pheno_table.txt" % (path_to_pointfinder, strain_ID), "r")
                file=point
            elif tool=="res":
                res = open("%s/%s/pheno_table.txt" % (path_to_resfinder, strain_ID), "r")
                file=res
            else:
                both = open("%s/%s/pheno_table.txt" % (path_to_pr, strain_ID), "r")
                file = both
                #only applied for this time
                #check the file exsistence
                # _, _, f_check = next(walk("%s/%s/pheno_table.txt" % (path_to_pr, strain_ID)))
                # if "pheno_table.txt" in f_check:
                #
                #     both = open("%s/%s/pheno_table.txt" % (path_to_pr, strain_ID), "r")
                #     file = both
                # else:
                #     path_data = "/net/projects/BIFO/patric_genome/"
                #     run_Res(path_data, strain_ID, species)
                #     both = open("%s/%s/pheno_table.txt" % (path_to_pr, strain_ID), "r")
                #     file = both

            for position, line in enumerate(file):

                if position >= lines_start_read:
                    try:
                        if anti == line.split()[0]:
                            # print(line)
                            # print(line.split()[2])
                            if line.split()[2] == "No":

                                y_pre.append(0)

                            else:
                                y_pre.append(1)
                    except:
                        pass

            # to_exclude = np.arange(0,16)
            # line= pd.read_csv(point,sep="\t", skiprows = to_exclude)
            # print(line.columns)

        if len(y_pre)>0:
            print('y_pre',y_pre)



            y = data_sub_anti['resistant_phenotype'].to_numpy()
            y=y

            print(len(y))
            print("mcc results")
            print(matthews_corrcoef(y, y_pre))



            ###confussion matrix
            tn, fp, fn, tp = confusion_matrix(y, y_pre).ravel()

            print("sensitivity", float(tp) / (tp + fn))
            print("specificity", float(tn) / (tn + fp))
            report=classification_report(y, y_pre, labels=[0, 1], output_dict=True)
            print(report)

            correct_results2 = []
            for each in y:
                correct_results2.append(int(each))

            prediction_results2 = []
            for each in y_pre:
                prediction_results2.append(int(each))

            print("f1_score", f1_score(correct_results2, prediction_results2))
            df = pd.DataFrame(report).transpose()
            df.to_csv('Results/summary/'+str(species.replace(" ", "_")) + '_' + str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'.txt', sep="\t")
        else:
            print("No information for antibiotic: ", anti)

def make_visualization(species,antibiotics,level,tool,score):
    '''
    make final summary
    :return:
    '''
    # path_to_pointfinder = "Results/Point_results" + str(species.replace(" ", "_")) + "/"
    # path_to_resfinder = "Results/Res_results_" + str(species.replace(" ", "_")) + "/"
    # path_to_pr = "Results/" + str(species.replace(" ", "_")) + "/"
    print(species)
    antibiotics_selected = ast.literal_eval(antibiotics)

    print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)
    final=pd.DataFrame(index=['f1-score','precision', 'recall'], columns= antibiotics_selected)
    print(final)
    for anti in antibiotics_selected:
        print(anti, '--------------------------------------------------------------------')
        try:
            data = pd.read_csv('Results/summary/'+str(species.replace(" ", "_")) + '_' + str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'.txt', sep="\t")
            print(data)
            final.loc['f1-score',str(anti)]=data.iloc[3,3]
            final.loc['precision',str(anti)] = data.iloc[3, 1]
            final.loc['recall',str(anti)] = data.iloc[3, 2]
            print(final)
        except:
            pass
    final=final.astype(float).round(2)
    final.to_csv('Results/summary/' + str(species.replace(" ", "_")) + '.csv', sep="\t")

def extract_info(s,l,tool,visualize,score):

    data = pd.read_csv('metadata/'+str(l)+'_Species_antibiotic_FineQuality.csv', index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]# drop the species with 0 in column 'number'.
    # for training model on part of the dataset:-------------
    # data=data.loc[['Escherichia coli'],:]
    data = data.loc[s, :]
    #data.at['Mycobacterium tuberculosis', 'modelling antibiotics']=['capreomycin', 'ciprofloxacin']
    # --------------------------------------------------------
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    print(data)
    if visualize == False:
        for df_species,antibiotics in zip(df_species, antibiotics):
            determination(df_species,antibiotics,l,tool)
    else:
        for df_species,antibiotics in zip(df_species, antibiotics):
            make_visualization(df_species,antibiotics,l,tool,score)




if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--l','--level',default=None, type=str, required=True,
                        help='Quality control: strict or loose')

    parser.add_argument('--t', '--tool', default='Both', type=str, required=True,
                        help='res, point, both')
    parser.add_argument('--s','--species', default=[], type=str,nargs='+',help='species to run: e.g.\'seudomonas aeruginosa\' \
     \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
     \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('-v', '--visualize', dest='v',
                        help='visualize the final outcome ', action='store_true', )
    parser.add_argument('--score', default='f1-score', type=str, required=False,
                        help='Score:f1-score, precision, recall, all. All scores are macro.')
    #parser.set_defaults(canonical=True)
    parsedArgs=parser.parse_args()
    # parser.print_help()
    print(parsedArgs)
    extract_info(parsedArgs.s, parsedArgs.l,parsedArgs.t,parsedArgs.v,parsedArgs.score)
    # extract_info(parsedArgs.s,parsedArgs.b,parsedArgs.l)
