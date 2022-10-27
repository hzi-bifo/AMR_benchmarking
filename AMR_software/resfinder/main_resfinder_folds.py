#!/usr/bin/env python3

import sys,os
# sys.path.append('../../')
sys.path.insert(0, os.getcwd())
from src.cv_folds import name2index
from src.amr_utility import file_utility,name_utility,load_data
from sklearn.metrics import auc,roc_curve,matthews_corrcoef,confusion_matrix,f1_score,precision_recall_fscore_support,classification_report
import zipfile,json
import argparse
import pandas as pd
import numpy as np
import random
import warnings,os,shutil
warnings.filterwarnings('ignore')


def determine(Tsamples,species,anti,f_no_zip,temp_path):
    path_temp1 =temp_path+"software_output"
    path_temp2 =temp_path+"analysis"
    file_utility.make_dir(path_temp1)
    file_utility.make_dir(path_temp2)



    y_pre=[]
    temp_number=random.randint(1, 10000)
    file_utility.make_dir(path_temp2+'/'+str(temp_number))
    for strain_ID in Tsamples:
        temp_file_name=path_temp2+'/'+str(temp_number)+'/'+str(species.replace(" ", "_"))+"temp.txt"
        temp_file = open(temp_file_name, "w+")
        if f_no_zip==True:# ResFinder results are not in zip format
            file = open("%s/%s/%s/pheno_table.txt" % (path_temp1, str(species.replace(" ", "_")),strain_ID), "r")

            for position, line in enumerate(file):
                if "# Antimicrobial	Class" in line:
                    start = position
                if "# WARNING:" in line:
                    end = position
            file = open("%s/%s/%s/pheno_table.txt" % (path_temp1,str(species.replace(" ", "_")), strain_ID), "r")

            for position, line in enumerate(file):
                try:
                    if (position > start) & (position < end):
                        # print(position)
                        # line=line.strip().split('\t')
                        temp_file.write(line)
                except:
                    if (position > start):
                        # print(position)
                        # line=line.strip().split('\t')
                        temp_file.write(line)
        else:#zip format
            with zipfile.ZipFile("%s/%s/%s.zip" % (path_temp1, str(species.replace(" ", "_")),strain_ID)) as z:
                file = z.open("%s/pheno_table.txt" % strain_ID, "r")
                for position, line in enumerate(file):
                    line=line.decode('utf-8')
                    if "# Antimicrobial	Class" in line:
                        start = position

                    if "# WARNING:" in line:
                        end = position
                file = z.open("%s/pheno_table.txt" % strain_ID, "r")
                for position, line in enumerate(file):
                    line = line.decode('utf-8')
                    try:
                        if (position > start) & (position < end) :
                            # print(position)
                            # line=line.strip().split('\t')
                            temp_file.write(line)
                    except:
                        if (position > start):
                            # print(position)
                            # line=line.strip().split('\t')
                            temp_file.write(line)

        temp_file.close()
        pheno_table =pd.read_csv(temp_file_name, index_col=None, header=None,
                                     names=['Antimicrobial', 'Class', 'WGS-predicted phenotype', 'Match', 'Genetic background'],
                                    sep="\t")

        # print(pheno_table.size)
        pheno_table_sub = pheno_table.loc[pheno_table['Antimicrobial'] == str(anti.translate(str.maketrans({'/': '+'}))), 'WGS-predicted phenotype']
        if pheno_table_sub.size>0:
            pheno=pheno_table_sub.values[0]
            # print(pheno)
            if pheno=='No resistance':
                y_pre.append(0)
            elif pheno=='Resistant':
                y_pre.append(1)
            else:
                if len(y_pre)>0:
                    print("Warning: may miss a sample regarding y_pre. ")
                    exit()
    shutil.rmtree(os.path.dirname(temp_file_name))
    return y_pre

def model(level,species,cv,f_phylotree,f_kma,f_no_zip,temp_path,temp_path_k,temp_path_b):


    antibiotics, ID, Y = load_data.extract_info(species, False, level)
    # antibiotics, ID, Y =[antibiotics[1]], [ID[1]], [Y[1]]# debugging
    print(species)
    print('====> Select_antibiotic:', len(antibiotics), antibiotics)
    i_anti=0

    for anti in antibiotics:
        print(anti)
        p_names = name_utility.GETname_meta(species,anti,level)
        folds_txt=name_utility.GETname_folds(species,anti,level,f_kma,f_phylotree)
        folders_sample = json.load(open(folds_txt, "rb"))
        folders_index=name2index.Get_index(folders_sample,p_names) # CV folds

        _, _,save_name_score=name_utility.GETname_model('resfinder_folds',level, species, anti,'resfinder',temp_path)
        file_utility.make_dir(os.path.dirname(save_name_score))

        id_all = ID[i_anti]#sample name list
        y_all = Y[i_anti]
        i_anti+=1
        id_all =np.array(id_all)
        y_all = np.array(y_all)

        mcc_test = []  # MCC results for the test data
        f1_test = []
        score_report_test = []
        aucs_test = []
        predictY_test=[]
        true_Y=[]
        sampleNames_test=[]
        for out_cv in range(cv):
            print('Starting outer CV: ', str(out_cv))
            test_samples_index = folders_index[out_cv]# a list of index

            id_test = id_all[test_samples_index]#sample name list
            y_test = y_all[test_samples_index]

            if species=='Neisseria gonorrhoeae' :
                y_pre=determine(id_test,species,anti,f_no_zip,temp_path_b)#based on Blastn version of resfinder
            else:
                y_pre=determine(id_test,species,anti,f_no_zip,temp_path_k)#based on KMA version of resfinder

            if len(y_pre)>0:
                # print('y_pre',len(y_pre))
                # y = y_test

                mcc=matthews_corrcoef(y_test, y_pre)
                f1macro=f1_score(y_test, y_pre, average='macro')
                report=classification_report(y_test, y_pre, labels=[0, 1],output_dict=True)
                # df_report = pd.DataFrame(report).transpose()
                fpr, tpr, _ = roc_curve(y_test, y_pre, pos_label=1)
                roc_auc = auc(fpr, tpr)
                # print(report)
                mcc_test.append(mcc)
                f1_test.append(f1macro)
                score_report_test.append(report)
                aucs_test.append(roc_auc)
                predictY_test.append(y_pre)
                true_Y.append(y_test.tolist())
                sampleNames_test.append(folders_sample[out_cv])
            else:
                mcc_test.append(None)
                f1_test.append(None)
                score_report_test.append(None)
                aucs_test.append(None)

                print("No information for antibiotic: ", anti)



        score ={'f1_test':f1_test,'score_report_test':score_report_test,'aucs_test':aucs_test,'mcc_test':mcc_test,
             'predictY_test':predictY_test,'ture_Y':true_Y,'samples':sampleNames_test}
            # [f1_test, score_report_test, aucs_test, mcc_test,predictY_test]

        with open(save_name_score + '_KMA_' + str(f_kma) + '_Tree_' + str(f_phylotree) + '.json',
                  'w') as f:  # overwrite mode
            json.dump(score, f)




def extract_info(level,s, cv,f_phylotree,f_kma,f_all,f_no_zip,temp_path):

    file_utility.make_dir(temp_path+'log/software/resfinder_folds/')
    temp_path_k=temp_path+'log/software/resfinder_k/'
    temp_path_b=temp_path+'log/software/resfinder_b/'

    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    if f_all == False:
        data = data.loc[s, :]
    df_species = data.index.tolist()
    # antibiotics = data['modelling antibiotics'].tolist()
    for species in df_species :
        model(level, species,cv,f_phylotree,f_kma,f_no_zip,temp_path,temp_path_k,temp_path_b)

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--level', default='loose', type=str, required=False,
                        help='Quality control: strict or loose.default=\'loose\'.')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folds.')
    parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                        help='kma based cv folds.')
    parser.add_argument("-cv", "--cv_number", default=10, type=int,
                        help='CV splits number. Default=10 ')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species.')
    parser.add_argument('-f_no_zip', '--f_no_zip', dest='f_no_zip', action='store_true',
                        help=' Point/ResFinder results are not stored in zip format.')
    # parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel. Default=1')
    parser.add_argument('-s','--species', default=[], type=str,nargs='+',help='species to run: e.g.\'seudomonas aeruginosa\' \
     \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
     \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                    help='Directory to store temporary files.')
    # parser.add_argument('-o', '--output_path', default='./', type=str, required=False,
    #                 help='Results folder.')
    parsedArgs=parser.parse_args()
    print(parsedArgs)
    extract_info(parsedArgs.level,parsedArgs.species, parsedArgs.cv_number,parsedArgs.f_phylotree,parsedArgs.f_kma,
                 parsedArgs.f_all,parsedArgs.f_no_zip,parsedArgs.temp_path)
