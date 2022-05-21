
import pandas as pd
import ast
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
import argparse
import zipfile
import numpy as np
from scipy.stats import ttest_rel
import pickle
from src.amr_utility import name_utility, file_utility



def determination(species,antibiotics,level,f_no_zip,temp_path,output_path):
    path_to_pr=temp_path+ str(species.replace(" ", "_"))
    path_2nd_temp =temp_path+"temp/"
    file_utility.make_dir(path_2nd_temp)
    antibiotics_selected = ast.literal_eval(antibiotics)
    print(species, '====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)
    mcc_all=[]
    Y_pred = []
    for anti in antibiotics_selected:
        print(anti,'--------------------------------------------------------------------')
        save_name_modelID=name_utility.GETname_meta(species,anti,level)
        data_sub_anti = pd.read_csv(save_name_modelID + '.txt', index_col=0, dtype={'genome_id': object}, sep="\t")
        data_sub_anti = data_sub_anti.drop_duplicates()
        y_pre =list()
        samples=data_sub_anti['genome_id'].to_list()

        for strain_ID in samples:

            # Read prediction info---------------------------------------------------------------------------
            # file = get_file(species, strain_ID, tool)
            temp_file = open(path_2nd_temp+str(species.replace(" ", "_"))+"temp.txt", "w+")

            if f_no_zip==True:# ResFinder results are not in zip format
                file = open("%s/%s/pheno_table.txt" % (path_to_pr,strain_ID), "r")
                for position, line in enumerate(file):
                    if "# Antimicrobial	Class" in line:
                        start = position
                    if "# WARNING:" in line:
                        end = position
                file = open("%s/%s/pheno_table.txt" % (path_to_pr, strain_ID), "r")
                for position, line in enumerate(file):
                    try:
                        if (position > start) & (position < end):
                            temp_file.write(line)
                    except:
                        if (position > start):
                            temp_file.write(line)
            else:#zip format
                with zipfile.ZipFile("%s/%s.zip" % (path_to_pr, strain_ID)) as z:
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
                                temp_file.write(line)
                        except:
                            if (position > start):
                                temp_file.write(line)

            temp_file.close()
            pheno_table =pd.read_csv(path_2nd_temp+str(species.replace(" ", "_"))+"temp.txt", index_col=None, header=None,
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
        if len(y_pre)>0:
            # print('y_pre',len(y_pre))
            y = data_sub_anti['resistant_phenotype'].to_numpy()
            mcc=matthews_corrcoef(y, y_pre)
            # print('y',len(y))
            # print("mcc results",mcc)
            report=classification_report(y, y_pre, labels=[0, 1],output_dict=True)
            # print(report)
            df = pd.DataFrame(report).transpose()
            df.to_csv(output_path + '/' +str(species.replace(" ", "_")) +'_'+ str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'.txt', sep="\t")
            mcc_all.append(mcc)
        else:
            mcc_all.append(None)
            print("No information for antibiotic: ", anti)


        with open(output_path + '/' +str(species.replace(" ", "_")) +'_'+ str(anti.translate(str.maketrans({'/': '_', ' ': '_'}))) + '.pickle', 'wb') as f:  # overwrite
            pickle.dump(y_pre, f)

    return mcc_all

def make_visualization(species,antibiotics,level,f_no_zip,temp_path,output_path):

    '''
    make final summary
    recall of the positive class is also known as  sensitivity ; recall of the negative class is "specificity".'''

    antibiotics_selected = ast.literal_eval(antibiotics)
    # print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)
    save_name_score,_=name_utility.GETname_ResfinderResults(level,species)
    output_path=output_path+save_name_score
    file_utility.make_dir(output_path)
    mcc_all=determination(species,antibiotics,level,f_no_zip,temp_path,output_path)
    final=pd.DataFrame(index=antibiotics_selected, columns=['f1_macro','precision', 'recall','accuracy','mcc','f1_positive',
                                                            'precision_neg','recall_neg','support','support_positive','f1_negative'] )
    # print(final)
    # print(mcc_all)


    i=0
    for anti in antibiotics_selected:
        # print(anti, '--------------------------------------------------------------------')
        try:
            data = pd.read_csv(output_path+ '/' +str(species.replace(" ", "_")) +'_'+
                               str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'.txt', index_col=0, header=0,sep="\t")
            # print(data)
            final.loc[str(anti),'f1_macro']=data.loc['macro avg','f1-score']
            final.loc[str(anti),'precision'] = data.loc['macro avg','precision']
            final.loc[str(anti),'recall'] = data.loc['macro avg','recall']
            final.loc[str(anti),'accuracy'] = data.loc['accuracy', 'f1-score']
            final.loc[str(anti), 'support'] = data.loc['macro avg','support']
            final.loc[str(anti), 'f1_positive'] = data.loc['1', 'f1-score']
            final.loc[str(anti), 'f1_negative'] = data.loc['0', 'f1-score']
            final.loc[str(anti), 'precision_neg'] = data.loc['0', 'precision']
            final.loc[str(anti), 'recall_neg'] = data.loc['0', 'recall']
            final.loc[str(anti), 'support_positive'] = data.loc['1', 'support']
            final.loc[str(anti), 'mcc'] = mcc_all[i]
        except:
            pass
        i+=1
    final=final.astype(float).round(2)
    print(final)
    final.to_csv(output_path + '.csv', sep="\t")


def com_blast_kma(df_species,antibiotics,level,tool,fscore,f_no_zip,output_path):
    kma_results = []
    blast_results=[]

    for species,antibiotics in zip(df_species, antibiotics):

        if species != 'Neisseria gonorrhoeae':
            print(species)
            print('*******************')
            save_name_score,save_name_score_blastn=name_utility.GETname_ResfinderResults(level,species)
            output_pathK=output_path+save_name_score
            output_pathB=output_path+save_name_score_blastn

            # for anti in antibiotics_selected:

            df_kma = pd.read_csv(output_pathK + '.csv',index_col=0,header=0 ,sep="\t")
            df_blast = pd.read_csv(output_pathB + '.csv',index_col=0 ,header=0,sep="\t")
            data_kma=df_kma[fscore].dropna().to_list()
            data_blast=df_blast[fscore].dropna().to_list()
            print(data_kma)
            print(data_blast)
            kma_results=kma_results+data_kma
            blast_results=blast_results+data_blast
            print(len(kma_results))

    result = ttest_rel(kma_results, blast_results)# paired T-test
    pvalue = result[1]
    print('P value=',pvalue)







def extract_info(s,level,tool,fscore,f_no_zip,f_com,f_all,temp_path,output_path):
    temp_path=temp_path+'log/temp/resfinder/'
    file_utility.make_dir(temp_path)
    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    if f_all == False:
        data = data.loc[s, :]
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    # print(df_species)
    if f_com: #Compare KMA Blastn version results.
       com_blast_kma(df_species,antibiotics,level,tool,fscore,f_no_zip,output_path)
    else:
        for df_species,antibiotics in zip(df_species, antibiotics):
            make_visualization(df_species,antibiotics,level,f_no_zip,temp_path,output_path)




if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--level',default='loose', type=str,
                        help='Quality control: strict or loose')
    parser.add_argument('-t', '--tool', default='Both', type=str,
                        help='res, point, both')
    parser.add_argument('-f_com', '--f_com', dest='f_com', action='store_true',
                        help='Compare the results of balst based resfinder and kma based resfinder')
    parser.add_argument('-f_no_zip', '--f_no_zip', dest='f_no_zip', action='store_true',
                        help=' Point/ResFinder results are not stored in zip format.')
    parser.add_argument('-s','--species', default=[], type=str,nargs='+',help='species to run: e.g.\'seudomonas aeruginosa\' \
     \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
     \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')

    parser.add_argument('-fscore', '--fscore', default='f1_macro', type=str, required=False,
                        help='the score for benchmarking, and used to choose the best classifier for each antibiotic. Can be f1_pos'
                             'f1_neg, accuracy.')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                    help='The log file')
    parser.add_argument('-o', '--output_path', default='./', type=str, required=False,
                    help='Results folder.')
    parsedArgs=parser.parse_args()
    # parser.print_help()
    extract_info(parsedArgs.species, parsedArgs.level,parsedArgs.tool,parsedArgs.fscore,parsedArgs.f_no_zip,parsedArgs.f_com,parsedArgs.f_all,parsedArgs.temp_path,parsedArgs.output_path)

