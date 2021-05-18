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
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import argparse
import zipfile

def make_dir(name):
    logDir = os.path.join(name)
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
            print("Can't create logging directory:", logDir)


def determination(species,antibiotics,level,tool):

    path_to_pr = "/net/sgi/metagenomics/data/khu/benchmarking/resfinder_results/" + str(
        species.replace(" ", "_"))
    logDir ="./temp/"
    make_dir(logDir)
    print(species)
    antibiotics_selected = ast.literal_eval(antibiotics)
    print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)
    mcc_all=[]
    for anti in antibiotics_selected:
        print(anti,'--------------------------------------------------------------------')

        save_name_modelID = 'metadata/model/' + str(level) + '/Data_' + str(species.replace(" ", "_")) + '_' + str(
            anti.translate(str.maketrans({'/': '_', ' ': '_'})))

        data_sub_anti = pd.read_csv(save_name_modelID + '.txt', index_col=0, dtype={'genome_id': object}, sep="\t")
        # save_name_speciesID = 'metadata/model/id_' + str(species.replace(" ", "_")) + '.list'
        # data_sub_anti = pd.read_csv(data_sub_anti , index_col=0, dtype={'genome_id': object}, sep="\t")
        data_sub_anti = data_sub_anti.drop_duplicates()
        y_pre =list()
        samples=data_sub_anti['genome_id'].to_list()

        for strain_ID in samples:

            # Read prediction info---------------------------------------------------------------------------
            # file = get_file(species, strain_ID, tool)
            temp_file = open("./temp/"+str(species.replace(" ", "_"))+"temp.txt", "w+")

            with zipfile.ZipFile("%s/%s.zip" % (path_to_pr, strain_ID)) as z:
                file = z.open("%s/pheno_table.txt" % strain_ID, "r")
                for position, line in enumerate(file):
                    line.decode('utf-8')
                    if "# Antimicrobial	Class" in line:
                        start = position

                    if "# WARNING:" in line:
                        end = position

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
            pheno_table =pd.read_csv("./temp/"+str(species.replace(" ", "_"))+"temp.txt", index_col=None, header=None,
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
            print('y_pre',len(y_pre))
            y = data_sub_anti['resistant_phenotype'].to_numpy()
            mcc=matthews_corrcoef(y, y_pre)
            print('y',len(y))
            print("mcc results",mcc)
            report=classification_report(y, y_pre, labels=[0, 1],output_dict=True)
            print(report)
            df = pd.DataFrame(report).transpose()
            df.to_csv('Results/summary/'+ str(level)+'/'+str(species.replace(" ", "_")) + '_' + str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'.txt', sep="\t")
            mcc_all.append(mcc)
        else:
            mcc_all.append(None)
            print("No information for antibiotic: ", anti)

        return mcc_all

def make_visualization(species,antibiotics,level,tool,score):
    '''
    make final summary
    recall of the positive class is also known as “sensitivity”; recall of the negative class is “specificity”.
    '''
    # path_to_pointfinder = "Results/Point_results" + str(species.replace(" ", "_")) + "/"
    # path_to_resfinder = "Results/Res_results_" + str(species.replace(" ", "_")) + "/"
    # path_to_pr = "Results/" + str(species.replace(" ", "_")) + "/"
    print(species)
    antibiotics_selected = ast.literal_eval(antibiotics)
    print('====> Select_antibiotic:', len(antibiotics_selected), antibiotics_selected)


    mcc_all=determination(species,antibiotics,level,tool)
    final=pd.DataFrame(index=antibiotics_selected, columns=['f1_macro','precision', 'recall','accuracy','mcc','f1_positive',
                                                            'precision_positive','recall_positive','mcc','support','support_positive'] )
    print(final)
    i=0
    for anti in antibiotics_selected:
        print(anti, '--------------------------------------------------------------------')
        try:
            data = pd.read_csv('Results/summary/'+str(species.replace(" ", "_")) + '_' +
                               str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'.txt', sep="\t")
            print(data)
            final.loc[str(anti),'f1_macro']=data.loc['macro avg','f1-score']
            final.loc[str(anti),'precision'] = data.loc['macro avg','precision']
            final.loc[str(anti),'recall'] = data.loc['macro avg','recall']
            final.loc[str(anti),'accuracy'] = data.loc['accuracy', 'f1-score']
            final.loc[str(anti), 'support'] = data.loc['macro avg','support']
            final.loc[str(anti), 'f1_positive'] = data.loc['1', 'f1-score']
            final.loc[str(anti), 'precision_positive'] = data.loc['1', 'precision']
            final.loc[str(anti), 'recall_positive'] = data.loc['1', 'recall']
            final.loc[str(anti), 'support_positive'] = data.loc['1', 'support']
            final.loc[str(anti), 'mcc'] = mcc_all[i]
        except:
            pass
        i+=1
    final=final.astype(float).round(2)
    print(final)
    final.to_csv('Results/summary/'+ str(level)+'/'+str(species.replace(" ", "_")) + '.csv', sep="\t")

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
    # if visualize == False:
    #     for df_species,antibiotics in zip(df_species, antibiotics):
    #         determination(df_species,antibiotics,l,tool)
    # else:
    for df_species,antibiotics in zip(df_species, antibiotics):
        make_visualization(df_species,antibiotics,l,tool,score)




if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--l','--level',default='loose', type=str,
                        help='Quality control: strict or loose')

    parser.add_argument('--t', '--tool', default='Both', type=str,
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
