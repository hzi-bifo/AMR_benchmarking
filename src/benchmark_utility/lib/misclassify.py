import sys
import os
sys.path.insert(0, os.getcwd())
import argparse
import pickle,json
from src.cv_folds import name2index
from src.amr_utility import name_utility, file_utility, load_data
import numpy as np
import pandas as pd
from collections import Counter

'''
analyze misclassified genomes
log: 7 Sep 2023: modifying due to classifier selection update.
'''

def get_genomes(softwareName,level, species, i_anti,anti,cv,fscore,temp_path,f_kma,f_phylotree,output_path):
    '''only for S2G2P, pts'''
    # Correctly predicted genome name list.


    _,save_name_final = name_utility.GETname_result(softwareName, species, fscore,f_kma,f_phylotree,'',output_path)
    with open(save_name_final + '_classifier.json') as f:
        classifier_selection = json.load(f)  ## 7 sep 2023
    chosen_cl=classifier_selection[i_anti]
    correct=[]
    all=[]
    for outer_cv in range(cv):
        # the classifier for each iteration of outer loop is selected based on inner loop CV. information extracted in result_analysis.py.
        chosen_cl_cv=chosen_cl[outer_cv]
        _,_ ,save_name_score= name_utility.GETname_model(softwareName,level, species, anti,chosen_cl_cv,temp_path)
        with open(save_name_score + '_KMA_' + str(f_kma) + '_Tree_' + str(f_phylotree) + '.json') as f:
            score = json.load(f)
        All_samples=score['samples']
        predictY_test=score['predictY_test']
        true_Y=score['ture_Y']

        j_genome=0
        for each in true_Y[outer_cv]:
            if predictY_test[outer_cv][j_genome] == each:
                correct.append(All_samples[outer_cv][j_genome])
            all.append(All_samples[outer_cv][j_genome])
            j_genome+=1
    correct=['iso_'+ a for a in correct]
    all=['iso_'+ a for a in all]

    return correct,all


def get_genomes_AA(species, anti,cv,temp_path,f_kma,f_phylotree):
    '''
    only for 'Aytan-Aktug': predictY_test format slightly different.
    all: samples involved for this scenario, this drug, this species.
    correct: samples being correctly predicted.
    '''
    learning=0.0
    epochs=0
    f_fixed_threshold=True
    f_nn_base=False
    f_optimize_score='f1_macro'
    save_name_score,_,_ =  name_utility.GETname_AAscoreSSSA('AytanAktug',species, anti,learning,\
                                                                         epochs,f_fixed_threshold,f_nn_base,\
                                                           f_optimize_score,temp_path,f_kma,f_phylotree)

    with open(save_name_score) as f:
        score = json.load(f)
    All_samples=score['samples']
    predictY_test=score['predictY_test']
    true_Y=score['ture_Y']

    # Correctly predicted genome name list.
    correct=[]
    all=[]
    for i_cv in range(cv):
        j_genome=0
        for each in true_Y[i_cv]:
            if predictY_test[i_cv][j_genome][0] == each:
                correct.append(All_samples[i_cv][j_genome])
            all.append(All_samples[i_cv][j_genome])
            j_genome+=1
    correct=['iso_'+ a for a in correct]
    all=['iso_'+ a for a in all]

    return correct,all


def get_genomes_kover(species,level,i_anti, anti,cv,fscore,temp_path,f_kma,f_phylotree,output_path):

    _,save_name_final = name_utility.GETname_result('kover', species, fscore,f_kma,f_phylotree,'',output_path)
    with open(save_name_final + '_classifier.json') as f:
        classifier_selection = json.load(f)  ## 7 sep 2023

    chosen_cl=classifier_selection[i_anti]
    correct=[]
    wrong=[]
    all=[]
    for outer_cv in range(cv):

        chosen_cl_cv=chosen_cl[outer_cv]
        _,_,meta_txt,_ = name_utility.GETname_model2('kover',level, species, anti,'',temp_path,f_kma,f_phylotree)
        with open(meta_txt+'_temp/'+str(chosen_cl_cv)+'_b_'+str(outer_cv)+'/results.json') as f:
            data = json.load(f)

        test_errors_list=data["classifications"]['test_errors']
        test_corrects_list=data["classifications"]['test_correct']
        correct=correct+test_errors_list
        wrong=wrong+test_corrects_list
    all=all+correct+wrong
    return correct, all



def generate_annotate_file(df_species,tool_list,cv,level,foldset,fscore,temp_path,output_path,annotation_name):
    for species in df_species :
        OCCUR=[] #  The number of species-antibiotic combinations a genome is in. For each specific species.
        MISCLASSIFY=[]# The number of occurring >1 may indicate being predicted wrongly in more than two methods or more than two antibiotics.

        for softwareName in tool_list:

            print(species,'----',softwareName,'----')
            antibiotics, ID, _ =  load_data.extract_info(species, False, level)
            if softwareName=='Aytan-Aktug':
                ##### MISCLASSIFY_allAnti=[] #note: the occurrence of each genome indicates its count for misclassification.
                OCCUR_allAnti=[] # the occurrence of each genome indicates its count for being involved in species-antibiotic combinations.
                for anti in antibiotics:

                    f_phylotree=False
                    f_kma=False
                    SampleName_correct_Random,occur=get_genomes_AA(species, anti,cv,temp_path,f_kma,f_phylotree)

                    f_phylotree=False
                    f_kma=True
                    SampleName_correct_Phylo,_=get_genomes_AA(species, anti,cv,temp_path,f_kma,f_phylotree)

                    f_phylotree=True
                    f_kma=False
                    SampleName_correct_KMA,_=get_genomes_AA(species, anti,cv,temp_path,f_kma,f_phylotree)

                    ### list out genomes that are correctly predicted by random folds, while woringly predicted by the other two.
                    for each in SampleName_correct_Random:
                        if (each not in SampleName_correct_KMA) and (each not in SampleName_correct_Phylo):
                            MISCLASSIFY.append(each)

                    OCCUR_allAnti=OCCUR_allAnti+occur
                OCCUR=OCCUR+OCCUR_allAnti
            elif softwareName=='kover':
                OCCUR_allAnti=[] # the occurrence of each genome indicates its count for being involved in species-antibiotic combinations.
                i_anti=0
                for anti in antibiotics:

                    f_phylotree=False
                    f_kma=False
                    SampleName_correct_Random,occur=get_genomes_kover(species,level,i_anti, anti,cv,fscore,temp_path,f_kma,f_phylotree,output_path)

                    f_phylotree=False
                    f_kma=True
                    SampleName_correct_Phylo,_=get_genomes_kover(species,level,i_anti, anti,cv,fscore,temp_path,f_kma,f_phylotree,output_path)

                    f_phylotree=True
                    f_kma=False
                    SampleName_correct_KMA,_=get_genomes_kover(species,level,i_anti, anti,cv,fscore,temp_path,f_kma,f_phylotree,output_path)
                    i_anti+=1
                    ### list out genomes that are correctly predicted by random folds, while woringly predicted by the other two.
                    for each in SampleName_correct_Random:
                        if (each not in SampleName_correct_KMA) and (each not in SampleName_correct_Phylo):
                            MISCLASSIFY.append(each)
                    # print(len(occur))
                    OCCUR_allAnti=OCCUR_allAnti+occur
                OCCUR=OCCUR+OCCUR_allAnti
            else: ### S2G2P, PTS.
                OCCUR_allAnti=[]
                i_anti=0
                for anti in antibiotics:
                    f_phylotree=False
                    f_kma=False
                    SampleName_correct_Random,occur=get_genomes(softwareName,level, species, i_anti,anti,cv,fscore,temp_path,f_kma,f_phylotree,output_path)

                    f_phylotree=False
                    f_kma=True
                    SampleName_correct_Phylo,_=get_genomes(softwareName,level, species, i_anti,anti,cv,fscore,temp_path,f_kma,f_phylotree,output_path)

                    f_phylotree=True
                    f_kma=False
                    SampleName_correct_KMA,_=get_genomes(softwareName,level, species, i_anti,anti,cv,fscore,temp_path,f_kma,f_phylotree,output_path)
                    i_anti+=1
                    ### list out genomes that are correctly predicted by random folds, while woringly predicted by the other two.
                    for each in SampleName_correct_Random:
                        if (each not in SampleName_correct_KMA) and (each not in SampleName_correct_Phylo):
                            MISCLASSIFY.append(each)
                    OCCUR_allAnti=OCCUR_allAnti+occur

                OCCUR=OCCUR+OCCUR_allAnti

        # print(len(MISCLASSIFY))
        mis_unique=list(dict.fromkeys(MISCLASSIFY))
        MISCLASSIFY_dic=Counter(MISCLASSIFY)
        print('MISCLASSIFY_dic: ',len(MISCLASSIFY_dic))
        # print(MISCLASSIFY_dic)
        OCCUR_dic=Counter(OCCUR)
        print('OCCUR',len(OCCUR_dic))
        # print(OCCUR_dic)

        if len(MISCLASSIFY_dic)>0:
            # print out the max occurring, just for double checking.
            print(max(MISCLASSIFY_dic.values()) )#E.coli 31, Sa: 36
        print('max(OCCUR_dic.values())',max(OCCUR_dic.values()) )#11,11


        save_file_name=output_path+ 'Results/other_figures_tables/MisclassifiedGenomes_'+ annotation_name+'/'+str(species.replace(" ", "_"))
        file_utility.make_dir(os.path.dirname(save_file_name))
        with open(save_file_name+ '.json','w') as f:  # overwrite mode
            json.dump(MISCLASSIFY_dic, f)
        with open(save_file_name+ '_Alloccur.json','w') as f:  # overwrite mode
            json.dump(OCCUR_dic, f)



        f = open(save_file_name+"_Alloccur.txt", "w")
        for i in MISCLASSIFY_dic:
            f.write(i+','+ str(100 * MISCLASSIFY_dic[i] / OCCUR_dic[i]) +','+ str(OCCUR_dic[i]/len(tool_list))+'\n')
        f.close()

        f = open(save_file_name+"_ratio.txt", "w") ### for all 4 methods misclassification analysis, this ratio averaged all the 4 methods.
        for i in MISCLASSIFY_dic:
            r=  MISCLASSIFY_dic[i] / OCCUR_dic[i]   ###(MISCLASSIFY_dic[i]/len(tool_list)) / (OCCUR_dic[i]/len(tool_list))
            r=str(round(r, 2))
            f.write(i+','+ r +'\n')
        f.close()



def extract_info(s,f_all,level,cv,fscore,temp_path,output_path):


    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    if f_all == False:
        data = data.loc[s, :]
    df_species = data.index.tolist()



    foldset=['Random folds', 'Phylogeny-aware folds','Homology-aware folds']
    tool_list=[ 'seq2geno','phenotypeseeker','Aytan-Aktug','kover']
    generate_annotate_file(df_species,tool_list,cv,level,foldset,fscore,temp_path,output_path,'4methods')
    #
    tool_list=['seq2geno']
    generate_annotate_file(df_species,tool_list,cv,level,foldset,fscore,temp_path,output_path,'s2g2p')

    tool_list=['phenotypeseeker']
    generate_annotate_file(df_species,tool_list,cv,level,foldset,fscore,temp_path,output_path,'pts')

    tool_list=['kover']
    generate_annotate_file(df_species,tool_list,cv,level,foldset,fscore,temp_path,output_path,'kover')

    tool_list=['Aytan-Aktug']
    generate_annotate_file(df_species,tool_list,cv,level,foldset,fscore,temp_path,output_path,'AA')



















if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                        help='Directory to store temporary files.')
    parser.add_argument('-o', '--output_path', default='./', type=str, required=False,
                        help='Directory to store CV scores.')
    parser.add_argument('-l', '--level', default='loose', type=str,
                        help='Quality control: strict or loose')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument("-cv", "--cv", default=10, type=int,
                        help='CV splits number')
    parser.add_argument('-fscore', '--fscore', default='f1_macro', type=str, required=False,
                        help='the score used to choose the best classifier for each antibiotic. \
                        Can be one of: \'f1_macro\',\'f1_positive\',\'f1_negative\',\'accuracy\','
                             '\'clinical_f1_negative\',\'clinical_precision_neg\',\'clinical_recall_neg\'')

    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'Pseudomonas aeruginosa\' \
                    \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
                    \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')

    parsedArgs = parser.parse_args()
    extract_info(parsedArgs.species, parsedArgs.f_all,parsedArgs.level,parsedArgs.cv, parsedArgs.fscore,parsedArgs.temp_path,parsedArgs.output_path)
