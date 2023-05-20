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
'''

def get_genomes(score,cv,fold,SampleName_correct_Phylo,SampleName_wrong_Phylo,SampleName_correct_KMA,
                SampleName_wrong_KMA,SampleName_correct_Random,SampleName_wrong_Random,OCCUR,MISCLASSIFY):
    All_samples=score['samples']
    predictY_test=score['predictY_test']
    true_Y=score['ture_Y']

    # Correctly predicted genome name list.
    correct=[]
    wrong=[]
    all=[]
    for i_cv in range(cv):
        j_genome=0
        for each in true_Y[i_cv]:
            if predictY_test[i_cv][j_genome] == each:
                correct.append(All_samples[i_cv][j_genome])
            else:
                wrong.append(All_samples[i_cv][j_genome])
            all.append(All_samples[i_cv][j_genome])
            j_genome+=1
    # print(correct)
    # print(wrong)
    correct=['iso_'+ a for a in correct]
    wrong=['iso_'+ a for a in wrong]


    if fold=='Phylogeny-aware folds':
         SampleName_correct_Phylo=SampleName_correct_Phylo+correct
         SampleName_wrong_Phylo=SampleName_wrong_Phylo+wrong
    elif fold=='Homology-aware folds':
         SampleName_correct_KMA=SampleName_correct_KMA+correct
         SampleName_wrong_KMA=SampleName_wrong_KMA+wrong
    else:#'Random folds', or 'no folds'
        SampleName_correct_Random=SampleName_correct_Random+correct
        SampleName_wrong_Random=SampleName_wrong_Random+wrong
        ###OCCUR=OCCUR+all

    ### list out genomes that are correctly predicted by random folds, while woringly predicted by the other two.

    for each in SampleName_correct_Random:
        if (each not in SampleName_correct_KMA) and (each not in SampleName_correct_Phylo):
            MISCLASSIFY.append(each)
    return MISCLASSIFY


def get_genomes2(score,cv,fold,SampleName_correct_Phylo,SampleName_wrong_Phylo,SampleName_correct_KMA,
                SampleName_wrong_KMA,SampleName_correct_Random,SampleName_wrong_Random,OCCUR,MISCLASSIFY):
    All_samples=score['samples']
    predictY_test=score['predictY_test']
    true_Y=score['ture_Y']

    # Correctly predicted genome name list.
    correct=[]
    wrong=[]
    all=[]
    for i_cv in range(cv):
        j_genome=0
        for each in true_Y[i_cv]:
            if predictY_test[i_cv][j_genome][0] == each:
                correct.append(All_samples[i_cv][j_genome])
            else:
                wrong.append(All_samples[i_cv][j_genome])
            all.append(All_samples[i_cv][j_genome])
            j_genome+=1
    correct=['iso_'+ a for a in correct]
    wrong=['iso_'+ a for a in wrong]
    all=['iso_'+ a for a in all]

    if fold=='Phylogeny-aware folds':
         SampleName_correct_Phylo=SampleName_correct_Phylo+correct
         SampleName_wrong_Phylo=SampleName_wrong_Phylo+wrong
    elif fold=='Homology-aware folds':
         SampleName_correct_KMA=SampleName_correct_KMA+correct
         SampleName_wrong_KMA=SampleName_wrong_KMA+wrong
    else:#'Random folds', or 'no folds'
        SampleName_correct_Random=SampleName_correct_Random+correct
        SampleName_wrong_Random=SampleName_wrong_Random+wrong
        OCCUR=OCCUR+all

    ### list out genomes that are correctly predicted by random folds, while woringly predicted by the other two.

    for each in SampleName_correct_Random:
        if (each not in SampleName_correct_KMA) and (each not in SampleName_correct_Phylo):
            MISCLASSIFY.append(each)
    return MISCLASSIFY,OCCUR

def extract_info(s,f_all,level,cv,fscore,temp_path,output_path):


    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
    if f_all == False:
        data = data.loc[s, :]
    df_species = data.index.tolist()
    ### antibiotics = data['modelling antibiotics'].tolist()




    foldset=['Random folds', 'Phylogeny-aware folds','Homology-aware folds']
    tool_list=[ 'seq2geno','phenotypeseeker','Aytan-Aktug','kover']
    # tool_list=['seq2geno']




    for species in df_species :
        OCCUR=[] #  The number of species-antibiotic combinations a genome is in. For each specific species.
        MISCLASSIFY=[]# The number of occurring >1 may indicate being predicted wrongly in more than two methods or more than two antibiotics.
        if 'Aytan-Aktug' not in tool_list: # only for the sake of provide the OCCUR
            f_phylotree=False
            f_kma=False
            fold='Random folds'
            MISCLASSIFY_=[]

            antibiotics, ID, _ =  load_data.extract_info(species, False, level)
            learning=0.0
            epochs=0
            f_fixed_threshold=True
            f_nn_base=False
            f_optimize_score='f1_macro'
            ##1. random folds
            ### f_kma,f_phylotree=False,False
            i=0
            for anti in antibiotics:
                SampleName_correct_Random=[]
                SampleName_correct_Phylo=[]
                SampleName_correct_KMA=[]
                SampleName_wrong_Random=[]
                SampleName_wrong_Phylo=[]
                SampleName_wrong_KMA=[]
                save_name_score,_,_ =  name_utility.GETname_AAscoreSSSA('AytanAktug',species, anti,learning,\
                                                                         epochs,f_fixed_threshold,f_nn_base,\
                                                           f_optimize_score,temp_path,f_kma,f_phylotree)


                with open(save_name_score) as f:
                    score = json.load(f)
                _,OCCUR=get_genomes2(score,cv,fold,SampleName_correct_Phylo,SampleName_wrong_Phylo,SampleName_correct_KMA,
                        SampleName_wrong_KMA,SampleName_correct_Random,SampleName_wrong_Random,OCCUR,MISCLASSIFY_)







        for softwareName in tool_list:

            for fold in foldset:
                if fold=='Phylogeny-aware folds':
                    f_phylotree=True
                    f_kma=False
                    # fscore_format= fscore
                elif fold=='Homology-aware folds':
                    f_phylotree=False
                    f_kma=True
                    # fscore_format="weighted-"+fscore #only for Aytan-Aktug SSSA
                else:#'Random folds', or 'no folds'
                    f_phylotree=False
                    f_kma=False
                    # fscore_format= fscore

                print(species,'----',softwareName,'----', fold)
                antibiotics, ID, _ =  load_data.extract_info(species, False, level)
                if softwareName=='Aytan-Aktug':
                    learning=0.0
                    epochs=0
                    f_fixed_threshold=True
                    f_nn_base=False
                    f_optimize_score='f1_macro'
                    ##1. random folds
                    ### f_kma,f_phylotree=False,False
                    i=0
                    for anti in antibiotics:
                        SampleName_correct_Random=[]
                        SampleName_correct_Phylo=[]
                        SampleName_correct_KMA=[]
                        SampleName_wrong_Random=[]
                        SampleName_wrong_Phylo=[]
                        SampleName_wrong_KMA=[]
                        save_name_score,_,_ =  name_utility.GETname_AAscoreSSSA('AytanAktug',species, anti,learning,\
                                                                                 epochs,f_fixed_threshold,f_nn_base,\
                                                                   f_optimize_score,temp_path,f_kma,f_phylotree)


                        with open(save_name_score) as f:
                            score = json.load(f)

                        MISCLASSIFY,OCCUR=get_genomes2(score,cv,fold,SampleName_correct_Phylo,SampleName_wrong_Phylo,SampleName_correct_KMA,
                                SampleName_wrong_KMA,SampleName_correct_Random,SampleName_wrong_Random,OCCUR,MISCLASSIFY)



                elif softwareName=='kover':
                    _,save_name_final = name_utility.GETname_result(softwareName, species, fscore,f_kma,f_phylotree,'',output_path)
                    results=pd.read_csv(save_name_final + '_SummaryBenchmarking.txt', sep="\t",index_col=0)
                    chosen_classifier=results['classifier'].tolist()


                    i=0
                    for anti in antibiotics:
                        SampleName_correct_Random=[]
                        SampleName_correct_Phylo=[]
                        SampleName_correct_KMA=[]
                        SampleName_wrong_Random=[]
                        SampleName_wrong_Phylo=[]
                        SampleName_wrong_KMA=[]

                        chosen_cl=chosen_classifier[i]
                        _,_,meta_txt,_ = name_utility.GETname_model2('kover',level, species, anti,'',temp_path,f_kma,f_phylotree)

                        correct=[]
                        wrong=[]
                        for outer_cv in range(cv):
                            with open(meta_txt+'_temp/'+str(chosen_cl)+'_b_'+str(outer_cv)+'/results.json') as f:
                                data = json.load(f)

                            test_errors_list=data["classifications"]['test_errors']
                            test_corrects_list=data["classifications"]['test_correct']
                            correct=correct+test_errors_list
                            wrong=wrong+test_corrects_list



                        if fold=='Phylogeny-aware folds':
                             SampleName_correct_Phylo=SampleName_correct_Phylo+correct
                             SampleName_wrong_Phylo=SampleName_wrong_Phylo+wrong
                        elif fold=='Homology-aware folds':
                             SampleName_correct_KMA=SampleName_correct_KMA+correct
                             SampleName_wrong_KMA=SampleName_wrong_KMA+wrong
                        else:#'Random folds', or 'no folds'
                            SampleName_correct_Random=SampleName_correct_Random+correct
                            SampleName_wrong_Random=SampleName_wrong_Random+wrong

                        ### list out genomes that are correctly predicted by random folds, while woringly predicted by the other two.

                        for each in SampleName_correct_Random:
                            if (each not in SampleName_correct_KMA) and (each not in SampleName_correct_Phylo):
                                MISCLASSIFY.append(each)


                else:


                    ##1. random folds
                    ### f_kma,f_phylotree=False,False
                    _,save_name_final = name_utility.GETname_result(softwareName, species, fscore,f_kma,f_phylotree,'',output_path)
                    results=pd.read_csv(save_name_final + '_SummaryBenchmarking.txt', sep="\t",index_col=0)
                    chosen_classifier=results['classifier'].tolist()


                    i=0
                    for anti in antibiotics:
                        SampleName_correct_Random=[]
                        SampleName_correct_Phylo=[]
                        SampleName_correct_KMA=[]
                        SampleName_wrong_Random=[]
                        SampleName_wrong_Phylo=[]
                        SampleName_wrong_KMA=[]
                        chosen_cl=chosen_classifier[i]
                        i+=1
                        _,_ ,save_name_score= name_utility.GETname_model(softwareName,level, species, anti,chosen_cl,temp_path)
                        with open(save_name_score + '_KMA_' + str(f_kma) + '_Tree_' + str(f_phylotree) + '.json') as f:
                            score = json.load(f)
                        MISCLASSIFY=get_genomes(score,cv,fold,SampleName_correct_Phylo,SampleName_wrong_Phylo,SampleName_correct_KMA,
                                SampleName_wrong_KMA,SampleName_correct_Random,SampleName_wrong_Random,OCCUR,MISCLASSIFY)





        # print(len(MISCLASSIFY))
        mis_unique=list(dict.fromkeys(MISCLASSIFY))
        MISCLASSIFY_dic=Counter(MISCLASSIFY)
        print(len(MISCLASSIFY_dic))
        print(MISCLASSIFY_dic)
        OCCUR_dic=Counter(OCCUR)
        print(len(OCCUR_dic))
        print(OCCUR_dic)


        # print out the max occurring
        print(max(MISCLASSIFY_dic.values()) )#E.coli 31, Sa: 36
        print(max(OCCUR_dic.values())  )#11,11


        save_file_name=output_path+ 'Results/other_figures_tables/MisclassifiedGenomes/'+ str(species.replace(" ", "_"))
        file_utility.make_dir(os.path.dirname(save_file_name))
        with open(save_file_name+ '.json','w') as f:  # overwrite mode
            json.dump(MISCLASSIFY_dic, f)
        with open(save_file_name+ '_Alloccur.json','w') as f:  # overwrite mode
            json.dump(OCCUR_dic, f)

        f = open(save_file_name+".txt", "w")
        for i in MISCLASSIFY_dic:
            f.write(i +' '+ str(MISCLASSIFY_dic[i])+'\n')
        f.close()

        f = open(save_file_name+"_Alloccur.txt", "w")
        for i in MISCLASSIFY_dic:
            f.write(i+','+ str(MISCLASSIFY_dic[i]) +','+ str(OCCUR_dic[i])+'\n')
        f.close()

        f = open(save_file_name+"_ratio.txt", "w")
        for i in MISCLASSIFY_dic:
            r= MISCLASSIFY_dic[i]/ OCCUR_dic[i]
            r=str(round(r, 2))
            f.write(i+','+ r +'\n')
        f.close()












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
