import sys
import os
sys.path.insert(0, os.path.abspath('../'))
import amr_utility.name_utility
import amr_utility.load_data
import amr_utility.file_utility
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle,json
import pandas as pd
import seaborn as sns
from sklearn.metrics import  classification_report

''' Compare performance on Antibitoics shared by multiple species.'''

def FromReport(score_report_test,fscore):
    '''
    extract scores from saved report.
    '''

    f1=[]
    accuracy=[]
    f1_pos = []
    f1_neg = []
    f_no = []
    for i in np.arange(10):
        report = score_report_test[i]
        report=pd.DataFrame(report).transpose()
        # print(report)
        if fscore== 'f1_macro':
            if report.loc['1', 'support']==0 or report.loc['0', 'support']==0:  #  only one pheno in test folder, we don't include them for final results.
                f_no.append(i)
                print('Only one phenotype in the testing folder! This folder\'s score will not be counted w.r.t. average. ' )
            else:
                accuracy.append(report.loc['accuracy', 'f1-score'])
        elif fscore=='f1_negative':
            if report.loc['0', 'support']==0:
                f_no.append(i)
                print('Only R phenotype in the testing folder! This folder\'s score will not be counted w.r.t. average. ' )
            else:
                accuracy.append(report.iat[2,2])#no use of this score
        elif fscore=='f1_positive':
            if report.loc['1', 'support']==0:
                f_no.append(i)
                print('Only S phenotype in the testing folder! This folder\'s score will not be counted w.r.t. average. ' )
            else:
                accuracy.append(report.iat[2,2])#no use of this score
        elif fscore=='accuracy':
            accuracy.append(report.iat[2,2])#no use of this score


        f1.append(report.loc['macro avg','f1-score'])
        f1_pos.append(report.loc['1', 'f1-score'])
        f1_neg.append(report.loc['0', 'f1-score'])

    if f_no != []:
        #rm the iteration's results, where no resistance phenotype in the test folder.
        f1 = [i for j, i in enumerate(f1) if j not in f_no]
        accuracy = [i for j, i in enumerate(accuracy) if j not in f_no]
        f1_pos = [i for j, i in enumerate(f1_pos) if j not in f_no]
        f1_neg = [i for j, i in enumerate(f1_neg) if j not in f_no]
    if fscore=='f1_macro':
        score_list=f1
    elif fscore=='f1_negative':
        score_list=f1_neg
    elif fscore=='f1_positive':
        score_list=f1_pos

    return score_list




def combine_data(species_list,anti,fscore, f_phylotree, f_kma,tool_list,merge_name):
    # This function makes a matrix of all tools' results.
    #todo start from here
    # so far only f1_macro based.
    df_plot = pd.DataFrame(columns=[fscore, 'species', 'software'])
    columns_name=df_plot.columns.tolist()
    df_plot_sub = pd.DataFrame(columns=columns_name)
    print(tool_list)
    for species in species_list:
        for tool in tool_list:
            if tool=='Point-/ResFinder':
                results_file='./benchmarking2_kma/resfinder/Results/summary/loose/'+str(species.replace(" ", "_"))+'.csv'
                results=pd.read_csv(results_file, header=0, index_col=0,sep="\t")
                if anti in results.index.to_list():
                    score=results.loc[anti,fscore]
                else:
                    score=np.nan
                df_plot_sub.loc['s'] = [score,species,tool]
                df_plot = df_plot.append(df_plot_sub, sort=False)

            if tool=='Neural networks':
                save_name_score = amr_utility.name_utility.GETname_multi_bench_save_name_score(species, anti, 'loose',
                                                                                                   0.0, 0,
                                                                                                   True,
                                                                                                    False,
                                                                                                    'f1_macro')
                save_name_score='./benchmarking2_kma/'+save_name_score
                if f_phylotree:
                    score = pickle.load(open(save_name_score + '_all_score_Tree.pickle', "rb"))
                if f_kma:
                    score = pickle.load(open(save_name_score + '_all_score.pickle', "rb"))
                else:
                    score = pickle.load(open(save_name_score + '_all_score_Random.pickle', "rb"))
                score_report_test = score[3]
                score_list= FromReport(score_report_test,fscore)
                df_plot_sub = pd.DataFrame(index=range(len(score_list)),columns=columns_name)
                df_plot_sub[fscore]=score_list
                df_plot_sub['species']=[species]*len(score_list)
                df_plot_sub['software']=[tool]*len(score_list)
                df_plot = df_plot.append(df_plot_sub, sort=False)
            if tool=='Seq2Geno2Pheno':
                if species !='Mycobacterium tuberculosis':#no MT information.
                    #get chosen_cl
                    _, results_file = amr_utility.name_utility.GETsave_name_final(fscore,species, f_kma, f_phylotree, '')
                    results_file='./seq2geno/'+results_file
                    results=pd.read_csv(results_file + '_SummeryBenchmarking.txt', header=0, index_col=0,sep="\t")
                    chosen_cl=results.loc[anti,'classifier']
                    _, _, save_name_score = amr_utility.name_utility.Pts_GETname('loose', species, anti,chosen_cl)
                    save_name_score='./seq2geno/'+save_name_score
                    score = pickle.load(open(save_name_score + '_kma_' + str(f_kma) + '_tree_' + str(f_phylotree) + '.pickle',"rb"))  # todo,check
                    score_report_test = score[1]
                    score_list= FromReport(score_report_test,fscore)
                    df_plot_sub = pd.DataFrame(index=range(len(score_list)),columns=columns_name)
                    df_plot_sub[fscore]=score_list
                    df_plot_sub['species']=[species]*len(score_list)
                    df_plot_sub['software']=[tool]*len(score_list)
                    df_plot = df_plot.append(df_plot_sub, sort=False)

            if tool=='PhenotypeSeeker':
                # if species !='Mycobacterium tuberculosis':
                _, results_file = amr_utility.name_utility.GETsave_name_final(fscore,species, f_kma, f_phylotree, '')
                if f_kma:
                    results_file='./PhenotypeSeeker_Nov08/'+results_file
                elif f_phylotree:
                    results_file='./PhenotypeSeeker_tree/'+results_file
                else:
                    results_file='./PhenotypeSeeker_random/'+results_file

                results=pd.read_csv(results_file + '_SummeryBenchmarking.txt', header=0, index_col=0,sep="\t")
                chosen_cl=results.loc[anti,'classifier']
                _, _, save_name_score = amr_utility.name_utility.Pts_GETname('loose', species, anti,chosen_cl)
                if f_kma:
                    save_name_score='./PhenotypeSeeker_Nov08/'+save_name_score
                elif f_phylotree:
                    save_name_score='./PhenotypeSeeker_tree/'+save_name_score
                else:
                    save_name_score='./PhenotypeSeeker_random/'+save_name_score
                score = pickle.load(open(save_name_score + '_kma_' + str(f_kma) + '_tree_' + str(f_phylotree) + '.pickle',"rb"))  # todo,check
                score_report_test = score[1]
                score_list= FromReport(score_report_test,fscore)
                df_plot_sub = pd.DataFrame(index=range(len(score_list)),columns=columns_name)
                df_plot_sub[fscore]=score_list
                df_plot_sub['species']=[species]*len(score_list)
                df_plot_sub['software']=[tool]*len(score_list)
                df_plot = df_plot.append(df_plot_sub, sort=False)
            if tool=='Kover':

                _, results_file = amr_utility.name_utility.GETsave_name_final(fscore,species, f_kma, f_phylotree, '')
                if f_kma:
                    results_file='./kover/'+results_file
                elif f_phylotree:
                    results_file='./kover_tree/'+results_file
                else:
                    results_file='./kover_random/'+results_file
                results=pd.read_csv(results_file + '_SummeryBenchmarking.txt', header=0, index_col=0,sep="\t")
                chosen_cl=results.loc[anti,'classifier']
                f1_list,accuracy_list,f1_positive_list,f1_negative_list=[],[],[],[]


                name, meta_txt, _ = amr_utility.name_utility.Pts_GETname('loose', species, anti,'')
                if f_kma:
                    name='./kover/'+name
                    meta_txt='./kover/'+meta_txt
                elif f_phylotree:
                    name='./kover_tree/'+name
                    meta_txt='./kover_tree/'+meta_txt
                else:
                    name='./kover_random/'+name
                    meta_txt='./kover_random/'+meta_txt

                name_list = pd.read_csv(name, index_col=0, dtype={'genome_id': object}, sep="\t")
                name_list.loc[:,'ID'] = 'iso_' + name_list['genome_id'].astype(str)
                name_list2 = name_list.loc[:, ['ID', 'resistant_phenotype']]
                for outer_cv in range(10):
                    index_checking =str(species.replace(" ", "_"))+'/'+str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))
                    if fscore!='accuracy':
                        #check if in the ignore list
                        if f_kma:
                            file='./kover/'
                        elif f_phylotree:
                            file='./kover_tree/'
                        else:
                            file='./kover_random/'

                        ignore_dictionary = np.load(file+'cv_folders/loose/igore_list'+'_kma_'+str(f_kma)+'_tree_'+str(f_phylotree)+'_'+fscore+'.npy',allow_pickle='TRUE').item()
                        # print(ignore_dictionary[index_checking])
                        checking_list=ignore_dictionary[index_checking]
                    else:
                        checking_list=[]
                    if outer_cv not in checking_list:
                        with open(meta_txt+'_temp/'+str(chosen_cl)+'_b_'+str(outer_cv)+'/results.json') as f:
                            data = json.load(f)

                            test_errors_list=data["classifications"]['test_errors']
                            test_corrects_list=data["classifications"]['test_correct']




                            y_true=[]
                            y_pre=[]
                            for each in test_corrects_list:
                                p=name_list2[name_list2['ID']==each].iat[0,1]
                                # print(p)
                                y_true.append(p)
                                y_pre.append(p)
                            for each in test_errors_list:
                                p=name_list2[name_list2['ID']==each].iat[0,1]
                                if p==1:
                                    y_true.append(0)
                                    y_pre.append(1)
                                else:
                                    y_true.append(1)
                                    y_pre.append(0)

                            df=classification_report(y_true, y_pre, labels=[0, 1], output_dict=True)
                            report = pd.DataFrame(df).transpose()
                            accuracy_list.append(report.iat[2,2])#no use of this score
                            f1_list.append(report.loc['macro avg','f1-score'])
                            f1_positive_list.append(report.loc['1', 'f1-score'])
                            f1_negative_list.append(report.loc['0', 'f1-score'])


                if fscore=='f1_macro':
                    score_list=f1_list
                elif fscore=='f1_negative':
                    score_list=f1_negative_list
                elif fscore=='f1_positive':
                    score_list=f1_positive_list
                df_plot_sub = pd.DataFrame(index=range(len(score_list)),columns=columns_name)
                df_plot_sub[fscore]=score_list
                df_plot_sub['species']=[species]*len(score_list)
                df_plot_sub['software']=[tool]*len(score_list)
                df_plot = df_plot.append(df_plot_sub, sort=False)
            if tool=='Majority':
                _, _, save_name_score = amr_utility.name_utility.Pts_GETname('loose', species, anti,'majority')
                save_name_score='./majority/'+save_name_score
                score = pickle.load(open(save_name_score + '_kma_' + str(f_kma) + '_tree_' + str(f_phylotree) + '.pickle',"rb"))  # todo,check
                score_report_test = score[1]
                score_list= FromReport(score_report_test,fscore)

                df_plot_sub = pd.DataFrame(index=range(len(score_list)),columns=columns_name)
                df_plot_sub[fscore]=score_list
                df_plot_sub['species']=[species]*len(score_list)
                df_plot_sub['software']=[tool]*len(score_list)
                df_plot = df_plot.append(df_plot_sub, sort=False)
    return df_plot


def ComByAnti(level,s, fscore, cv_number, f_phylotree, f_kma,f_all):
    '''
    Plot benchmarking resutls by antibiotics. Only those antibiotics that are with data of multi-species.
    Tool:
    '''

    data = pd.read_csv('metadata/' + str(level) + '_multi-species_summary.csv', index_col=0,
                               dtype={'genome_id': object}, sep="\t")

    if f_phylotree:
        list_species=data.index.tolist()[1:-1]#MT no data.

    else:
        list_species = data.index.tolist()[:-1]
    data = data.loc[list_species, :]
    data = data.loc[:, (data != 0).any(axis=0)]
    print(data)
    merge_name = []
    for n in list_species:
        merge_name.append(n[0] + n.split(' ')[1][0])
    merge_name = '_'.join(merge_name)  # e.g.Se_Kp_Pa
    #NN_discreate multi-species.


    fig, axs = plt.subplots(5,4,figsize=(20, 25))
    # fig.subplots_adjust(top=0.88)
    plt.tight_layout(pad=6)
    fig.subplots_adjust(wspace=0.25, hspace=0.5, top=0.95, bottom=0.08)



    # fig.suptitle(title,size=20, weight='bold')
    # [axi.set_axis_off() for axi in axs.ravel()[0:9]] #remove outer frame
    blue=(0.12156862745098039, 0.4666666666666667, 0.7058823529411765)
    orange=(1.0, 0.4980392156862745, 0.054901960784313725)
    green= (0.17254901960784313, 0.6274509803921569, 0.17254901960784313)
    purp=(0.5803921568627451, 0.403921568627451, 0.7411764705882353)
    red=(0.8392156862745098, 0.15294117647058825, 0.1568627450980392)
    red='#ef3b2c'
    brown=(0.5490196078431373, 0.33725490196078434, 0.29411764705882354)
    # if fscore=='f1_macro' or fscore=='accuracy':
    # colors = [blue,"orange",  purp , green , red, '#653700']# #ffd343brown


    tool_list=['Point-/ResFinder', 'Neural networks',  'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover','Majority']
    colors = [blue,"orange", purp , green , red, brown]# #ffd343


    with open('./src/AntiAcronym_dict.pkl', 'rb') as f:
        map_acr = pickle.load(f)
    i=0
    df_s = data.T.dot(data.T.columns + ';').str.rstrip(';')#get anti names  marked with 1
    # print(df_s)
    All_antibiotics = data.columns.tolist()
    for anti in All_antibiotics:
        species=df_s[anti].split(';')
        data_plot=combine_data(species,anti,fscore, f_phylotree, f_kma,tool_list,merge_name)

        # print(data_plot)
        # print(df1)
        row = (i // 4)
        col = i % 4
        i+=1

        data_plot["species_acro"]=data_plot['species'].apply(lambda x:x[0] +". "+ x.split(' ')[1])

        # g = df.plot(ax=axs[row, col],kind="bar",color=colors, x='software',y=antibiotics)
        g = sns.barplot(x="species_acro", y=fscore, hue='software',
                        data=data_plot, dodge=True, ax=axs[row, col],palette=colors,hue_order=tool_list)
        anti=anti+'('+map_acr[anti]+')'
        g.set_title(anti, weight='bold',size=18)
        if anti in[ 'tetracycline(TE)']:
            g.set_xticklabels(g.get_xticklabels(), rotation=30,size=18, horizontalalignment='right',style='italic')
        elif anti in['gentamicin(GM)' ]:
            g.set_xticklabels(g.get_xticklabels(), rotation=20,size=18, horizontalalignment='center',style='italic')
        else:
            g.set_xticklabels(g.get_xticklabels(), rotation=10, size=18,horizontalalignment='center',style='italic')
        g.set(ylim=(0, 1.1))
        plt.yticks([0,0.2,0.4,0.6,0.8, 1])
        g.set_xlabel('')
        g.set_ylabel(fscore,size = 18)

        if i!=1:
            # handles, labels = g.get_legend_handles_labels()
            # g.legend('', '')
            g.get_legend().remove()

        else:
            handles, labels = g.get_legend_handles_labels()
            if f_kma:
                g.legend(bbox_to_anchor=(0.4,1.3), ncol=8,fontsize=18,frameon=False)
            if f_phylotree:
                g.legend(bbox_to_anchor=(4.5,1.3), ncol=8,fontsize=18,frameon=False)
            else:
                g.legend(bbox_to_anchor=(4.5,1.3), ncol=8,fontsize=18,frameon=False)
    fig.savefig('log/results/ByAnti_'+'kma_'+str(f_kma)+'_tree_'+str(f_phylotree)+'_'+fscore+'.pdf')
