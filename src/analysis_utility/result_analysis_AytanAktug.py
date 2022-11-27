import sys,os
# sys.path.append('../')
sys.path.insert(0, os.getcwd())
import numpy as np
from src.amr_utility import name_utility, file_utility, load_data
from src.analysis_utility.lib import extract_score,make_table,math_utility
import argparse,json,pickle
import pandas as pd
from src.cv_folds import name2index
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score,classification_report


def l2n(arrays):
    np_arrays = []
    for array in arrays:
        np_arrays.append(np.array(array))
    return np_arrays

def j2l(score_temp):
    # transfer json socre verion to list verion
    f1_test=score_temp['f1_test']
    score_report_test=score_temp['score_report_test']
    aucs_test=score_temp['aucs_test']
    mcc_test=score_temp['mcc_test']
    thresholds_selected_test=score_temp['thresholds_selected_test']
    hyperparameters_test=score_temp['hyperparameters_test']
    actual_epoc_test=score_temp['actual_epoc_test']
    actual_epoc_test_std=score_temp['actual_epoc_test_std']

    score=[f1_test,score_report_test,aucs_test,mcc_test,thresholds_selected_test,hyperparameters_test,actual_epoc_test,actual_epoc_test_std]

    return score

def extract_OldFormat(hy_para_all,hy_para_fre,species, anti,learning, epochs,f_fixed_threshold,f_nn_base,f_optimize_score,temp_path,f_kma,f_phylotree):
    #for extracting results from our CV results stored in pickle files, which was a format not used in this project anymore.

    folder=temp_path+'log/software/AytanAktug/analysis/SSSA/'+ str(species.replace(" ", "_"))
    save_name_score=folder +'/'+str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'_lr_'+ str(learning)+'_ep_'+str(epochs)+'_fixT_'+str(f_fixed_threshold)
    if f_phylotree:
        score = pickle.load(open(save_name_score + '_all_score_Tree.pickle', "rb"))
    elif f_kma:
        score =  pickle.load(open(save_name_score + '_all_score.pickle', "rb"))
    else:
        score =pickle.load(open(save_name_score + '_all_score_Random.pickle', "rb"))
    f1macro=score[1]
    aucs_test = score[4]
    score_report_test = score[3]
    mcc_test = score[2]
    thresholds_selected_test = score[0]
    hy_para_all.append([score[6],score[7],score[8]])#1*n_cv
    #vote the most frequent used hyper-para
    hy_para_collection=score[6]#10 dimension. each reapresents one outer loop.
    common,ind= math_utility.get_most_fre_hyper(hy_para_collection)
    hy_para_fre.append(common.to_dict())
    return f1macro,aucs_test,score_report_test,mcc_test,thresholds_selected_test,hy_para_all,hy_para_fre

def extract_info_clinical_SSSA(level,species,cv,learning,epochs,f_fixed_threshold,f_nn_base,f_optimize_score,f_phylotree,f_kma, temp_path):
    antibiotics, _, _ =  load_data.extract_info(species, False, level)
    score_list=['clinical_f1_negative','clinical_precision_neg', 'clinical_recall_neg']
    summary_table_ByClassifier_all = []
    for anti in antibiotics:
        print(species,anti)
        summary_table_ByClassifier_ = pd.DataFrame(index=['value'],columns=score_list)

        save_name_score,_,_ =  name_utility.GETname_AAscoreSSSA('AytanAktug',species, anti,learning,
                                                                         epochs,f_fixed_threshold,f_nn_base,
                                                           f_optimize_score,temp_path,f_kma,f_phylotree)


        try:#new version
            with open(save_name_score) as f:
                score = json.load(f)
            score_report_test=score['score_report_test']
        except:#old version
            folder=temp_path+'log/software/AytanAktug/analysis/SSSA/'+ str(species.replace(" ", "_"))
            save_name_score=folder +'/'+str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+'_lr_'+ str(learning)+'_ep_'+str(epochs)+'_fixT_'+str(f_fixed_threshold)
            if f_phylotree:
                score = pickle.load(open(save_name_score + '_all_score_Tree.pickle', "rb"))
            elif f_kma:
                score =  pickle.load(open(save_name_score + '_all_score.pickle', "rb"))
            else:
                score =pickle.load(open(save_name_score + '_all_score_Random.pickle', "rb"))
            score_report_test = score[3]

        summary_table_ByClassifier =  extract_score.score_clinical(summary_table_ByClassifier_, cv, score_report_test)
        summary_table_ByClassifier_all.append(summary_table_ByClassifier)

    final =  make_table.make_visualization_clinical(score_list, summary_table_ByClassifier_all, antibiotics)
    return final

def extract_info_clinical_SSMA(level,species,cv,learning,epochs,f_fixed_threshold,f_nn_base,f_optimize_score,f_phylotree,f_kma, temp_path):#todo check
    antibiotics, _, _ =  load_data.extract_info(species, False, level)
    score_list=['clinical_f1_negative','clinical_precision_neg', 'clinical_recall_neg']
    summary_table_ByClassifier_all = []

    save_name_score,_, _ = name_utility.GETname_AAscoreSSMA('AytanAktug',species,learning, epochs,
                 f_fixed_threshold, f_nn_base, f_optimize_score,temp_path,f_kma,f_phylotree)

    try:#new version
        with open(save_name_score) as f:
            score = json.load(f)
        score_report_test=score['score_report_test']
    except:#old version
        folder=temp_path+'log/software/AytanAktug/analysis/SSMA/'+ str(species.replace(" ", "_"))
        save_name_score=folder +'/multiAnti_lr_'+ str(learning)+'_ep_'+str(epochs)+'_fixT_'+str(f_fixed_threshold)
        if f_phylotree:
            score = pickle.load(open(save_name_score + '_all_score_Tree.pickle', "rb"))
        elif f_kma:
            score =  pickle.load(open(save_name_score + '_all_score.pickle', "rb"))
        else:
            score =pickle.load(open(save_name_score + '_all_score_Random.pickle', "rb"))
        score_report_test = score[3]

    summary_table_ByClassifier_=pd.DataFrame(index='value', columns=score_list)
    count_anti = 0
    for anti in antibiotics:
        summary_table_ByClassifier=extract_score.score_clinical(summary_table_ByClassifier_, cv, score_report_test[count_anti])
        count_anti+=1
        summary_table_ByClassifier_all.append(summary_table_ByClassifier)
    final =  make_table.make_visualization_clinical(score_list, summary_table_ByClassifier_all, antibiotics)
    return final


def extract_info(out_score,fscore,f_SSMA,f_SSSA,f_MSMA_discrete,f_MSMA_conMix,f_MSMA_conLOO,f_split_species,f_all,f_match_single,list_species,level,cv,
                 epochs, learning,f_fixed_threshold,f_nn_base,f_phylotree,f_kma,f_optimize_score, temp_path, output_path):
    if f_SSMA:

        main_meta,_=name_utility.GETname_main_meta(level)
        data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
        data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
        if f_all:
            list_species = data.index.tolist()
            data = data.loc[list_species, :]
        else:
            data = data.loc[list_species, :]
        df_species = data.index.tolist()
        print(data)
        for species in df_species:

            antibiotics, ID, Y =  load_data.extract_info(species, False, level)
            save_name_score,_, _ = name_utility.GETname_AAscoreSSMA('AytanAktug',species,learning, epochs,
                 f_fixed_threshold, f_nn_base, f_optimize_score,temp_path,f_kma,f_phylotree)


            with open(save_name_score  + '.json') as f:
                score_temp = json.load(f)
            score=j2l(score_temp)

            save_name_score_final =  name_utility.GETname_AAresult('AytanAktug',species,learning, epochs,f_fixed_threshold,\
                                     f_nn_base,f_optimize_score,f_kma,f_phylotree,'SSMA',output_path)
            file_utility.make_dir(os.path.dirname(save_name_score_final))
            final=make_table.multi_make_visualization(fscore,antibiotics, cv,score)


            ######################################
            ##### Add clinical oriented scores ['clinical_f1_negative','clinical_precision_neg', 'clinical_recall_neg']
            ##### Nov 2022
            clinical_table=extract_info_clinical_SSMA(level,species,cv,learning,epochs,f_fixed_threshold,f_nn_base,f_phylotree,f_kma, temp_path)#todo redo
            final = pd.concat([final, clinical_table], axis=1, join="inner")
            #########################################################################################################################################


            final.to_csv(save_name_score_final + '_SummaryBenchmarking.txt', sep="\t")




    elif f_SSSA==True:
        main_meta,_=name_utility.GETname_main_meta(level)
        data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
        data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.
        if f_all:
            list_species = data.index.tolist()
            data = data.loc[list_species, :]
        else:
            data = data.loc[list_species, :]
        df_species = data.index.tolist()

        for species in df_species:

            antibiotics, _, _ =  load_data.extract_info(species, False, level)
            summary_all=[]
            hy_para_all=[]
            hy_para_fre=[]
            for anti in antibiotics:
                print(anti)
                try: #new format
                    save_name_score,_,_ =  name_utility.GETname_AAscoreSSSA('AytanAktug',species, anti,learning,
                                                                         epochs,f_fixed_threshold,f_nn_base,
                                                           f_optimize_score,temp_path,f_kma,f_phylotree)


                    with open(save_name_score) as f:
                        score_temp = json.load(f)
                    score=j2l(score_temp)
                    f1macro=score[0]
                    score_report_test = score[1]
                    aucs_test = score[2]
                    mcc_test = score[3]
                    thresholds_selected_test = score[4]
                    hy_para_all.append([score[5],score[6],score[7]])#1*n_cv
                    #vote the most frequent used hyper-para
                    hy_para_collection=score[5]#10 dimension. each reapresents one outer loop.
                    # try:
                    common,ind= math_utility.get_most_fre_hyper(hy_para_collection)
                    hy_para_fre.append(common.to_dict())
                    # except:
                    #     hy_para_fre.append(None)
                except: #old version format.

                    f1macro,aucs_test,score_report_test,mcc_test,thresholds_selected_test,hy_para_all,hy_para_fre=\
                        extract_OldFormat(hy_para_all,hy_para_fre,species, anti,learning, epochs,f_fixed_threshold,f_nn_base,f_optimize_score,temp_path,f_kma,f_phylotree)


                summary = pd.DataFrame(index=['mean', 'std', 'weighted-mean', 'weighted-std'],
                                       columns=['f1_macro', 'precision_macro', 'recall_macro', 'accuracy_macro',
                                                'mcc', 'f1_positive','f1_negative', 'precision_neg', 'recall_neg', 'auc',
                                                'threshold', 'support', 'support_positive'])

                if f_kma:
                    summary = extract_score.score_summary(None, summary, cv, score_report_test, f1macro, aucs_test,mcc_test,
                                            thresholds_selected_test)
                else:# f_phylotree or f_random
                    summary =  extract_score.score_summary_Tree(None, summary, cv, score_report_test, f1macro,aucs_test, mcc_test,
                                                     thresholds_selected_test)

                summary_all.append(summary)

            save_name_score_final =  name_utility.GETname_AAresult('AytanAktug',species,learning, epochs,f_fixed_threshold,\
                                     f_nn_base,f_optimize_score,f_kma,f_phylotree,'SSSA',output_path)
            file_utility.make_dir(os.path.dirname(save_name_score_final))
            if f_kma:#f_kma
                final,final_plot,final_std= make_table.make_visualization(out_score,summary_all,  antibiotics )
            else:# f_random, f_phylotree
                final, final_plot,final_std =  make_table.make_visualization_Tree(out_score,summary_all, antibiotics)

            final['the most frequent hyperparameter'] = hy_para_fre
            final['selected hyperparameter'] = hy_para_all # add hyperparameter information. Each antibiotic has 10 hyper-para, each for one outer loop.



            ######################################
            ##### Add clinical oriented scores ['clinical_f1_negative','clinical_precision_neg', 'clinical_recall_neg'] to the main tables and PLOT table
            ##### Nov 2022
            clinical_table=extract_info_clinical_SSSA(level,species,cv,learning,epochs,f_fixed_threshold,f_nn_base,f_phylotree,f_kma, temp_path)
            final = pd.concat([final, clinical_table], axis=1, join="inner")
            final_plot = pd.concat([final_plot, clinical_table], axis=1, join="inner")
            #########################################################################################################################################

            final.to_csv(save_name_score_final + '_SummaryBenchmarking.txt', sep="\t")
            final_plot.to_csv(save_name_score_final + '_SummaryBenchmarking_PLOT.txt', sep="\t")
            final_std.to_csv(save_name_score_final + '_SummaryBenchmarking_std.txt', sep="\t")


    elif f_MSMA_discrete:
        out_score='neg' #'f1_macro', 'f1_positive','f1_negative','precision_neg', 'recall_neg', 'accuracy'
        if f_split_species==False and f_match_single==False:
            merge_name = []
            data = pd.read_csv('./data/PATRIC/meta/'+str(level)+'_multi-species_summary.csv', index_col=0,
                           dtype={'genome_id': object}, sep="\t")
            if f_all:
                list_species = data.index.tolist()[:-1]
                data = data.loc[list_species, :]
            else:
                data = data.loc[list_species, :]
                data = data.loc[:, (data.sum() > 1)]


            # --------------------------------------------------------
            # drop columns(antibotics) all zero
            data = data.loc[:, (data != 0).any(axis=0)]
            All_antibiotics = data.columns.tolist()
            for n in list_species:
                merge_name.append(n[0] + n.split(' ')[1][0])
            merge_name = '_'.join(merge_name)  # e.g.Se_Kp_Pa

            save_name_score,save_name_weights, save_name_loss = name_utility.GETname_AAscoreMSMA('AytanAktug',merge_name,learning, epochs,
                         f_fixed_threshold, f_nn_base, f_optimize_score,temp_path,f_kma,f_phylotree,'MSMA_discrete')


            save_name_score_final =  name_utility.GETname_AAresult('AytanAktug',merge_name,learning, epochs,f_fixed_threshold,\
                                         f_nn_base,f_optimize_score,f_kma,f_phylotree,'MSMA_discrete',output_path)
            file_utility.make_dir(os.path.dirname(save_name_score_final))
            with open(save_name_score  + '0_test.json') as f:
                score_temp = json.load(f)
            score=j2l(score_temp)

            final=make_table.multi_make_visualization_normalCV(out_score,All_antibiotics,score)
            final.to_csv(save_name_score_final + '_SummaryBenchmarking.txt', sep="\t")

        if f_MSMA_discrete and f_split_species:# split species-specific scores from discrete model and concatenated mixed species model.
            merge_name = []
            data = pd.read_csv('./data/PATRIC/meta/'+str(level)+'_multi-species_summary.csv', index_col=0,
                           dtype={'genome_id': object}, sep="\t")
            if f_all:
                list_species = data.index.tolist()[:-1]
                data = data.loc[list_species, :]
            else:
                data = data.loc[list_species, :]
                data = data.loc[:, (data.sum() > 1)]


            # --------------------------------------------------------
            # drop columns(antibotics) all zero
            data = data.loc[:, (data != 0).any(axis=0)]
            All_antibiotics = data.columns.tolist()
            df_anti = data.dot(data.columns + ';').str.rstrip(';')#get anti names  marked with 1
            for n in list_species:
                merge_name.append(n[0] + n.split(' ')[1][0])
            merge_name = '_'.join(merge_name)  # e.g.Se_Kp_Pa

            save_name_score,_, _ = name_utility.GETname_AAscoreMSMA('AytanAktug',merge_name,learning, epochs,
                         f_fixed_threshold, f_nn_base, f_optimize_score,temp_path,f_kma,f_phylotree,'MSMA_discrete')

            save_name_score_final =  name_utility.GETname_AAresult('AytanAktug',merge_name,learning, epochs,f_fixed_threshold,\
                                         f_nn_base,f_optimize_score,f_kma,f_phylotree,'MSMA_discrete',output_path)
            # save_name_score_final=os.path.dirname(save_name_score_final)

            with open(save_name_score  + '0_test.json') as f:
                score_temp = json.load(f)
            y_pre_all=score_temp['predictY_test']
            data_y = score_temp['ture_Y']
            data_y= np.array(data_y[0])
            y_pre_all= np.array(y_pre_all[0])
            out_cv=0

            _,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,path_name=\
                    name_utility.GETname_AAfeatureMSMA('AytanAktug',level,'', '',merge_name,temp_path,'MSMA_discrete')
            folds_txt=name_utility.GETname_foldsMSMA(merge_name,level,f_kma,f_phylotree)
            folders_sample_name = json.load(open(folds_txt, "rb"))
            folders_sample=name2index.Get_index(folders_sample_name,path_name)
            test_samples = folders_sample[out_cv] #index
            final_init=pd.DataFrame(index=list_species,columns=data.columns.tolist())
            for species in list_species:

                antibiotics=df_anti[species].split(';')
                for anti in antibiotics:

                    _,_,_,_,_,_,_,_,_,_,_,_,_,_,singleS_id=\
                        name_utility.GETname_AAfeatureSSSA('AytanAktug',level,species, anti,temp_path)
                    All_id=path_name #the same order as data_y
                    #get the index that current combination
                    singleS_list=np.genfromtxt(singleS_id,dtype='str')
                    All_list=np.genfromtxt(All_id,dtype='str')
                    singleS_index=[i for i, e in enumerate(All_list) if e in singleS_list]
                    singleS_test_index=[i for i, e in enumerate(test_samples) if e in singleS_index]
                    y_test=data_y[singleS_test_index]
                    y_pre=y_pre_all[singleS_test_index]
                    #get the index of curent antibiotics in the multi-s model anti list
                    anti_index=All_antibiotics.index(anti)

                    f1 = f1_score(y_test[:, anti_index], y_pre[:, anti_index], average='macro')
                    report = classification_report(y_test[:, anti_index], y_pre[:, anti_index], labels=[0, 1], output_dict=True)

                    report=pd.DataFrame(report).transpose()
                    f1_pos=(report.loc['1', 'f1-score'])
                    f1_neg=(report.loc['0', 'f1-score'])
                    accuracy=(report.iat[2,2])
                    precision_neg=(report.loc['0', 'precision'])
                    recall_neg=(report.loc['0', 'recall'])
                    if fscore=='f1_macro':
                        final_init.loc[species,anti]=f1
                    elif fscore=='f1_positive':
                        final_init.loc[species,anti]=f1_pos
                    elif fscore=='f1_negative':
                        final_init.loc[species,anti]=f1_neg
                    elif fscore=='accuracy':
                        final_init.loc[species,anti]=accuracy
                    elif fscore=='precision_neg':
                        final_init.loc[species,anti]=precision_neg
                    elif fscore=='recall_neg':
                        final_init.loc[species,anti]=recall_neg
                    else:
                        print('only <f1_macro,f1_positive,f1_positive,accuracy,precision_neg,recall_neg> possible so far')
                        exit(1)
            final_init.to_csv(save_name_score_final+'_split_discrete_model_'+str(fscore)+'.txt', sep="\t")
            print(final_init)

        elif f_MSMA_discrete and f_match_single: #match the single-species model results to the multi-s model table for a comparison.
            merge_name = []
            data = pd.read_csv('./data/PATRIC/meta/'+str(level)+'_multi-species_summary.csv', index_col=0,
                           dtype={'genome_id': object}, sep="\t")
            if f_all:
                list_species = data.index.tolist()[:-1]
                data = data.loc[list_species, :]
            else:
                data = data.loc[list_species, :]
                data = data.loc[:, (data.sum() > 1)]


            data = data.loc[:, (data != 0).any(axis=0)]
            df_anti = data.dot(data.columns + ';').str.rstrip(';')#get anti names  marked with 1
            final_init=pd.DataFrame(index=list_species,columns=data.columns.tolist())
            final_init_std=pd.DataFrame(index=list_species,columns=data.columns.tolist())
            for n in list_species:
                merge_name.append(n[0] + n.split(' ')[1][0])
            merge_name = '_'.join(merge_name)  # e.g.Se_Kp_Pa
            # --------------------------------------------------------
            # --------------------------------------------------------
            save_name_score_final =  name_utility.GETname_AAresult('AytanAktug',merge_name,learning, epochs,f_fixed_threshold,\
                                         f_nn_base,f_optimize_score,f_kma,f_phylotree,'MSMA_discrete',output_path)



            for species in list_species:

                anti=df_anti[species].split(';')

                #read in resutls
                single_s_score = name_utility.GETname_AAresult('AytanAktug',species,learning, epochs,f_fixed_threshold,\
                                         f_nn_base,f_optimize_score,f_kma,f_phylotree,'SSSA',output_path)

                data_score=pd.read_csv(single_s_score + '_SummaryBenchmarking_PLOT.txt', sep="\t", header=0, index_col=0)

                ###mean
                for each_anti in anti:
                    final_init.loc[species,each_anti]=data_score.loc[each_anti,'weighted-'+fscore]
                final_init.to_csv(save_name_score_final+'_SSSAmapping_'+fscore+'.txt', sep="\t")
                # print(final_init)

                ####std
                data_score_std=pd.read_csv(single_s_score + '_SummaryBenchmarking_std.txt', sep="\t", header=0, index_col=0)
                for each_anti in anti:
                    final_init_std.loc[species,each_anti]=data_score_std.loc[each_anti,'weighted-'+fscore]
                final_init_std.to_csv(save_name_score_final+'_SSSAmapping_'+fscore+'_std.txt', sep="\t")


    elif f_MSMA_conMix:
        if f_split_species==False:
            merge_name = []
            data = pd.read_csv('./data/PATRIC/meta/'+str(level)+'_multi-species_summary.csv', index_col=0,
                           dtype={'genome_id': object}, sep="\t")

            if f_all:
                list_species = data.index.tolist()[:-1]
                data = data.loc[list_species, :]
            else:
                data = data.loc[list_species, :]
                data = data.loc[:, (data.sum() > 1)]
            # --------------------------------------------------------
            # drop columns(antibotics) all zero
            data = data.loc[:, (data != 0).any(axis=0)]
            All_antibiotics = data.columns.tolist()  # all envolved antibiotics # todo
            for n in list_species:
                merge_name.append(n[0] + n.split(' ')[1][0])
            merge_name = '_'.join(merge_name)  # e.g.Se_Kp_Pa


            save_name_score_concat,_, _ = name_utility.GETname_AAscoreMSMA('AytanAktug',merge_name,learning, epochs,
                         f_fixed_threshold, f_nn_base, f_optimize_score,temp_path,f_kma,f_phylotree,'MSMA_concat_mixedS')

            save_name_score_final = name_utility.GETname_AAresult('AytanAktug',merge_name,learning, epochs,f_fixed_threshold,\
                                         f_nn_base,f_optimize_score,f_kma,f_phylotree,'MSMA_concat_mixedS',output_path)

            file_utility.make_dir(os.path.dirname(save_name_score_final))

            with open(save_name_score_concat  + '0_test.json') as f:
                score_temp = json.load(f)
            score=j2l(score_temp)


            final=make_table.multi_make_visualization_normalCV(out_score, All_antibiotics, score)
            final.to_csv(save_name_score_final + '_SummaryBenchmarking.txt', sep="\t")

        if f_MSMA_conMix and f_split_species:

            merge_name = []
            data = pd.read_csv('./data/PATRIC/meta/'+str(level)+'_multi-species_summary.csv', index_col=0,
                           dtype={'genome_id': object}, sep="\t")
            if f_all:
                list_species = data.index.tolist()[:-1]
                data = data.loc[list_species, :]
            else:
                data = data.loc[list_species, :]
                data = data.loc[:, (data.sum() > 1)]
            # --------------------------------------------------------
            # drop columns(antibotics) all zero
            data = data.loc[:, (data != 0).any(axis=0)]
            All_antibiotics = data.columns.tolist()  # all envolved antibiotics
            df_anti = data.dot(data.columns + ';').str.rstrip(';')#get anti names  marked with 1
            for n in list_species:
                merge_name.append(n[0] + n.split(' ')[1][0])
            merge_name = '_'.join(merge_name)  # e.g.Se_Kp_Pa


            save_name_score_concat,_, _ = name_utility.GETname_AAscoreMSMA('AytanAktug',merge_name,learning, epochs,
                         f_fixed_threshold, f_nn_base, f_optimize_score,temp_path,f_kma,f_phylotree,'MSMA_concat_mixedS')


            save_name_score_final = name_utility.GETname_AAresult('AytanAktug',merge_name,learning, epochs,f_fixed_threshold,\
                                         f_nn_base,f_optimize_score,f_kma,f_phylotree,'MSMA_concat_mixedS',output_path)

            with open(save_name_score_concat  + '0_test.json') as f:
                score_temp = json.load(f)
            y_pre_all=score_temp['predictY_test']
            data_y = score_temp['ture_Y']
            out_cv=0
            data_y= np.array(data_y[0])
            y_pre_all= np.array(y_pre_all[0])
            _,_,_,_,_,_,_,_,_,_,_ ,_,_,_,_, _, path_name_all=\
                            name_utility.GETname_AAfeatureMSMA_concat('AytanAktug',level,merge_name, merge_name,temp_path,'MSMA_concat')
            folds_txt=name_utility.GETname_foldsMSMA(merge_name,level,f_kma,f_phylotree)
            folders_sample_name = json.load(open(folds_txt, "rb"))
            folders_sample=name2index.Get_index(folders_sample_name,path_name_all)

            test_samples_index = folders_sample[out_cv] #index in All
            final_init=pd.DataFrame(index=list_species,columns=data.columns.tolist())

            for species in list_species:

                antibiotics=df_anti[species].split(';')
                for anti in antibiotics:
                    _,_,_,_,_,_,_,_,_,_,_,_,_,_,singleS_id=\
                        name_utility.GETname_AAfeatureSSSA('AytanAktug',level,species, anti,temp_path)
                    All_id=path_name_all #the same order as data_y
                    #get the index that current combination
                    singleS_list=np.genfromtxt(singleS_id,dtype='str') #from single-s model.
                    All_list=np.genfromtxt(All_id,dtype='str')
                    singleSA_index=[i for i, e in enumerate(All_list) if e in singleS_list] #index in All w.r.t. species&anti combination.
                    singleSA_test_index=[i for i, e in enumerate(test_samples_index) if e in singleSA_index]#index in y_pre_all w.r.t. species&anti combination.
                    y_test = data_y[singleSA_test_index]
                    y_pre=y_pre_all[singleSA_test_index]
                    #get the index of curent antibiotics in the multi-s model anti list
                    anti_index=All_antibiotics.index(anti)


                    f1 = f1_score(y_test[:, anti_index], y_pre[:, anti_index], average='macro')
                    report = classification_report(y_test[:, anti_index], y_pre[:, anti_index], labels=[0, 1], output_dict=True)
                    report=pd.DataFrame(report).transpose()
                    f1_pos=(report.loc['1', 'f1-score'])
                    f1_neg=(report.loc['0', 'f1-score'])
                    accuracy=(report.iat[2,2])
                    precision_neg=(report.loc['0', 'precision'])
                    recall_neg=(report.loc['0', 'recall'])

                    if fscore=='f1_macro':
                        final_init.loc[species,anti]=f1
                    elif fscore=='f1_positive':
                        final_init.loc[species,anti]=f1_pos
                    elif fscore=='f1_negative':
                        final_init.loc[species,anti]=f1_neg
                    elif fscore=='accuracy':
                        final_init.loc[species,anti]=accuracy
                    elif fscore=='precision_neg':
                        final_init.loc[species,anti]=precision_neg
                    elif fscore=='recall_neg':
                        final_init.loc[species,anti]=recall_neg
                    else:
                        print('only <f1_macro,f1_positive,f1_positive,accuracy,precision_neg,recall_neg> possible so far')
                        exit(1)

            final_init.to_csv(save_name_score_final+'_split_discrete_model_'+str(fscore)+'.txt', sep="\t")
            print(final_init)

    elif f_MSMA_conLOO:
        out_score='neg' #f1_macro,f1_positive,f1_positive,accuracy,precision_neg,recall_neg
        merge_name = []
        data = pd.read_csv('./data/PATRIC/meta/'+str(level)+'_multi-species_summary.csv', index_col=0,
                       dtype={'genome_id': object}, sep="\t")
        if f_all:
            list_species = data.index.tolist()[:-1]
            data = data.loc[list_species, :]
        else:
            data = data.loc[list_species, :]
            data = data.loc[:, (data.sum() > 1)]

        # --------------------------------------------------------
        # drop columns(antibotics) all zero
        data = data.loc[:, (data != 0).any(axis=0)]
        All_antibiotics = data.columns.tolist()  # all envolved antibiotics
        for n in list_species:
            merge_name.append(n[0] + n.split(' ')[1][0])
        merge_name = '_'.join(merge_name)  # e.g.Se_Kp_Pa


        count = 0
        for species_testing in list_species:
            print('species_testing',species_testing)
            list_species_training = list_species[:count] + list_species[count + 1:]
            count += 1
            # do a nested CV on list_species, select the best estimator for testing on the standing out species
            merge_name_train = []
            for n in list_species_training:
                merge_name_train.append(n[0] + n.split(' ')[1][0])
            merge_name_train = '_'.join(merge_name_train)  # e.g.Se_Kp_Pa
            merge_name_test = species_testing.replace(" ", "_")
            # 1. testing on the left-out species scores
            save_name_score ,_, _ = name_utility.GETname_AAscoreMSMA_concat('AytanAktug',merge_name,merge_name_test,learning, epochs,
                         f_fixed_threshold, f_nn_base, f_optimize_score,temp_path,f_kma,f_phylotree,'MSMA_concatLOO')

            save_name_score_final = name_utility.GETname_AAresult('AytanAktug',merge_name,learning, epochs,f_fixed_threshold,\
                                     f_nn_base,f_optimize_score,f_kma,f_phylotree,'MSMA_concatLOO',output_path)
            file_utility.make_dir(os.path.dirname(save_name_score_final))

            with open(save_name_score  + '_TEST.json') as f:
                score_temp = json.load(f)
            score=j2l(score_temp)

            final=make_table.concat_multi_make_visualization(out_score, All_antibiotics, score)
            final.to_csv(save_name_score_final + '_'+merge_name_test+'_SummaryBenchmarking.txt', sep="\t")


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                        help='Directory to store temporary files.')
    parser.add_argument('-o', '--output_path', default='./', type=str, required=False,
                        help='Directory to store CV scores.')
    # parser.add_argument('-t_p', '--threshold_point', default=0.8, type=float,
    #                     help='Threshold for identity of Pointfinder. ')
    # parser.add_argument('-l_p', '--min_cov_point', default=0.6, type=float,
    #                     help=' Minimum (breadth-of) coverage of Pointfinder. ')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument("-f_SSMA", "--f_SSMA",  dest='f_SSMA', action='store_true',
                            help='flag for single-species multi-anti model')
    parser.add_argument("-f_SSSA", "--f_SSSA",  dest='f_SSSA', action='store_true',
                            help='flag for single-species-anti model')
    parser.add_argument("-f_MSMA_discrete", "--f_MSMA_discrete",  dest='f_MSMA_discrete', action='store_true',
                            help='flag for MSMA_discrete')
    parser.add_argument("-f_MSMA_conMix", "--f_MSMA_conMix",  dest='f_MSMA_conMix', action='store_true',
                            help='flag for MSMA_conMix')
    parser.add_argument("-f_MSMA_conLOO", "--f_MSMA_conLOO",  dest='f_MSMA_conLOO', action='store_true',
                            help='flag for MSMA_conMix')
    parser.add_argument("-f_split_species", "--f_split_species",  dest='f_split_species', action='store_true',
                            help='flag for splitting score as a whole into species-specific scores from discrete \
                                 model and concatenated mixed species model. so far only f1_macro')

    parser.add_argument("-f_match_single", "--f_match_single", dest='f_match_single', action='store_true',
                        help='flag for match single-species model results to multi-species model results for a comparison.')
    parser.add_argument('-out_score', '--out_score', default='f', type=str,
                        help='Scores of the final output table. f:f_macro,f_pos,f_neg,f_micro. all:all scores. neg:f1_macro,precision,recall,accuracy')
    parser.add_argument('-fscore', '--fscore', default='f1_macro', type=str, required=False,
                        help='The score used for final comparison. Can be one of: \'f1_macro\',\'f1_positive\',\'f1_negative\',\'accuracy\'.')
    parser.add_argument("-cv", "--cv_number", default=10, type=int,
                        help='CV splits number')
    parser.add_argument("-e", "--epochs", default=1000, type=int,
                        help='epochs')
    parser.add_argument("-learning", "--learning", default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('-l', '--level', default='loose', type=str,
                        help='Quality control: strict or loose')
    parser.add_argument('-f_fixed_threshold', '--f_fixed_threshold', dest='f_fixed_threshold', action='store_true',
                        help='set a fixed threshod:0.5.')
    parser.add_argument('-f_optimize_score', '--f_optimize_score', default='f1_macro', type=str,
                        help='the optimizing-score for choosing the best estimator in inner loop. Choose: auc or f1_macro.')
    parser.add_argument('-f_nn_base', '--f_nn_base', dest='f_nn_base', action='store_true',
                        help='benchmarking baseline.')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.') #only single-species model
    parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                        help='kma based cv folders.')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
            \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
            \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parsedArgs = parser.parse_args()
    # parser.print_help()

    extract_info(parsedArgs.out_score,parsedArgs.fscore,parsedArgs.f_SSMA,parsedArgs.f_SSSA,parsedArgs.f_MSMA_discrete,
                 parsedArgs.f_MSMA_conMix,parsedArgs.f_MSMA_conLOO,parsedArgs.f_split_species,
                 parsedArgs.f_all,parsedArgs.f_match_single,parsedArgs.species,parsedArgs.level,
                 parsedArgs.cv_number,parsedArgs.epochs,parsedArgs.learning,parsedArgs.f_fixed_threshold,parsedArgs.f_nn_base,
                 parsedArgs.f_phylotree,parsedArgs.f_kma,parsedArgs.f_optimize_score,parsedArgs.temp_path,parsedArgs.output_path)



