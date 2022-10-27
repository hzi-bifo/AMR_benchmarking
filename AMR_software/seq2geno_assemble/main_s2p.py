#!/usr/bin/python
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import numpy as np
from src.amr_utility import name_utility, file_utility, load_data
import argparse
import itertools
import pandas as pd
import hyper_range
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import auc,roc_curve,matthews_corrcoef,confusion_matrix,f1_score,precision_recall_fscore_support,classification_report
import pickle,json
from sklearn import preprocessing
from src.cv_folds import name2index


def extract_info(path_sequence,temp_path,s,f_all,f_prepare_meta, f_phylotree, f_kma,level,f_ml,cv,n_jobs):


    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]
    if f_all == False:
        data = data.loc[s, :]
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    file_utility.make_dir(temp_path+'log/software/seq2geno/software_output/cano6mer/temp') #for kmers

    if f_prepare_meta:
        #prepare the dna list accroding to S2G format.
        for species, antibiotics in zip(df_species, antibiotics):
            antibiotics, _, _ =  load_data.extract_info(species, False, level)
            ALL=[]
            for anti in antibiotics:
                name,_,_ = name_utility.GETname_model('seq2geno',level, species, anti,'',temp_path)
                name_list = pd.read_csv(name, index_col=0, dtype={'genome_id': object}, sep="\t")
                name_list['path']=str(path_sequence) +'/'+ name_list['genome_id'].astype(str)+'.fna'
                name_list['genome_id'] = 'iso_' + name_list['genome_id'].astype(str)
                pseudo=np.empty(len(name_list.index.to_list()),dtype='object')
                pseudo.fill ('./AMR_software/seq2geno/example/CH2500.1.fastq.gz, /AMR_software/seq2geno/example/CH2500.2.fastq.gz')
                name_list['path_pseudo'] = pseudo
                ALL.append(name_list)

            _,dna_list, assemble_list, yml_file, run_file_name,wd =  name_utility.GETname_S2Gfeature( species, temp_path,6)
            file_utility.make_dir(os.path.dirname(dna_list))

            #combine the list for all antis
            species_dna=ALL[0]
            for i in ALL[1:]:
                species_dna = pd.merge(species_dna, i, how="outer", on=["genome_id",'path','path_pseudo'])# merge antibiotics within one species
            # print(species_dna)
            species_dna_final=species_dna.loc[:,['genome_id','path']]
            species_dna_final.to_csv(assemble_list, sep="\t", index=False,header=False)
            species_pseudo = species_dna.loc[:, ['genome_id', 'path_pseudo']]
            species_pseudo.to_csv(dna_list, sep="\t", index=False, header=False)


            #prepare yml files based on a basic version at working directory

            # cmd_cp='cp seq2geno_inputs.yml %s' %wd
            # subprocess.run(cmd_cp, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            wd_results = wd + 'results'
            file_utility.make_dir(wd_results)

            fileDir = os.path.dirname(os.path.realpath('__file__'))
            wd_results = os.path.join(fileDir, wd_results)
            assemble_list=os.path.join(fileDir, assemble_list)
            dna_list=os.path.join(fileDir, dna_list)
            # #modify the yml file
            a_file = open("./AMR_software/seq2geno/seq2geno_inputs.yml", "r")
            list_of_lines = a_file.readlines()
            list_of_lines[12] = "    100000\n"
            list_of_lines[14] = "    %s\n" % dna_list
            list_of_lines[26] = "    %s\n" % wd_results
            list_of_lines[28] = "    %s\n" % assemble_list


            a_file = open(yml_file, "w")
            a_file.writelines(list_of_lines)
            a_file.close()



    if f_ml:

        for species, antibiotics in zip(df_species, antibiotics):
            antibiotics, ID, Y = load_data.extract_info(species, False, level)
            i_anti = 0
            # antibiotics, ID, Y = antibiotics[9:11], ID[9:11], Y[9:11]
            # antibiotics, ID, Y = antibiotics[11:], ID[11:], Y[11:]
            for anti in antibiotics:

                id_all = ID[i_anti]  # sample name list, e.g. [1352.10013,1352.10014,1354.10,1366.10]
                y_all = Y[i_anti]
                i_anti+=1
                mer6_file ,_, _, _, _,log_feature =  name_utility.GETname_S2Gfeature(species, temp_path,6)

                s2g_file=log_feature+'results/RESULTS/bin_tables'
                data_feature1 = pd.read_csv(s2g_file+'/gpa.mat_NONRDNT', index_col=0,sep="\t")
                data_feature2 = pd.read_csv(s2g_file + '/indel.mat_NONRDNT',index_col=0,sep="\t")
                data_feature3 = pd.read_hdf(mer6_file)
                data_feature3 = data_feature3.T

                init_feature = np.zeros((len(id_all), 1), dtype='uint16')


                data_model_init = pd.DataFrame(init_feature, index=id_all, columns=['initializer'])
                X_all = pd.concat([data_model_init, data_feature3.reindex(data_model_init.index)], axis=1)
                #only scale the 6mer data
                scaler = preprocessing.StandardScaler(with_mean=True, with_std=True).fit(X_all)
                X_all = pd.DataFrame(data=scaler.transform(X_all),
                               index=X_all.index,
                               columns=X_all.columns)
                X_all.index= 'iso_'+X_all.index
                id_all=['iso_'+ s  for s in id_all]
                X_all = pd.concat([X_all, data_feature1.reindex(X_all.index)], axis=1)
                X_all = pd.concat([X_all, data_feature2.reindex(X_all.index)], axis=1)
                X_all = X_all.drop(['initializer'], axis=1)

                id_all = np.array(id_all)
                y_all = np.array(y_all)
                for chosen_cl in ['svm','lr', 'rf','lsvm']:

                    hyper_space,cl=hyper_range.hyper_range(chosen_cl)
                    # 1. by each classifier.2. by outer loop. 3. by inner loop. 4. by each hyper-para

                    mcc_test = []  # MCC results for the test data
                    f1_test = []
                    score_report_test = []
                    aucs_test = []
                    hyperparameters_test = []
                    predictY_test=[]
                    true_Y=[]
                    sampleNames_test=[]
                    estimator_test=[]

                    _, _,save_name_score=name_utility.GETname_model('seq2geno', level,species, anti,chosen_cl,temp_path)
                    file_utility.make_dir(os.path.dirname(save_name_score))
                    for out_cv in range(cv):
                        print(anti,'. Starting outer: ', str(out_cv), '; chosen_cl: ', chosen_cl)

                        p_names = name_utility.GETname_meta(species,anti,level)
                        folds_txt=name_utility.GETname_folds(species,anti,level,f_kma,f_phylotree)
                        folders_sample = json.load(open(folds_txt, "rb"))
                        folders_index=name2index.Get_index(folders_sample,p_names) # CV folds


                        test_samples_index = folders_index[out_cv]# a list of index
                        id_test = id_all[test_samples_index]#sample name list
                        y_test = y_all[test_samples_index]
                        train_val_train_index =folders_index[:out_cv] +folders_index[out_cv + 1:]
                        id_val_train = id_all[list(itertools.chain.from_iterable(train_val_train_index))]  # sample name list
                        y_val_train = y_all[list(itertools.chain.from_iterable(train_val_train_index))]
                        X_val_train=X_all.loc[id_val_train,:]
                        X_test=X_all.loc[id_test,:]



                        pipe = Pipeline(steps=[('cl', cl)])

                        search = GridSearchCV(estimator=pipe, param_grid=hyper_space, n_jobs=n_jobs,
                                                  scoring='f1_macro',
                                                  cv=create_generator(train_val_train_index), refit=True)

                        search.fit(X_all, y_all)
                        hyperparameters_test_sub=search.best_estimator_
                        current_pipe=hyperparameters_test_sub
                        # -------------------------------------------------------
                        # retrain on train and val
                        current_pipe.fit(X_val_train, y_val_train)
                        y_test_pred = current_pipe.predict(X_test)
                        # scores
                        f1 = f1_score(y_test, y_test_pred, average='macro')
                        report = classification_report(y_test, y_test_pred, labels=[0, 1], output_dict=True)
                        mcc = matthews_corrcoef(y_test, y_test_pred)
                        fpr, tpr, _ = roc_curve(y_test, y_test_pred, pos_label=1)
                        roc_auc = auc(fpr, tpr)
                        f1_test.append(f1)
                        score_report_test.append(report)
                        aucs_test.append(roc_auc)
                        mcc_test.append(mcc)
                        hyperparameters_test.append(hyperparameters_test_sub)
                        predictY_test.append( y_test_pred.tolist())
                        true_Y.append(y_test.tolist())
                        sampleNames_test.append(folders_sample[out_cv])
                        estimator_test.append(current_pipe)


                    score ={'f1_test':f1_test,'score_report_test':score_report_test,'aucs_test':aucs_test,'mcc_test':mcc_test,
                         'predictY_test':predictY_test,'ture_Y':true_Y,'samples':sampleNames_test}
                    score2= {'hyperparameters_test':hyperparameters_test,'estimator_test':estimator_test}
                    with open(save_name_score + '_KMA_' + str(f_kma) + '_Tree_' + str(f_phylotree) + '.json',
                              'w') as f:  # overwrite mode
                        json.dump(score, f)
                    with open(save_name_score + '_kma_' + str(f_kma) + '_tree_' + str(f_phylotree) + '_model.pickle',
                              'wb') as f:  # overwrite mode
                        pickle.dump(score2, f)


def create_generator(nFolds):
    for idx in range(len(nFolds)):
        test =nFolds[idx]
        train = list(itertools.chain(*[fold for idy, fold in enumerate(nFolds) if idy != idx]))
        yield train, test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-path_sequence', '--path_sequence', type=str, required=False,
                        help='Path of the directory with PATRIC sequences')
    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                        help='Directory to store temporary files.')
    parser.add_argument('-l', '--level', default='loose', type=str,
                        help='Quality control: strict or loose')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species.')
    parser.add_argument('-f_prepare_meta', '--f_prepare_meta', dest='f_prepare_meta', action='store_true',
                        help='Prepare the list files for S2G.')
    parser.add_argument('-f_finished', '--f_finished', dest='f_finished', action='store_true',
                        help='delete large unnecessary tempt files')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folds.')
    parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                        help='kma based cv folds.')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'Pseudomonas aeruginosa\' \
                \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
                \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('-f_ml', '--f_ml', dest='f_ml', action='store_true',
                        help='ML')  #todo delete this
    parser.add_argument("-cv", "--cv_number", default=10, type=int,
                        help='CV splits number. Default=10 ')
    parser.add_argument('-n_jobs','--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parsedArgs = parser.parse_args()
    extract_info(parsedArgs.path_sequence,parsedArgs.temp_path,parsedArgs.species,parsedArgs.f_all,parsedArgs.f_prepare_meta,
                parsedArgs.f_phylotree,parsedArgs.f_kma,parsedArgs.level,parsedArgs.f_ml,parsedArgs.cv_number,parsedArgs.n_jobs)
