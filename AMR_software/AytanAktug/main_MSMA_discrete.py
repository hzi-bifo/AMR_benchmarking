import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
# sys.path.append('../../')
sys.path.insert(0, os.getcwd())
from src.amr_utility import name_utility, file_utility,load_data
from AMR_software.AytanAktug.data_preparation import merge_scaffolds,scored_representation_blast,ResFinder_analyser_blast,merge_resfinder_pointfinder,merge_input_output_files,merge_resfinder
import argparse,json
import pandas as pd
import numpy as np
from AMR_software.AytanAktug.nn import nn_MSMA_discrete
from src.cv_folds import cluster2folds



def merge_feature(merge_name,temp_path,list_species,All_antibiotics,level):
    '''
    :return: merged feature matrix , data_x, data_y, data_name
    '''
    count=0
    id_feature_all = []  # feature dataframe of each species
    id_pheno_all = []
    feature_dimension_all=pd.DataFrame( index=list_species,columns=['feature dimension'])

    if len(list_species)<2:
        print('pleas feed in at lest 2 species.')
        exit(1)
    _,_,_,_,path_ID_multi,_,_,_,_,_,_,_,_,path_feature,path_x,path_y,path_name=\
                name_utility.GETname_AAfeatureMSMA('AytanAktug',level,'', '',merge_name,temp_path,'MSMA_discrete')

    for species in list_species:
        _,path_ID_multi_each,path_metadata_multi_each,path_metadata_multi_each_main,_,\
           _,path_cluster,path_cluster_results,path_cluster_temp,path_point_repre_results_each,\
           path_res_repre_results_each,path_mutation_gene_results_each,path_res_result,_,_,_,_=\
                name_utility.GETname_AAfeatureMSMA('AytanAktug',level,species, '',merge_name,temp_path,'MSMA_discrete')

        meta_s = pd.read_csv(path_metadata_multi_each_main, sep="\t", header=0, index_col=0, dtype={'id': object, 'pheno': int})

        #feature matrix:
        feature_s=np.genfromtxt(path_mutation_gene_results_each, dtype="str")
        n_feature_s=feature_s.shape[1]-1#number of features for this species
        feature_dimension_all.loc[species,'feature dimension']=n_feature_s
        df_feature_s=pd.DataFrame(feature_s, index=None, columns=np.insert(np.array(np.arange(n_feature_s)+count,dtype='object'), 0, 'id'))#,dtype={'id': object}
        id_pheno_all.append(meta_s)
        id_feature_all.append(df_feature_s)
        # print(df_feature_s)
        count += n_feature_s

    feature_dimension_all.to_csv(path_feature+'/feature_Dimension.txt', sep="\t")
    df_feature_s_f=id_feature_all[0]
    for i in id_feature_all[1:]:
        df_feature_s_f= pd.concat([df_feature_s_f, i], ignore_index=True, sort=False)

    df_pheno_s_f = id_pheno_all[0]
    for i in id_pheno_all[1:]:
        df_pheno_s_f = pd.concat([df_pheno_s_f, i], ignore_index=True, sort=False)

    #Merge meta and pheno to make sure the use the same id list(order).
    df_feature_s_f = pd.merge(df_feature_s_f, df_pheno_s_f, how="outer", on=["id"])

    df_feature_s_f = df_feature_s_f.set_index('id')
    id_list = np.genfromtxt(path_ID_multi, dtype="str")
    df_feature_s_f = df_feature_s_f.reindex(id_list)

    # Pad nan with 0 in feature matrix, with -1 in phen matrix
    df_feature_final=df_feature_s_f.loc[:,np.array(np.arange(sum(feature_dimension_all['feature dimension'].to_list())),dtype='object')]#exclude pheno part
    df_feature_final=df_feature_final.fillna(0)
    df_feature_final.to_csv(path_x,index=False,header=False, sep="\t")

    df_phenotype_final=df_feature_s_f.loc[:, All_antibiotics]
    df_phenotype_final=df_phenotype_final.fillna(-1)
    print(df_phenotype_final) #[20989 rows x 20 columns]
    df_phenotype_final.to_csv(path_y,index=False,header=False, sep="\t")
    df_feature_s_f.index.to_series().to_csv(path_name,header=False, index=False,sep="\t")

    print('Feature part of discerte multi-s model finished. Can procede to NN model now.')




def prepare_meta(list_species,data,level,selected_anti,merge_name,temp_path):
    '''
    return: each species' metadata of selected antibitocs. combined metadata of all selected speceis(all antibiotics).
    '''

    if selected_anti==[]:
        cols = data.columns
        bt = data.apply(lambda x: x > 0)#all possible antibiotics
        data_species_anti = bt.apply(lambda x: list(cols[x.values]), axis=1)
        print(data_species_anti)# dataframe of each species and coresponding selected antibiotics.
    else:
        print('Not possible to choose anti by user yet. May come in the future...')
        exit(1)

    # ---------------------------------------------------------------
    _,_,_,_,path_ID_multi,path_metadata_multi,_,_,_,_,_,_,_,_,_,_,_=\
                    name_utility.GETname_AAfeatureMSMA('AytanAktug',level,'', '',merge_name,temp_path,'MSMA_discrete')
    metadata_pheno_all=[]
    for species in list_species:
        print(species)
        metadata_pheno_all_sub=[]
        _,path_ID_multi_each_main,_,path_metadata_multi_each_main,_,_,_,_,_,_,_,_,_,_,_,_,_=\
                    name_utility.GETname_AAfeatureMSMA('AytanAktug',level,species, '',merge_name,temp_path,'MSMA_discrete')
        for anti in data_species_anti[species]:
            path_metadata_pheno,_,path_metadata_multi_each,_,_,_,_,_,_,_,_,_,_,_,_,_,_=\
                    name_utility.GETname_AAfeatureMSMA('AytanAktug',level,species, anti,merge_name,temp_path,'MSMA_discrete')
            file_utility.make_dir(os.path.dirname(path_metadata_multi_each))
            meta_pheno_temp = pd.read_csv(path_metadata_pheno, index_col=0, dtype={'genome_id': object}, sep="\t")
            meta_pheno_temp.to_csv(path_metadata_multi_each, sep="\t", index=False,header=False)

            metadata_pheno = pd.read_csv(path_metadata_multi_each,  sep="\t",header=None,names=['id',anti],dtype={'id': object,'pheno':int})
            metadata_pheno_all_sub.append(metadata_pheno)
        if len(metadata_pheno_all_sub)>1:
            metadata_pheno=metadata_pheno_all_sub[0]
            for i in metadata_pheno_all_sub[1:]:
                metadata_pheno = pd.merge(metadata_pheno, i, how="outer", on=["id"])# merge antibiotics within one species
        else:
            pass#no need for merge


        metadata_pheno.to_csv(path_metadata_multi_each_main, sep="\t", index=True, header=True)
        metadata_pheno['id'].to_csv(path_ID_multi_each_main, sep="\t", index=False, header=False)
        metadata_pheno_all.append(metadata_pheno)
    metadata_pheno_f=metadata_pheno_all[0]
    for i in metadata_pheno_all[1:]:
        metadata_pheno_f =  metadata_pheno_f.append(i) # append all the species

    metadata_pheno_f.to_csv(path_metadata_multi, sep="\t",index=True, header=True)
    metadata_pheno_f['id'].to_csv(path_ID_multi, sep="\t", index=False,header=False)






def extract_info(path_sequence,list_species,selected_anti,level,f_all,f_pre_meta,
                 f_pre_cluster,f_cluster_folds,f_res,f_merge_mution_gene,f_matching_io,f_nn,cv,i_CV,
                 epochs, learning,f_scaler,f_fixed_threshold,f_nn_base,f_nn_score,f_phylotree,f_kma,f_optimize_score,temp_path):
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
    data = data.loc[:, (data != 0).any(axis=0)]    # drop columns(antibotics) all zero
    All_antibiotics = data.columns.tolist()  # all envolved antibiotics
    for n in list_species:
        merge_name.append(n[0] + n.split(' ')[1][0])
    merge_name = '_'.join(merge_name)  # e.g.Se_Kp_Pa# data storage: one combination one file!


    if f_pre_meta==True:
        print('Involved antibiotics: ',All_antibiotics)
        prepare_meta(list_species,data,level,selected_anti,merge_name,temp_path)

    if f_pre_cluster:

        for species in list_species:
            print(species)
            _,path_ID_multi_each_main,_,path_metadata_multi_each_main,_,_,path_cluster,_,_,_,_,_,_,_,_,_,_=\
                    name_utility.GETname_AAfeatureMSMA('AytanAktug',level,species, '',merge_name,temp_path,'MSMA_discrete')
            file_utility.make_dir(os.path.dirname(path_cluster))
            merge_scaffolds.extract_info(path_sequence,path_ID_multi_each_main, path_cluster, 16)



    if f_cluster_folds:
        _,_,_,_,path_ID_multi,_,_,_,_,_,_,_,_,_,_,_,_=\
                    name_utility.GETname_AAfeatureMSMA('AytanAktug',level,'', '',merge_name,temp_path,'MSMA_discrete')
        path_cluster_results=[]


        for species in list_species:
            _,_,_,_,_,_,path_cluster,path_cluster_results_eachS,_,_,_,_,_,_,_,_,_=\
                    name_utility.GETname_AAfeatureMSMA('AytanAktug',level,species, '',merge_name,temp_path,'MSMA_discrete')
            path_cluster_results.append(path_cluster_results_eachS)


        _, _, folders_sampleName = cluster2folds.prepare_folders(cv, 42, path_ID_multi,path_cluster_results, 'new')

        folds_txt=name_utility.GETname_foldsMSMA(merge_name,level, True, False)

        file_utility.make_dir(os.path.dirname(folds_txt))
        with open(folds_txt, 'w') as f:
                json.dump(folders_sampleName, f)

    # =================================
    # 2. Analysing PointFinder results
    # Analysing ResFinder results
    # =================================
    if f_res == True:
        for species in list_species:
            _,path_ID_multi_each,path_metadata_multi_each,path_metadata_multi_each_main,path_ID_multi,\
               path_metadata_multi,path_cluster,path_cluster_results,path_cluster_temp,path_point_repre_results_each,\
               path_res_repre_results_each,path_mutation_gene_results_each,path_res_result,_,_,_,_=\
                    name_utility.GETname_AAfeatureMSMA('AytanAktug',level,species, '',merge_name,temp_path,'MSMA_discrete')
            if species in ['Klebsiella pneumoniae','Escherichia coli','Staphylococcus aureus','Mycobacterium tuberculosis','Salmonella enterica',
                       'Neisseria gonorrhoeae','Enterococcus faecium','Campylobacter jejuni']:

                scored_representation_blast.extract_info(path_res_result, path_ID_multi_each, path_point_repre_results_each, True,True)#SNP,no zip

            ResFinder_analyser_blast.extract_info(path_res_result, path_ID_multi_each, path_res_repre_results_each,True)  # GPA,no zip


    if f_merge_mution_gene == True:
        for species in list_species:
            _,path_ID_multi_each,path_metadata_multi_each,path_metadata_multi_each_main,path_ID_multi,\
               path_metadata_multi,path_cluster,path_cluster_results,path_cluster_temp,path_point_repre_results_each,\
               path_res_repre_results_each,path_mutation_gene_results_each,path_res_result,_,_,_,_=\
                    name_utility.GETname_AAfeatureMSMA('AytanAktug',level,species, '',merge_name,temp_path,'MSMA_discrete')
            if species in ['Klebsiella pneumoniae', 'Escherichia coli', 'Staphylococcus aureus',
                           'Mycobacterium tuberculosis', 'Salmonella enterica',
                           'Neisseria gonorrhoeae', 'Enterococcus faecium', 'Campylobacter jejuni']:
                merge_resfinder_pointfinder.extract_info(path_point_repre_results_each,
                                                                                      path_res_repre_results_each,
                                                                                      path_mutation_gene_results_each)
            else:  # only AMR gene feature
                merge_resfinder.extract_info(path_ID_multi_each, path_res_repre_results_each,path_mutation_gene_results_each)

    if f_matching_io == True:
        #Different from single-s model.
        merge_feature(merge_name, temp_path, list_species,All_antibiotics, level)


    # =================================
    #3.  run model
    # =================================
    if f_nn == True:
        _,_,_,_,path_ID_multi,_,_,_,_,_,_,_,_,_,path_x,path_y,path_name=\
                name_utility.GETname_AAfeatureMSMA('AytanAktug',level,'', '',merge_name,temp_path,'MSMA_discrete')

        save_name_score,save_name_weights, save_name_loss = name_utility.GETname_AAscoreMSMA('AytanAktug',merge_name,learning, epochs,
                     f_fixed_threshold, f_nn_base, f_optimize_score,temp_path,f_kma,f_phylotree,'MSMA_discrete')#if learning=0.0, and epoch = 0, it means hyper parameter selection mode.

        file_utility.make_dir(os.path.dirname(save_name_score))
        file_utility.make_dir(os.path.dirname(save_name_weights))
        file_utility.make_dir(os.path.dirname(save_name_loss))

        if f_nn_score: #test on the hold-out test set
            # after above finished for each of 5 folds CV
            sys.stdout = open(save_name_loss+"_test"+'.txt', 'w')
            nn_MSMA_discrete.multi_test(merge_name,'MSMA', level, path_x, path_y, path_name,cv, f_scaler, f_fixed_threshold,
                               f_nn_base, f_phylotree,f_kma, f_optimize_score,save_name_weights, save_name_score)

        else: #run  5-fold CV. # hyperparmeter selection
            print('check')
            sys.stdout = open(save_name_loss+str(i_CV)+'.txt', 'w')
            nn_MSMA_discrete.multi_eval(merge_name,'MSMA', level, path_x, path_y, path_name,[i_CV], f_scaler,
                            f_nn_base, f_phylotree,f_kma, f_optimize_score, save_name_weights,save_name_score)

if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_sequence', '--path_sequence', type=str,required=False,
                        help='Path of the directory with PATRIC sequences.')
    parser.add_argument('-l', '--level', default='loose', type=str,
                        help='Quality control: strict or loose')
    parser.add_argument('-f_pre_meta', '--f_pre_meta', dest='f_pre_meta', action='store_true',
                        help=' Prepare metadata for multi-species model.')
    parser.add_argument('-f_pre_cluster', '--f_pre_cluster', dest='f_pre_cluster', action='store_true',
                        help='Prepare files for Kma clustering')
    parser.add_argument('-f_cluster_folds', '--f_cluster_folds', dest='f_cluster_folds', action='store_true',
                        help=' Generate KMA folds.')
    parser.add_argument('-f_cluster_folders', '--f_cluster_folders', dest='f_cluster_folders', action='store_true',
                        help='Compare new split method with old(original) method.')  # c program
    parser.add_argument('-f_res', '--f_res', dest='f_res', action='store_true',
                        help='Analyse Point/ResFinder results')
    parser.add_argument('-f_merge_mution_gene', '--f_merge_mution_gene', dest='f_merge_mution_gene',
                        action='store_true',
                        help=' Merging ResFinder and PointFinder results')
    parser.add_argument('-f_matching_io', '--f_matching_io', dest='f_matching_io', action='store_true',
                        help='Matching input and output results')
    # para for CV
    parser.add_argument('-f_nn', '--f_nn', dest='f_nn', action='store_true',
                        help='Run the NN model')
    parser.add_argument('-f_nn_score', '--f_nn_scorer', dest='f_nn_score', action='store_true',
                        help='Wrap NN model scores from outer loops.')
    parser.add_argument("-cv", "--cv_number", default=6, type=int,
                        help='k+1, where k is CV splits number, and 1 is the hold out test set.')
    parser.add_argument("-i_CV", "--i_CV", type=int,
                        help=' the number of CV iteration to run.')
    parser.add_argument("-e", "--epochs", default=0, type=int,
                        help='epochs (only for output names purpose).  0 indicate using the hyperparameter optimization.')
    parser.add_argument("-learning", "--learning", default=0.0, type=float,
                        help='learning rate (only for output names purpose).  0.0 indicate using the hyperparameter optimization.')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                        help='kma based cv folders.')
    parser.add_argument('-f_scaler', '--f_scaler', dest='f_scaler', action='store_true',
                        help='normalize the data')
    parser.add_argument('-f_fixed_threshold', '--f_fixed_threshold', dest='f_fixed_threshold', action='store_true',
                        help='set a fixed threshod:0.5.')
    parser.add_argument('-f_nn_base', '--f_nn_base', dest='f_nn_base', action='store_true',
                        help='benchmarking baseline.')

    parser.add_argument('-f_optimize_score', '--f_optimize_score', default='f1_macro', type=str,
                        help='the optimizing-score for choosing the best estimator in inner loop. Choose: auc or f1_macro.')
    parser.add_argument('-anti', '--anti', default=[], type=str, nargs='+', help='one antibioticseach time to run: e.g.\'ciprofloxacin\' \
	            \'gentamicin\' \'ofloxacin\' \'tetracycline\' \'trimethoprim\' \'imipenem\' \
	            \'meropenem\' \'amikacin\'...')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
            \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
            \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                help='Directory to store temporary files.')

    parsedArgs = parser.parse_args()
    # parser.print_help()
    extract_info(parsedArgs.path_sequence, parsedArgs.species,parsedArgs.anti, parsedArgs.level, parsedArgs.f_all,
                 parsedArgs.f_pre_meta, parsedArgs.f_pre_cluster, parsedArgs.f_cluster_folds,
                 parsedArgs.f_res, parsedArgs.f_merge_mution_gene, parsedArgs.f_matching_io,
                 parsedArgs.f_nn, parsedArgs.cv_number,parsedArgs.i_CV, parsedArgs.epochs,
                 parsedArgs.learning, parsedArgs.f_scaler, parsedArgs.f_fixed_threshold,
                 parsedArgs.f_nn_base,parsedArgs.f_nn_score,parsedArgs.f_phylotree,parsedArgs.f_kma,parsedArgs.f_optimize_score,parsedArgs.temp_path)


# future work todo: if anti can be selected, then some saving names should be reconsidered.
