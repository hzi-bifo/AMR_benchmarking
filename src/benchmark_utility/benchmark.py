#!/usr/bin/python

import sys,os
# sys.path.append('../')
sys.path.insert(0, os.getcwd())
import src.benchmark_utility.lib.MAINtable,src.benchmark_utility.lib.table_analysis, src.benchmark_utility.lib.SampleSize,\
    src.benchmark_utility.lib.Com_BySpecies,src.benchmark_utility.lib.Com_BySpecies_each, src.benchmark_utility.lib.pairbox,\
    src.benchmark_utility.lib.ByAnti_errorbar,src.benchmark_utility.lib.pairbox_majority,src.benchmark_utility.lib.pairbox_majority,\
    src.benchmark_utility.lib.ByAnti_errorbar_each,src.benchmark_utility.lib.heatmap, src.benchmark_utility.lib.pairbox_antibiotic, \
    src.benchmark_utility.lib.pairbox_separateFig
import argparse




'''
This script summarizes benchmarking results as graphs and tables.
'''




def extract_info(level,species, fscore,  f_all,f_species, f_anti,f_robust,f_sample,
                 f_table,f_table_analysis,f_clinical_analysis,f_hmap,output_path):

    ####################################################################################################################
    ### 1.Supplemental File 1 Performance(F1-macro, negative F1-score, positive F1-score, accuracy) of five methods alongside with the baseline method (Majority)
    ### w.r.t. random folds, phylogeny-aware folds, and homology-aware folds for Supplementary File 1.
    ###################################################################################################################
    ####################################################################################################################
    ### 2. Supplemental File 5
    ##  Performance(F1-macro, F1-positive, F1-negative, accuracy) of KMA-based Point-/ResFinder and
    # BLAST-based Point-/ResFinder on each combination’s whole dataset.
    ###################################################################################################################
    if f_table:
        foldset=['Random folds', 'Phylogeny-aware folds','Homology-aware folds']
        tool_list=['Point-/ResFinder' ,'Aytan-Aktug', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover','ML Baseline (Majority)']
        save_file_name=output_path+ 'Results/supplement_figures_tables/S1_cv_results.xlsx'
        src.benchmark_utility.lib.MAINtable.extract_info(level,species, f_all ,output_path,tool_list,foldset,save_file_name)

        foldset=['no folds']
        tool_list=['KMA-based Point-/ResFinder','Blastn-based Point-/ResFinder']
        save_file_name=output_path+ 'Results/supplement_figures_tables/S5_cv_results_resfinder.xlsx'
        src.benchmark_utility.lib.MAINtable.extract_info(level,species, f_all ,output_path,tool_list,foldset,save_file_name)
    ####################################################################################################################
    ### 3. Supplemental File 6.
    ##  3.11-3.12 tables for further analysis (ML comparison with ResFinder, ML baseline)
    ###3.2 Three pieces of software lists, under random folds, phylogeny-aware folds, and homology-aware folds.
    ##  Each list provides the software (or several) with highest F1-macro mean, and lowest F1-macro standard deviation
    ##  if several are with the same highest mean.
    ## Paired t-test
    ###################################################################################################################
    if f_table_analysis:

        # foldset=['Random folds', 'Phylogeny-aware folds','Homology-aware folds']
        # tool_list=[ 'Aytan-Aktug', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover']
        # com_tool_list=['Point-/ResFinder']
        # src.benchmark_utility.lib.table_analysis.extract_info(level,species,fscore, f_all ,output_path,'1',tool_list,foldset,com_tool_list)
        #
        # foldset=['Random folds', 'Phylogeny-aware folds','Homology-aware folds']
        # tool_list=['Point-/ResFinder','Aytan-Aktug', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover']
        # com_tool_list=['ML Baseline (Majority)']
        # src.benchmark_utility.lib.table_analysis.extract_info(level,species,fscore,  f_all ,output_path,'1',tool_list,foldset,com_tool_list)

        foldset=['Random folds', 'Phylogeny-aware folds','Homology-aware folds']
        tool_list=['Point-/ResFinder', 'Aytan-Aktug', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover']
        src.benchmark_utility.lib.table_analysis.extract_info(level,species, fscore, f_all ,output_path,'2',tool_list,foldset,'')
        #
        # foldset=['Homology-aware folds']
        # tool_list=['Point-/ResFinder', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover','Single-species-antibiotic Aytan-Aktug',
        #            'Single-species multi-antibiotics Aytan-Aktug','Discrete databases multi-species model',
        #         'Concatenated databases mixed multi-species model', 'Concatenated databases leave-one-out multi-species model']
        # src.benchmark_utility.lib.table_analysis.extract_info(level,species,fscore,  f_all ,output_path,'2',tool_list,foldset,'')
        #

        ###paired t-test
        # foldset=['Random folds', 'Phylogeny-aware folds','Homology-aware folds']
        # tool_list=['Point-/ResFinder', 'Aytan-Aktug', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover']
        # src.benchmark_utility.lib.table_analysis.extract_info(level,species,fscore,f_all ,output_path,'3',tool_list,foldset,'')


    ### Clinical-oriented performance analysis
    ###  compared the software performance regarding F1-negative and precision-negative
    if f_clinical_analysis:

        foldset=['Random folds', 'Phylogeny-aware folds','Homology-aware folds']
        tool_list=[ 'Aytan-Aktug', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover']
        com_tool_list=['Point-/ResFinder']
        src.benchmark_utility.lib.table_analysis.extract_info(level,species,fscore, f_all ,output_path,'1',tool_list,foldset,com_tool_list)

        foldset=['Random folds', 'Phylogeny-aware folds','Homology-aware folds']
        tool_list=['Point-/ResFinder','Aytan-Aktug', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover']
        com_tool_list=['ML Baseline (Majority)']
        src.benchmark_utility.lib.table_analysis.extract_info(level,species,fscore, f_all ,output_path,'1',tool_list,foldset,com_tool_list)

        foldset=['Random folds', 'Phylogeny-aware folds','Homology-aware folds']
        tool_list=['Point-/ResFinder', 'Aytan-Aktug', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover']
        src.benchmark_utility.lib.table_analysis.extract_info(level,species, fscore, f_all ,output_path,'2',tool_list,foldset,'')

        foldset=['Homology-aware folds']
        tool_list=['Point-/ResFinder', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover','Single-species-antibiotic Aytan-Aktug',
                   'Single-species multi-antibiotics Aytan-Aktug','Discrete databases multi-species model',
                'Concatenated databases mixed multi-species model', 'Concatenated databases leave-one-out multi-species model']
        src.benchmark_utility.lib.table_analysis.extract_info(level,species,fscore, f_all ,output_path,'2',tool_list,foldset,'')


    ####################################################################################################################
    ### 4.  Fig. 4 & Supplemental File 2 Fig. S5-7 Error bar plot.
    ## Software performance (F1-macro  with 95% with confidence intervals) comparison on combinations sharing the same antibiotic,
    ## but different species.
    ## e.g. E. coli-ATM and K. pneumoniae-ATM combinations under homology-aware folds.
    ####################################################################################################################
    if f_anti:
        tool_list=['Point-/ResFinder','Aytan-Aktug',  'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover','ML Baseline (Majority)']
        f_phylotree=False
        f_kma=False
        src.benchmark_utility.lib.ByAnti_errorbar.ComByAnti(level,tool_list,fscore, f_phylotree,f_kma,output_path)
        f_phylotree=False
        f_kma=True
        src.benchmark_utility.lib.ByAnti_errorbar.ComByAnti(level,tool_list,fscore, f_phylotree,f_kma,output_path)
        f_phylotree=True
        f_kma=False
        src.benchmark_utility.lib.ByAnti_errorbar.ComByAnti(level,tool_list,fscore, f_phylotree,f_kma,output_path)


        f_phylotree=False
        f_kma=False
        src.benchmark_utility.lib.ByAnti_errorbar_each.ComByAnti(level,tool_list,fscore, f_phylotree,f_kma,output_path)



    ####################################################################################################################
    ### 5.  Fig. 5 & Supplemental File 2 Fig. S3.  Radar plot.
    ## 5.1 Software performance (F1-macro)  of 5 methods + ML baseline.
    ## 5.2 Fig. 5 e.g. E. coli phylogeny-aware folds
    ## 5.3  Supplemental File 2 Fig. S4.  Point-/ResFinder performance (F1-macro) on the whole data set (blue) and by 3 sets of folds
    ####################################################################################################################
    if f_species:
        save_file_name=output_path+ 'Results/supplement_figures_tables/S3_Com_BySpecies_Radar'
        tool_list=['Point-/ResFinder', 'Aytan-Aktug', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover','ML Baseline (Majority)']
        transparent=1
        src.benchmark_utility.lib.Com_BySpecies.draw(tool_list,level,species, fscore, f_all,transparent,output_path,save_file_name)
        #---------------------------------------------------------------------------------------------------------------
        save_file_name=output_path+ 'Results/supplement_figures_tables/S4_Com_resfinder_resfinderFolds'
        tool_list=['Point-/ResFinder not valuated by folds','Point-/ResFinder evaluated by folds']
        transparent=0.7
        src.benchmark_utility.lib.Com_BySpecies.draw(tool_list,level,species, fscore, f_all,transparent,output_path,save_file_name)
        ###---------------------------------------------------------------------------------------------------------------
        ### zoom-in for one e.g.
        save_file_name=output_path+ 'Results/final_figures_tables/F5_radarplot_'
        tool_list=['Point-/ResFinder', 'Aytan-Aktug', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover','ML Baseline (Majority)']
        species=['Escherichia coli']
        f_phylotree=True
        f_kma=False
        src.benchmark_utility.lib.Com_BySpecies_each.draw(tool_list,level,species, fscore,f_phylotree,f_kma,output_path,save_file_name)

    ####################################################################################################################
    ### 6.  Fig. 6 & Supplemental File 2 Fig. S8-11.   Paired box plot
    ## 6.11 Performance change for each species. Performance change for each software. Performance change for each species-antibiotic combination.
    ## 6.12  Performance change for ML baseline.
    ## 6.2  Performance (std) change.
    ####################################################################################################################
    if f_robust:
        ###by species and antibiotics, respectively. mean.  Fig. 6
        # src.benchmark_utility.lib.pairbox.extract_info(level,species, fscore,f_all,'1','mean',output_path)
        # src.benchmark_utility.lib.pairbox_separateFig.extract_info(level,species, fscore,f_all,'1','mean',output_path)

        # ###by species and antibiotics, respectively. std. Supplemental File 2 Fig. S8
        # src.benchmark_utility.lib.pairbox.extract_info(level,species, fscore,f_all,'1','std',output_path)
        # ###by species-anti combinations  Supplemental File 2 Fig. S10-11
        # src.benchmark_utility.lib.pairbox.extract_info(level,species, fscore,f_all,'3','mean',output_path)
        # src.benchmark_utility.lib.pairbox.extract_info(level,species, fscore,f_all,'3','std',output_path)
        #
        # ###only for ML baseline majority Supplemental File 2 Fig. S9
        src.benchmark_utility.lib.pairbox_majority.extract_info(level,species, fscore,f_all,'1','mean',output_path)
        src.benchmark_utility.lib.pairbox_majority.extract_info(level,species, fscore,f_all,'1','std',output_path)


        ### April 2023 newly added. across each antibiotics.
        # src.benchmark_utility.lib.pairbox_antibiotic.extract_info(level,species, fscore,f_all,'1','mean',output_path)





    ####################################################################################################################
    ### 7.  Fig. 1  sample size
    ####################################################################################################################
    if f_sample:
        save_file_name=output_path+ 'Results/final_figures_tables/samplesize.png'
        src.benchmark_utility.lib.SampleSize.extract_info(level, save_file_name)

    ####################################################################################################################
    ### 6.  Fig. 3  heatmap
    ####################################################################################################################
    if f_hmap:
        save_file_name=output_path+ 'Results/final_figures_tables/'
        foldset=['Random folds', 'Phylogeny-aware folds','Homology-aware folds']
        tool_list=['Point-/ResFinder','Aytan-Aktug', 'Seq2Geno2Pheno','PhenotypeSeeker', 'Kover']

        src.benchmark_utility.lib.heatmap.extract_info(level,fscore,foldset,tool_list,output_path,save_file_name)




if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--l', '--level', default='loose', type=str, required=False,
                        help='Quality control: strict or loose')
    parser.add_argument("-cv", "--cv_number", default=10, type=int,
                        help='CV splits number')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'seudomonas aeruginosa\' \
         \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
         \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument('-f_species', '--f_species', dest='f_species', action='store_true',
                        help='benchmarking by species.')
    parser.add_argument('-fscore', '--fscore', default='f1_macro', type=str, required=False,
                        help='the score used to choose the best classifier for each antibiotic. \
                        Can be one of: \'f1_macro\',\'f1_positive\',\'f1_negative\',\'accuracy\','
                             '\'clinical_f1_negative\',\'clinical_precision_neg\',\'clinical_recall_neg\'')
    parser.add_argument('-f_robust', '--f_robust', dest='f_robust', action='store_true',
                        help='plotting pairbox w.r.t. 3 folds split methods.')
    parser.add_argument('-f_sample', '--f_sample', dest='f_sample', action='store_true',
                        help='Plot sample size bar graph.')
    parser.add_argument('-f_table', '--f_table', dest='f_table', action='store_true',
                        help='Performance scores for Supplemental File 1.')
    parser.add_argument('-f_table_analysis', '--f_table_analysis', dest='f_table_analysis', action='store_true',
                        help='Winner lists and tables for further analysis.')
    parser.add_argument('-f_clinical_analysis', '--f_clinical_analysis', dest='f_clinical_analysis', action='store_true',
                        help='Comparision based on F1-negative and precision-negative.')
    parser.add_argument('-f_anti', '--f_anti', dest='f_anti', action='store_true',
                        help='Compare software performance on combinations sharing antibiotics but different species.')
    parser.add_argument('-f_hmap', '--f_hmap', dest='f_hmap', action='store_true',
                        help='Plotting heatmaps of F1-macro mean.')
    parser.add_argument('-o', '--output_path', default='./', type=str, required=False,
                        help='Directory to store CV scores.')
    parsedArgs = parser.parse_args()

    extract_info(parsedArgs.l,parsedArgs.species,parsedArgs.fscore,parsedArgs.f_all,parsedArgs.f_species,parsedArgs.f_anti,
                 parsedArgs.f_robust,parsedArgs.f_sample,parsedArgs.f_table,parsedArgs.f_table_analysis,
                 parsedArgs.f_clinical_analysis,parsedArgs.f_hmap,parsedArgs.output_path)
