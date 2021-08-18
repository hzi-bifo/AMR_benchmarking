import os

import numpy as np
import shutil
import ast
import statistics
import operator
import time
import amr_utility.name_utility
import amr_utility.graph_utility
import amr_utility.file_utility
import argparse
import amr_utility.load_data



def pre_bash_roary(df_species,antibiotics,path_large_temp,level,path_sequence):
    for species, antibiotics in zip(df_species, antibiotics):
        # for storage large prokka and roary temp files

        amr_utility.file_utility.make_dir(path_large_temp + '/results_roary/' + str(level))

        # ----------------by each anti, no need any more. Aug 10th.
        '''
        #for storage large prokka and roary temp files
        #1.
        run_file = open('./cv_folders/' + str(level) + '/' + str(species.replace(" ", "_")) + "_roary1.sh", "w")
        run_file.write("#!/bin/bash")
        run_file.write("\n")
        if path_sequence == '/vol/projects/BIFO/patric_genome':
            run_file = amr_utility.file_utility.hzi_cpu_header(run_file, str(species.replace(" ", "_")) +'run_roary1', 2)
        antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
        for anti in antibiotics:
            run(path_sequence,path_large_temp,species, anti, level, f_phylo_prokka, f_phylo_roary, f_pre_cluster, f_cluster,f_cluster_folders, run_file, f_res,
                f_merge_mution_gene,
                f_matching_io, f_merge_species, f_nn, cv, random, hidden, epochs, re_epochs, learning, f_scaler,
                f_fixed_threshold, f_nn_base,f_phylotree,f_optimize_score)
        run_file.write("echo \" one species finished. \"")
        run_file.close()
        #2.
        f_phylo_roary = 'step2'
        run_file = open('./cv_folders/' + str(level) + '/' + str(species.replace(" ", "_")) + "_roary2.sh", "w")
        run_file.write("#!/bin/bash")
        run_file.write("\n")
        if path_sequence == '/vol/projects/BIFO/patric_genome':
            run_file = amr_utility.file_utility.hzi_cpu_header(run_file,str(species.replace(" ", "_")) + 'run_roary2', 20)
        antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
        for anti in antibiotics:

            run(path_sequence,path_large_temp,species, anti, level, f_phylo_prokka, f_phylo_roary, f_pre_cluster, f_cluster,f_cluster_folders, run_file, f_res,
                f_merge_mution_gene,
                f_matching_io, f_merge_species, f_nn, cv, random, hidden, epochs, re_epochs, learning, f_scaler,
                f_fixed_threshold, f_nn_base,f_phylotree,f_optimize_score)
        run_file.write("echo \" All finished. \"")
        run_file.close()
        '''
        # ----------------------------------------------------------
        # Phylo-tree for all strains from one species.
        # 1.
        # rm existing files



        run_file = open('./cv_folders/' + str(level) + '/' + str(species.replace(" ", "_")) + "_roary1_all.sh", "w")
        run_file.write("#!/bin/bash")
        run_file.write("\n")
        if path_sequence == '/vol/projects/BIFO/patric_genome':
            run_file = amr_utility.file_utility.hzi_cpu_header(run_file,
                                                               str(species.replace(" ", "_")) + 'run_roary1_all', 2)
        # path_metadata_prokka='metadata/model/id_' + str(species.replace(" ", "_"))#some isolates are not selected for modelling.
        path_metadata_prokka ='cv_folders/'+str(level) + '/'+str(species.replace(" ", "_"))+'/id_list'  # all isolates are involved in at least one model.
        path_prokka = path_large_temp + '/prokka/' + str(species.replace(" ", "_"))

        path_large_temp_roary_all = path_large_temp + '/roary/' + str(level) + '/' + str(species.replace(" ", "_"))
        # rm the path_large_temp_roary_all folders if already exits
        if os.path.isdir(path_large_temp_roary_all):
            shutil.rmtree(path_large_temp_roary_all)
        amr_utility.file_utility.make_dir(path_large_temp_roary_all)
        path_roary_results = path_large_temp + '/results_roary/' + str(level) + '/' + str(species.replace(" ", "_"))
        run_file.write("\n")
        cmd = 'cat %s|' % path_metadata_prokka
        run_file.write(cmd)
        run_file.write("\n")
        run_file.write('while read i; do')
        run_file.write("\n")
        cmd = 'cp %s/${i}/*.gff %s/${i}.gff' % (path_prokka, path_large_temp_roary_all)
        run_file.write(cmd)
        run_file.write("\n")
        run_file.write('done')
        run_file.close()

        # 2.
        run_file = open('./cv_folders/' + str(level) + '/' + str(species.replace(" ", "_")) + "_roary2_all.sh", "w")
        run_file.write("#!/bin/bash")
        run_file.write("\n")
        if path_sequence == '/vol/projects/BIFO/patric_genome':
            run_file = amr_utility.file_utility.hzi_cpu_header(run_file,
                                                               str(species.replace(" ", "_")) + 'run_roary2_all',
                                                               20)
        run_file.write("\n")
        cmd = 'roary -p 20 -f %s -e --mafft -v %s/*.gff -g 700000' % (path_roary_results, path_large_temp_roary_all)
        run_file.write(cmd)
        run_file.write("\n")
        # run_file.write("wait")
        # run_file.write("\n")
        run_file.close()
        # 3.
        # R pakage phylo-trees for all strains from one species.
        run_file = open('./cv_folders/' + str(level) + '/' + str(species.replace(" ", "_")) + "_roaryTree_all.sh",
                        "w")
        run_file.write("#!/bin/bash")
        run_file.write("\n")
        if path_sequence == '/vol/projects/BIFO/patric_genome':
            run_file = amr_utility.file_utility.hzi_cpu_header(run_file,
                                                               str(species.replace(" ",
                                                                                   "_")) + 'run_roaryTree_all',
                                                               2)
        # path_metadata_prokka = 'metadata/model/id_' + str(species.replace(" ", "_"))
        path_large_temp_roary_all = path_large_temp + '/results_roary/' + str(level) + '/' + str(
            species.replace(" ", "_"))
        run_file.write("\n")
        cmd = 'Rscript --vanilla phylo_tree.r -f \'%s/core_gene_alignment.aln\' -o \'%s/nj_tree.newick\'' % (
            path_roary_results, path_roary_results)
        run_file.write(cmd)
        run_file.write("\n")
        # run_file.write("wait")
        # run_file.write("\n")
        run_file.close()

        '''
        # 3.fasttree. No use any more.
        f_phylo_roary = 'step3'
        run_file = open('./cv_folders/' + str(level) + '/' + str(species.replace(" ", "_")) + "_roary3.sh", "w")
        run_file.write("#!/bin/bash")
        run_file.write("\n")
        if path_sequence == '/vol/projects/BIFO/patric_genome':
            run_file = amr_utility.file_utility.hzi_cpu_header2(run_file,  str(species.replace(" ", "_")) + 'run_roary3', 1)
            run_file.write("\n")
        # run_file.write("export OMP_NUM_THREADS=20")
        run_file.write("\n")
        antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
        for anti in antibiotics:
            run(path_sequence, path_large_temp, species, anti, level, f_phylo_prokka, f_phylo_roary,
                f_pre_cluster, f_cluster, f_cluster_folders, run_file, f_res,
                f_merge_mution_gene,
                f_matching_io, f_merge_species, f_nn, cv, random, hidden, epochs, re_epochs, learning, f_scaler,
                f_fixed_threshold, f_nn_base,f_phylotree,f_optimize_score)
        # run_file.write("echo \" running \"")
        run_file.write("\n")
        run_file.write("wait")
        run_file.close()
        f_phylo_roary = 'step4'# R package phylo-trees
        run_file = open('./cv_folders/' + str(level) + '/' + str(species.replace(" ", "_")) + "_roary4.sh", "w")
        run_file.write("#!/bin/bash")
        run_file.write("\n")
        if path_sequence == '/vol/projects/BIFO/patric_genome':
            run_file = amr_utility.file_utility.hzi_cpu_header2(run_file,
                                                                str(species.replace(" ", "_")) + 'run_roary4',
                                                                1)
            run_file.write("\n")
        # run_file.write("export OMP_NUM_THREADS=20")
        run_file.write("\n")
        antibiotics, ID, Y = amr_utility.load_data.extract_info(species, False, level)
        for anti in antibiotics:
            run(path_sequence, path_large_temp, species, anti, level, f_phylo_prokka, f_phylo_roary,
                f_pre_cluster, f_cluster, f_cluster_folders, run_file, f_res,
                f_merge_mution_gene,
                f_matching_io, f_merge_species, f_nn, cv, random, hidden, epochs, re_epochs, learning, f_scaler,
                f_fixed_threshold, f_nn_base,f_phylotree, f_optimize_score)

        '''
    '''
    #4. join files that belong to the same species. No need any more. Aug 10th.
    # -------------------------------------------------------------------------------------------------------
    ID_files=[]
    for species, antibiotics in zip(df_species, antibiotics):
        ID_files.append('./cv_folders/' + str(level) + '/' + str(species.replace(" ", "_")) + "_roary1.sh")

    with open('./cv_folders/run_roary1.sh', 'w') as outfile:
        outfile.write("#!/bin/bash")
        outfile.write("\n")
        if path_sequence == '/vol/projects/BIFO/patric_genome':
            outfile = amr_utility.file_utility.hzi_cpu_header(outfile, 'run_roary1', 2)
        for names in ID_files:
            # Open each file in read mode
            with open(names) as infile:
                outfile.write(infile.read())
            outfile.write("\n")
    '''
    # -------------------------------------------------------------------------------------------------------
    # run multiple species at the same bash, only for hzi machine
    # and need to uncomment previous "#!/bin/bash"
    '''
    if path_sequence == '/vol/projects/BIFO/patric_genome':
        ID_files=[]
        for species, antibiotics in zip(df_species, antibiotics):
            ID_files.append('./cv_folders/' + str(level) + '/' + str(species.replace(" ", "_")) + "_roary2.sh")

        with open('./cv_folders/run_roary2.sh', 'w') as outfile:
            outfile.write("#!/bin/bash")

            outfile.write("\n")
            if path_sequence == '/vol/projects/BIFO/patric_genome':
                outfile = amr_utility.file_utility.hzi_cpu_header(outfile, 'run_roary2', 20)
            outfile.write("\n")
            # outfile.write("export OMP_NUM_THREADS=20")
            outfile.write("\n")
            for names in ID_files:
                # Open each file in read mode
                with open(names) as infile:
                    outfile.write(infile.read())
                outfile.write("\n")
            # -------------------------------------------------------------------------------------------------------
            ID_files = []
            for species, antibiotics in zip(df_species, antibiotics):
                ID_files.append(
                    './cv_folders/' + str(level) + '/' + str(species.replace(" ", "_")) + "_roary3.sh")

            with open('./cv_folders/run_roary3.sh', 'w') as outfile:
                outfile.write("#!/bin/bash")

                outfile.write("\n")
                if path_sequence == '/vol/projects/BIFO/patric_genome':
                    outfile = amr_utility.file_utility.hzi_cpu_header(outfile, 'run_roary3', 20)
                outfile.write("\n")
                outfile.write("export OMP_NUM_THREADS=20")
                outfile.write("\n")
                for names in ID_files:
                    # Open each file in read mode
                    with open(names) as infile:
                        outfile.write(infile.read())
                    outfile.write("\n")

    '''