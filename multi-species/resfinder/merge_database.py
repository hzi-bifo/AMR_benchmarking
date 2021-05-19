
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import numpy as np
import subprocess

def make_dir(name):
    logDir = os.path.join(name)
    if not os.path.exists(logDir):
        try:
            os.makedirs(logDir)
        except OSError:
            print("Can't create logging directory:", logDir)
def main():
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    merge='db_pointfinder/merge_species'
    make_dir(merge)
    # path of the new merged database
    path_merge=os.path.join(fileDir,merge)
    path_merge=os.path.abspath(os.path.realpath(path_merge))

    #path of pointfinder
    database_path = os.path.join(fileDir, 'db_pointfinder')
    database_path = os.path.abspath(os.path.realpath(database_path))

    #path of the gene summery file of new merged database
    path_gene_summary = os.path.join(fileDir, 'db_pointfinder/merge_species/genes.txt')
    path_gene_summary=os.path.abspath(os.path.realpath(path_gene_summary))

    # path of the RNA_gene summery file of new merged database
    path_RNAgene_summary = os.path.join(fileDir, 'db_pointfinder/merge_species/RNA_genes.txt')
    path_RNAgene_summary = os.path.abspath(os.path.realpath(path_RNAgene_summary))

    # resistens-overview.txt file merge
    resistens_file_summary = os.path.join(fileDir, 'db_pointfinder/merge_species/resistens-overview.txt')
    resistens_file_summary = os.path.abspath(os.path.realpath(resistens_file_summary))


    print(database_path)
    out_lst = []
    for entry in os.scandir(database_path):
        if not entry.name.startswith('.') and entry.is_dir():
            out_lst.append(entry.name)
    out_lst.remove('merge_species')
    # print(out_lst)

    open(path_gene_summary, "w")
    open(path_RNAgene_summary, "w")
    open(resistens_file_summary, "w")
    #copy a resistens-overview.txt from a random species
    cmd0 = 'cp ./db_pointfinder/escherichia_coli/resistens-overview.txt %s' % (path_merge)
    subprocess.run(cmd0, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    for species in out_lst:
        #get al the genes in the species file, plus one or two summary txt files.
        path_gene_summary_sub=os.path.join(database_path, species,'genes.txt')
        path_RNAgene_summary_sub=os.path.join(database_path, species,'RNA_genes.txt')
        print('path_gene_summary_sub',path_gene_summary_sub)
        path_gene= os.path.join(database_path, species)
        file_names = os.listdir(os.path.abspath(os.path.realpath(path_gene)))

        list_genes=[]
        print(path_gene)
        print(file_names)
        for item in file_names:
           if ".fsa" in item:
                list_genes.append(item)

        print(species,'**',list_genes)

        for gene_file in list_genes:

            print(gene_file)
            gene_name = gene_file.split('.')[0]
            cmd='cp %s/%s %s/%s_%s.fsa'%(path_gene,gene_file,path_merge,gene_name,species)
            subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,check=True)
            # modify the header in each fsa file
            with open(('%s/%s_%s.fsa') %(path_merge,gene_name,species))as f:
                lines = f.readlines()

            lines[0] = ">%s_%s\n"%(gene_name,species)

            with open(('%s/%s_%s.fsa') %(path_merge,gene_name,species), "w") as f:
                f.writelines(lines)
        #summerize gene.txt

        if 'genes.txt' in file_names:
            with open(path_gene_summary, 'a') as outfile:
                with open(path_gene_summary_sub) as infile:
                    line_list = infile.readlines()
                    for line in line_list:
                        line_new = line.split('\n')[0]
                        outfile.write(line_new + "_" + species)
                        outfile.write("\n")

        # summerize RNA_genes.txt
        if 'RNA_genes.txt' in file_names:
            with open(path_RNAgene_summary, 'a') as outfile:
                with open(path_RNAgene_summary_sub) as infile:
                    line_list=infile.readlines()
                    for line in line_list:
                        line_new=line.split('\n')[0]
                        outfile.write(line_new+"_"+species)
                        outfile.write("\n")






if __name__ == '__main__':
    main()