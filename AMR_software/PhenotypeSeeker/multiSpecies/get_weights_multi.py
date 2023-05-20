__author__ = "Erki Aun"
__version__ = "0.7.3"

import sys
import os
# sys.path.append('../')
sys.path.insert(0, os.getcwd())
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, _DistanceMatrix
from ete3 import Tree
import Bio
import numpy as np
import pandas as pd
import argparse
from src.amr_utility import name_utility


# Functions for calculating the mash distances and GSC weights for
#This scripts output distances.mat,  weight.npy for each antibiotics and each  outer CV loop.




def get_weights(meta_txt,cv,names_of_samples_list):

    temp_addr= meta_txt+'_temp/CV_tr'+str(cv)+"/"
    # cls.get_mash_distances()
    _mash_output_to_distance_matrix(names_of_samples_list, temp_addr+"mash_distances.mat",temp_addr)
    dist_mat = _distance_matrix_modifier(temp_addr+"distances.mat")
    _distance_matrix_to_phyloxml(names_of_samples_list, dist_mat,temp_addr)
    _phyloxml_to_newick(temp_addr+"tree_xml.txt",temp_addr)
    #Calculating the GSC weights from mash distance matrix
    weights =  GSC_weights_from_newick(temp_addr+"tree_newick.txt", normalize="mean1")

    return weights





def _mash_output_to_distance_matrix(names_of_samples_list, mash_distances,temp_addr):  # mash_distances.mat
    with open(mash_distances) as f1:
        with open(temp_addr+"distances.mat", "w+") as f2:
            counter = 0

            f2.write(names_of_samples_list[counter])
            for line in f1:
                distance = line.split()[2]
                f2.write("\t" + distance)
                counter += 1
                if counter % (len(names_of_samples_list)) == 0:
                    if counter !=len(names_of_samples_list) ** 2:
                        f2.write(
                            "\n" + names_of_samples_list[counter //len(names_of_samples_list)]
                        )



def _distance_matrix_modifier(distance_matrix):
    # Modifies distance matrix to be suitable argument
    # for Bio.Phylo.TreeConstruction._DistanceMatrix function
    distancematrix = []
    with open(distance_matrix) as f1:
        counter = 2
        for line in f1:
            line = line.strip().split()
            distancematrix.append(line[1:counter])
            counter += 1
    for i in range(len(distancematrix)):
        for j in range(len(distancematrix[i])):
            distancematrix[i][j] = float(distancematrix[i][j])
    return (distancematrix)



def _distance_matrix_to_phyloxml(samples_order, distance_matrix,temp_addr):
    # Converting distance matrix to phyloxml

    dm = _DistanceMatrix(samples_order, distance_matrix)
    tree_xml = DistanceTreeConstructor().nj(dm)
    with open(temp_addr+"tree_xml.txt", "w+") as f1:
        Bio.Phylo.write(tree_xml, f1, "phyloxml")



def _phyloxml_to_newick(phyloxml,temp_addr):
    # Converting phyloxml to newick
    with open(temp_addr+"tree_newick.txt", "w+") as f1:  # The file is created if it does not exist, otherwise it is truncated.
        Bio.Phylo.convert(phyloxml, "phyloxml", f1, "newick")



def GSC_weights_from_newick(newick_tree, normalize="sum1"):
    # Calculating Gerstein Sonnhammer Coathia weights from Newick
    # string. Returns dictionary where sample names are keys and GSC
    # weights are values.
    cls_tree = Tree(newick_tree, format=1)
    clip_branch_lengths(cls_tree)
    set_branch_sum(cls_tree)
    set_node_weight(cls_tree)

    weights = {}
    for leaf in cls_tree.iter_leaves():
        weights[leaf.name] = leaf.NodeWeight
    if normalize == "mean1":
        weights = {k: v * len(weights) for k, v in weights.items()}
    return (weights)



def clip_branch_lengths(tree, min_val=1e-9, max_val=1e9):
    for branch in tree.traverse("levelorder"):
        if branch.dist > max_val:
            branch.dist = max_val
        elif branch.dist < min_val:
            branch.dist = min_val



def set_branch_sum(tree):
    total = 0
    for child in tree.get_children():
        set_branch_sum(child)
        total += child.BranchSum
        total += child.dist
    tree.BranchSum = total



def set_node_weight(tree):
    parent = tree.up
    if parent is None:
        tree.NodeWeight = 1.0
    else:
        tree.NodeWeight = parent.NodeWeight * \
                          (tree.dist + tree.BranchSum) / parent.BranchSum
    for child in tree.get_children():
        set_node_weight(child)

def extract_info(s,anti,level,cv,f_all,f_phylotree,f_kma,temp_path):

    s=[str(i.replace("_", " ")) for i in s]
    main_meta,_=name_utility.GETname_main_meta(level)
    data = pd.read_csv(main_meta, index_col=0, dtype={'genome_id': object}, sep="\t")
    data = data[data['number'] != 0]  # drop the species with 0 in column 'number'.

    if f_all == False:
        data = data.loc[s, :]
    # --------------------------------------------------------
    df_species = data.index.tolist()
    antibiotics = data['modelling antibiotics'].tolist()
    # print(data)
    for species, antibiotics in zip(df_species, antibiotics):

        ### _,_,meta_txt,_ = name_utility.GETname_model2('phenotypeseeker',level, species, anti,'',temp_path,f_kma,f_phylotree)
        _,_,meta_txt,_ =name_utility.GETname_model3('phenotypeseeker',level, species, anti,'',temp_path)
        names_of_samples_list = np.genfromtxt(meta_txt+'_Train_'+str(cv)+'_id2', dtype= "str")
        names_of_samples_list=names_of_samples_list.tolist()
        weights_dic=get_weights(meta_txt,cv,names_of_samples_list)# in the form of dic.
        np.save(meta_txt+'_temp/CV_tr'+str(cv)+'/weight', weights_dic)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-temp', '--temp_path', default='./', type=str, required=False,
                        help='Directory to store temporary files.')
    parser.add_argument('-f_phylotree', '--f_phylotree', dest='f_phylotree', action='store_true',
                        help=' phylo-tree based cv folders.')
    parser.add_argument('-f_kma', '--f_kma', dest='f_kma', action='store_true',
                        help='kma based cv folders.')
    parser.add_argument('-anti', '--anti',  type=str,required=True,
                        help='antibiotics.')
    parser.add_argument('-l', '--level', default='loose', type=str,
                        help='Quality control: strict or loose')
    parser.add_argument("-cv", "--cv", required=True,
                        help='CV splits you are working now')
    parser.add_argument('-f_all', '--f_all', dest='f_all', action='store_true',
                        help='all the possible species, regarding multi-model.')
    parser.add_argument('-s', '--species', default=[], type=str, nargs='+', help='species to run: e.g.\'Pseudomonas aeruginosa\' \
                    \'Klebsiella pneumoniae\' \'Escherichia coli\' \'Staphylococcus aureus\' \'Mycobacterium tuberculosis\' \'Salmonella enterica\' \
                    \'Streptococcus pneumoniae\' \'Neisseria gonorrhoeae\'')

    # parser.add_argument('--n_jobs', default=1, type=int, help='Number of jobs to run in parallel.')
    parsedArgs = parser.parse_args()
    extract_info( parsedArgs.species,parsedArgs.anti,parsedArgs.level,parsedArgs.cv,parsedArgs.f_all,parsedArgs.f_phylotree,parsedArgs.f_kma,parsedArgs.temp_path)
