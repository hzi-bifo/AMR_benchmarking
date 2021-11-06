#!/usr/bin/env/python
# @Author: kxh
# @Date:   Nov 3rd 2021
# @Last Modified by:   
# @Last Modified time:
import hashlib
import os
import collections
import pandas as pd
import numpy as np
from Bio import AlignIO
from shutil import copyfile
from Bio import SeqIO
'''
This script maps name of each sample to a new str with hashlib
Only for the sake of phylo-tree based CV folders generation.
env: nakemake_env
'''

species=['Salmonella enterica',"Neisseria gonorrhoeae","Acinetobacter baumannii",'Pseudomonas aeruginosa']
#todo 'Mycobacterium tuberculosis'

for S in species:
    print(S)
    s=str(S.replace(" ", "_"))
    #replace names in target files
    align_f='/net/sgi/metagenomics/data/khu/benchmarking/seq2geno/log/temp/loose/'+s+'/results/denovo/roary/core_gene_alignment.aln'
    temp = '/net/sgi/metagenomics/data/khu/benchmarking/seq2geno/log/temp/loose/' + s + '/results/denovo/roary/temp.aln'
    final='/net/sgi/metagenomics/data/khu/benchmarking/seq2geno/log/temp/loose/' + s + '/results/denovo/roary/core_gene_alignment_renamed.aln'
    map_file='/net/sgi/metagenomics/data/khu/benchmarking/seq2geno/log/temp/loose/' + s + '/results/denovo/roary/mapping.npy'
    copyfile(align_f, temp)
    original_file = temp
    corrected_file = final
    mapping_dict = collections.defaultdict(list)#mapping
    with open(original_file) as original, open(corrected_file, 'w') as corrected:
        records = SeqIO.parse(original_file, 'fasta')
        for record in records:
            # print(record.id)

            old_header = record.id
            new = hashlib.md5(old_header.encode('utf-8')).hexdigest()
            mapping_dict[old_header].append(new)#mapping
            record.id = new
            record.description = new  # <- Add this line
            # print(record.id)
            SeqIO.write(record, corrected, 'fasta')
    np.save(map_file, mapping_dict)#
    # Load------------
    # read_dictionary = np.load('my_file.npy',allow_pickle='TRUE').item()
    # print(read_dictionary['hello']) # displays "world"
    #-----------------
    os.remove(temp)


# copyfile("core_gene_alignment.aln", "temp.aln")
# # handle = open("new.aln","w+")
# # for alignment in AlignIO.parse("target.aln", "fasta"):
# #     # print("Alignment of length %i" % alignment.get_alignment_length())
# #     print(alignment)
# #     print(len(alignment))
# #     for record in alignment:
# #         # print("%s %s %s" % (record.seq, record.name, record.id))
# #         print(record.id,record.name)
# #         old_header=record.id
# #         new_header=hashlib.md5(old_header.encode('utf-8')).hexdigest()
# #         record.id=new_header
# #         record.description =new_header
# original_file ="temp.aln"
# corrected_file = "target.aln"
#
# with open(original_file) as original, open(corrected_file, 'w') as corrected:
#     records = SeqIO.parse(original_file, 'fasta')
#     for record in records:
#         print(record.id)
#         old_header = record.id
#         new= hashlib.md5(old_header.encode('utf-8')).hexdigest()
#         record.id = new
#         record.description =new # <- Add this line
#         print(record.id)
#         SeqIO.write(record, corrected, 'fasta')