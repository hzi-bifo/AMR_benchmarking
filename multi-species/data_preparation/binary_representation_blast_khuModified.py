#!/usr/bin/env/python
import getopt, sys
import warnings
import argparse
import pandas as pd
import numpy as np
import os
import collections
import re
# if not sys.warnoptions:
#     warnings.simplefilter("ignore")





def extract_info(input_path,file,out_path):

	##this script analyses all the pointfinder results 
	##this script produces an output file including all the mutated positions. 
	##According to the absence presence of the mutation, each feature is marked as 1 or 0. 
	##What this script does not provide that which kind of mutation has? all the mutations are taken into consideration equally.
	##Last updated ; 04.03.2019

	##be sure there is no undesired file

	points = np.genfromtxt(file, dtype='str')
	dict_mutations = collections.defaultdict(list)
	all_samples = collections.defaultdict(list)
	# all_aa = []
	for sample in points:
		list_point = os.listdir("%s/%s" % (input_path, sample))
		# print(list_point)
		for item in list_point:
			if "PointFinder_results.txt" in item:
				data_tsv = open("%s/%s/%s" % (input_path, sample, item), "r")
				tsv = data_tsv.readlines()
				temp_dict = collections.defaultdict(list)
				for each in tsv[1:]:
					each = each.replace(" promotor", "")
					each = each.replace(" promoter", "")
					splitted1 = each.split("\t")[0]
					splitted = splitted1.split(" ")[0:2]
					position = re.findall("(\-?\d+)", splitted[1])
					# print(splitted,position)
					dict_mutations[splitted[0].lower()].append(position[0])
					temp_dict[splitted[0].lower()].append(position[0])
				all_samples[sample].append(temp_dict)

		# print(dict_mutations)
		
		results = open(out_path, "w")

	dict_mutations2 = collections.defaultdict(list)

	for each in dict_mutations:
		uniq_res = list(set(list(dict_mutations[each])))
		dict_mutations2[each].append(uniq_res)

	all_genes = []
	for each in dict_mutations2:
		all_genes.append(each)
 
	for k in all_samples:
		results.write(str(k))
		results.write("\t")
		for g in all_genes:  
			if g in list(all_samples[k][0]):
				for each in dict_mutations2[g][0]:
					if each in all_samples[k][0][g]:
						results.write("1")
						results.write("\t")
					else:
						results.write("-1")
						results.write("\t")
			else:
				if g in dict_mutations2:
					for a in dict_mutations2[g][0]:
						results.write("-1")
						results.write("\t")
		results.write("\n")
		
	results.close()

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("-r","--respath", default=None, type=str, required=True,
						help='Path to the Resfinder tool results, i.e. extracted AMR gene information.')
	parser.add_argument("-l","--list", default=None, type=str, required=True,
						help='List of samples.')
	parser.add_argument("-o","--output", default=None, type=str, required=True,
						help='Output file names')

	parsedArgs = parser.parse_args()
	# parser.print_help()
	# print(parsedArgs)
	extract_info(parsedArgs.respath,parsedArgs.list,parsedArgs.output)

if __name__ == '__main__':
    main()