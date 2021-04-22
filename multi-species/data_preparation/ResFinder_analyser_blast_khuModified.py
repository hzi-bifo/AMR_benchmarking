#!/usr/bin/env/python
import getopt, sys
import warnings
import argparse
import pandas as pd
import numpy as np
# if not sys.warnoptions:
#     warnings.simplefilter("ignore")

def extract_info(input_path,file,out_path):
	####this script finds all the acquired genes and returns an output file
	output = open(out_path, "w")

	file_list=np.genfromtxt(file,dtype='str')
	# print(file_list)
	for strain in file_list:
		# print(strain)
		aq_open = open("%s/%s/ResFinder_results_tab.txt" % (input_path, str(strain)), "r")
		aq = aq_open.readlines()
		for i in range(1,len(aq)):
			output.write(str(strain))
			output.write("\t")
			output.write(aq[i].split("\t")[0])
			output.write("\t")
			output.write(str(float(aq[i].split("\t")[3])/100))
			output.write("\n")
		aq_open.close()

	output.close()


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