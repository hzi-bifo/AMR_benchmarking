#!/usr/bin/env/python

import os
import sys
# sys.path.append('../')
sys.path.insert(0, os.getcwd())
import numpy as np
import argparse
import getopt, sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")




def extract_info(input_path,files,output,spacer):

	files=np.genfromtxt(files, dtype='str')
	new_file = open(output, "w")#'w' for only writing (an existing file with the same name will be erased)
	for i in files:
		# print(i)
		open_scaf = open("%s/%s.fna" % (input_path, i), "r")
		scaf = open_scaf.readlines()

		##write the header in the file
		new_file.write(">%s" % i)
		new_file.write("\n")

		##merge the scaffolds with gaps, should be greater than the k-mer size
		for each in scaf:
			if ">" not in each:
				each_new = each.replace("\n","")
				new_file.write(each_new)
			else:
				new_file.write(str("N"*spacer))

	new_file.close()

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("-i", "--input", default=None, type=str, required=True,
						help='Scaffolds path.')
	parser.add_argument("-f","--file_list", default=None, type=str, required=True,
						help='List of samples.')
	parser.add_argument("-s", "--space", default=None, type=int, required=True,
						help='List of samples.')
	parser.add_argument("-o","--output", default=None, type=str, required=True,
						help='Output file names')

	parsedArgs = parser.parse_args()
	# parser.print_help()
	# print(parsedArgs)
	extract_info(parsedArgs.input,parsedArgs.file_list,parsedArgs.output,parsedArgs.space)

if __name__ == '__main__':
    main()
