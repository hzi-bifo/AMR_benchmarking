#!/usr/bin/env/python
import getopt, sys
import warnings
import argparse
import pandas as pd
import numpy as np
import zipfile
# if not sys.warnoptions:
#     warnings.simplefilter("ignore")

def extract_info(input_path,file,out_path,f_no_zip):
	####this script finds all the acquired genes and returns an output file
	output = open(out_path, "w")

	file_list=np.genfromtxt(file,dtype='str')
	# print(file_list)
	if f_no_zip==False:
		for strain in file_list:
			# print(strain)
			with zipfile.ZipFile("%s/%s.zip" % (input_path, strain)) as z:
				aq_open = z.open("%s/ResFinder_results_tab.txt" % strain, "r")
				# aq_open = open("%s/%s/ResFinder_results_tab.txt" % (input_path, str(strain)), "r")
				aq = aq_open.readlines()

				for i in range(1,len(aq)):
					output.write(str(strain))
					output.write("\t")
					output.write(aq[i].decode("utf-8").split("\t")[0])#delete ".decode("utf-8")" if results not zipped.
					output.write("\t")
					output.write(str(float(aq[i].decode("utf-8").split("\t")[3])/100))#delete ".decode("utf-8")" if results not zipped.
					output.write("\n")
				aq_open.close()

		output.close()
	else:
		for strain in file_list:

			aq_open = open("%s/%s/ResFinder_results_tab.txt" % (input_path, str(strain)), "r")
			aq = aq_open.readlines()

			for i in range(1, len(aq)):
				output.write(str(strain))
				output.write("\t")
				output.write(
					aq[i].split("\t")[0])  # delete ".decode("utf-8")" if results not zipped.
				output.write("\t")
				output.write(str(float(aq[i].split("\t")[
										   3]) / 100))  # delete ".decode("utf-8")" if results not zipped.
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
	parser.add_argument('-f_no_zip', '--f_no_zip', dest='f_no_zip', action='store_true',
						help=' Point/ResFinder results are not stored in zip format.')
	parsedArgs = parser.parse_args()
	# parser.print_help()
	# print(parsedArgs)
	extract_info(parsedArgs.respath,parsedArgs.list,parsedArgs.output,parsedArgs.f_no_zip)

if __name__ == '__main__':
    main()