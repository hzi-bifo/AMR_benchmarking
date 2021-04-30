#!/usr/bin/env/python
import getopt, sys
import warnings


if not sys.warnoptions:
    warnings.simplefilter("ignore")


##read commandline arguments first 
fullCmdArguments = sys.argv
##further argumnets 
argumentList = fullCmdArguments[1:]

unixOptions = "e:t:s:p:a:b:c:d:h"
gnuOptions = ["ecoli=", "tuber=","salmo=", "stap=", "eout=", "tout=", "sout=", "staout=", "help"]


try:
	arguments, values = getopt.getopt(argumentList, unixOptions, gnuOptions)
except getopt.error as err:
	print(str(err))
	sys.exit(2)

for currentArgument, currentValue in arguments:
	if currentArgument in ("-h", "--help"):
		print("-e --ecoli = E. coli input")
		print("-t --tuber = M. tuberculosis input")
		print("-s --salmo = S. enterica input")
		print("-p --stap = S. aureus input")
		print("-a --eout = E. coli output")
		print("-b --tout = M. tuberculosis output")
		print("-c --sout = S. enterica output")
		print("-d --staout = S. aureus output")
		print("-h --help = show the help message (have fun!))")
		sys.exit()


def main():

	##this script four different species.
	##this script is generated for a specific case:
	##TB has six outputs,
	##others have 5 missing features in the outputs. 
	##species input and output files are required.
	##output file is a merged file. 

	import numpy as np

	###species#####

	for currentArgument, currentValue in arguments:
		if currentArgument in ("-e","--pointfin"):
			ecoli_path = currentValue

	for currentArgument, currentValue in arguments:
		if currentArgument in ("-t","--resfin"):
			tb_path = currentValue

	for currentArgument, currentValue in arguments:
		if currentArgument in ("-s","--pointfin"):
			sal_path = currentValue

	for currentArgument, currentValue in arguments:
		if currentArgument in ("-p","--resfin"):
			sta_path = currentValue

	for currentArgument, currentValue in arguments:
		if currentArgument in ("-a","--pointfin"):
			ecoli_path2 = currentValue

	for currentArgument, currentValue in arguments:
		if currentArgument in ("-b","--resfin"):
			tb_path2 = currentValue

	for currentArgument, currentValue in arguments:
		if currentArgument in ("-c","--pointfin"):
			sal_path2 = currentValue

	for currentArgument, currentValue in arguments:
		if currentArgument in ("-d","--resfin"):
			sta_path2 = currentValue

	###tuberculosis###
	tb = np.loadtxt("%s" % tb_path)
	tb_out = np.loadtxt("%s" % tb_path2)

	###E.coli####
	coli_feature = np.loadtxt("%s" % ecoli_path)#feature matrix
	coli_out = np.loadtxt("%s" % ecoli_path2)

	###Salmonella###
	sal = np.loadtxt("%s" % sal_path)
	sal_out = np.loadtxt("%s" % sal_path2)

	##Staphylococcus
	stap = np.loadtxt("%s" % sta_path)
	stap_out = np.loadtxt("%s" % sta_path2)




	coli_n = len(coli_feature)#sample numbers
	coli_feat_n = len(coli_feature[0])  # feature numbers

	tb_n = len(tb)
	tb_feat = len(tb[0])

	sal_feat = len(sal[0])
	sal_n = len(sal)

	stap_feat = len(stap[0])
	stap_n = len(stap)


	dist = ["coli"]*coli_n + ["tb"]*tb_n + ["sal"]*sal_n + ["stap"]*stap_n

	total_feat = coli_feat_n + tb_feat + sal_feat + stap_feat
	total_n = coli_n + tb_n + sal_n + stap_n 

	new_file = open("tb_ecoli_sal_stap.txt", "w")
	new_file_out = open("tb_ecoli_sal_stap_outputs.txt", "w")

	for each in range(total_n):#total samples length/number
		if dist[each] == "coli":
			for i in range(coli_feat_n):
				new_file.write(str(coli_feature[each][i]))
				new_file.write("\t")
			for z in range(tb_feat):
				new_file.write(str(0.0))
				new_file.write("\t")
			for x in range(sal_feat):
				new_file.write(str(0.0))
				new_file.write("\t")
			for w in range(stap_feat):
				new_file.write(str(0.0))
				new_file.write("\t")
			for m in range(5):
				new_file_out.write(str(-1))
				new_file_out.write("\t")

			new_file_out.write(str(coli_out[each]))
			new_file_out.write("\n")
			new_file.write("\n")

		elif dist[each] == "tb":
			for i in range(coli_feat_n):
				new_file.write(str(0.0))
				new_file.write("\t")
			for z in range(tb_feat):
				new_file.write(str(tb[each-coli_n][z]))
				new_file.write("\t")
			for x in range(sal_feat):
				new_file.write(str(0.0))
				new_file.write("\t")
			for w in range(stap_feat):
				new_file.write(str(0.0))
				new_file.write("\t")
			for m in [0,1,2,3,4,5]:
				new_file_out.write(str(tb_out[each-coli_n][m]))
				new_file_out.write("\t")
			new_file_out.write("\n")
			new_file.write("\n")



		elif dist[each] == "sal":
			for i in range(coli_feat_n):
				new_file.write(str(0.0))
				new_file.write("\t")
			for z in range(tb_feat):
				new_file.write(str(0.0))
				new_file.write("\t")
			for x in range(sal_feat):
				new_file.write(str(sal[each-coli_n-tb_n][x]))
				new_file.write("\t")
			for w in range(stap_feat):
				new_file.write(str(0.0))
				new_file.write("\t")
			for m in range(5):
				new_file_out.write(str(-1))
				new_file_out.write("\t")
			
			new_file_out.write(str(sal_out[each-coli_n-tb_n]))
			new_file_out.write("\n")
			new_file.write("\n")


		elif dist[each] == "stap":
			for i in range(coli_feat_n):
				new_file.write(str(0.0))
				new_file.write("\t")
			for z in range(tb_feat):
				new_file.write(str(0.0))
				new_file.write("\t")
			for x in range(sal_feat):
				new_file.write(str(0.0))
				new_file.write("\t")
			for w in range(stap_feat):
				new_file.write(str(stap[each-coli_n-tb_n-sal_n][w]))
				new_file.write("\t")
			for m in range(5):
				new_file_out.write(str(-1))
				new_file_out.write("\t")
			
			new_file_out.write(str(stap_out[each-coli_n-tb_n-sal_n]))
			new_file_out.write("\n")
			new_file.write("\n")

	new_file.close()
	new_file_out.close()
		
	
if __name__ == '__main__':
    main()	