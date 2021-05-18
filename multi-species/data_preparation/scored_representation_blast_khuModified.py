#!/usr/bin/env/python
import getopt, sys
import warnings
import numpy as np
import argparse
import os
import collections
import re
from Bio.SubsMat import MatrixInfo
import zipfile
# if not sys.warnoptions:
#     warnings.simplefilter("ignore")



def extract_info(input_path,file,out_path,mutcol):

	##this script analyses all the pointfinder results
	##takes the pointfinder input and does blosum coding
	##each feature has additional binary column which indicates absence/presence of the mutation

	##generates blosum62 matrix
	blosum = MatrixInfo.blosum62

	###nucleotide substitution matrix 
	##transition and transversion matrix
	nucleotide = {}
	nucleotide[("A", "A")] = 1
	nucleotide[("T", "T")] = 1
	nucleotide[("G", "G")] = 1
	nucleotide[("C", "C")] = 1

	nucleotide[("A", "T")] = -3
	nucleotide[("A", "C")] = -3
	nucleotide[("A", "G")] = -3

	nucleotide[("C", "T")] = -3
	nucleotide[("G", "T")] = -3

	nucleotide[("G", "C")] = -3
	
	nucleotide[("A", "N")] = 0
	nucleotide[("T", "N")] = 0
	nucleotide[("G", "N")] = 0
	nucleotide[("C", "N")] = 0
	nucleotide[("N", "N")] = 0

	list_point = np.genfromtxt(file, dtype='str')
	##be sure there is no undesired file


	##generate dictionaries, keys are genes, items are mutations
	dict_mutations = collections.defaultdict(list)
	all_samples = collections.defaultdict(list)
	all_aa = []

	for item in list_point:
		# if "PointFinder_results.txt" in os.listdir("%s/%s/" % (input_path, item)): #for not zipped res results
		with zipfile.ZipFile("%s/%s.zip" % (input_path, item)) as z:
			# data_tsv = open("%s/%s/PointFinder_results.txt" % (input_path, item), "r")#for not zipped res results
			data_tsv = z.open("%s/PointFinder_results.txt" % item, "r")
			tsv = data_tsv.readlines()
			temp_dict = collections.defaultdict(list)
			for each in tsv[1:]:
					each= each.decode("utf-8")#only needed if the res results are zipped.
					each = each.replace(" promotor", "")
					each = each.replace(" promoter", "")
					
					splitted1 = each.split("\t")[0]
					splitted = splitted1.split(" ")[0:2]
				
					if splitted[1][0] == "p":
						position = re.findall("(\-?\d+)", splitted[1])
						mutation = str(position[0]) + "_" + str(splitted[1][2]) + "_" + str(splitted[1][-1])
						
						if "ins" not in each and "del" not in each and "delins" not in each:
								dict_mutations[splitted[0].lower()].append(str(position[0]) + "_" + str(splitted[1][2]))
								temp_dict[splitted[0].lower()].append(mutation)
								all_aa.append(str(splitted[1][-1]))
						else:
								splitted2 = (each.split("\t")[1]).split("->")[0].replace(" ", "")
								splitted3 = (each.split("\t")[1]).split("->")[1].replace(" ", "")
								splitted4 = (each.split("\t")[2]).split("->")[0].replace(" ", "")
								splitted5 = (each.split("\t")[2]).split("->")[1].replace(" ", "") 
								if "*" in each:
									dict_mutations[splitted[0].lower()].append(str(position[0])+ "_" + splitted[1][2])
									temp_dict[splitted[0].lower()].append(str(position[0]))
								elif "del" in each:
									if len(position) > 1:
										i = 0
										for l in range(int(position[0]), int(position[1])+1):
											dict_mutations[splitted[0].lower()].append(str(l)+ "_" + splitted4[i])
											temp_dict[splitted[0].lower()].append(str(l))	
											i = i + 1
											

									else:
										dict_mutations[splitted[0].lower()].append(str(position[0])+ "_" + splitted[1][2])
										temp_dict[splitted[0].lower()].append(str(position[0]))
										
								elif "ins" in each:
									dict_mutations[splitted[0].lower()].append(str(position[0])+ "_" + splitted[1][2])
									temp_dict[splitted[0].lower()].append(str(position[0]))
									
								else:
									print(each)
							

					elif splitted[1][0] == "n":
						position = re.findall("(\-?\d+)", splitted[1])
						splitted2 = (each.split("\t")[1]).split("->")[0].replace(" ", "")
						splitted3 = (each.split("\t")[1]).split("->")[1].replace(" ", "")
						if "ins" not in each and "del" not in each and "delins" not in each:
							dict_mutations[splitted[0].lower()].append(str(position[0]) + "_" + str(splitted2) + "n")
							temp_dict[splitted[0].lower()].append(str(position[0]) + "_" + str(splitted2) + "_" + str(splitted3)+"n")	

						else:
							if "del" in each:
								if len(position) > 1:
									print(position)
									for l in range(int(position[0]), int(position[1])+1):
										dict_mutations[splitted[0].lower()].append(str(l) + "_" + str(splitted2) + "n")##wild type
										temp_dict[splitted[0].lower()].append(str(l))
										
								else:
									dict_mutations[splitted[0].lower()].append(str(position[0]) + "_" + str(splitted2) + "n")##wild type
									temp_dict[splitted[0].lower()].append(str(position[0]))
									
							else:
								dict_mutations[splitted[0].lower()].append(str(position[0]) + "_" + str(splitted2) + "n")
								temp_dict[splitted[0].lower()].append(str(position[0]))
								
											

					elif splitted[1][0] == "r":
						position = re.findall("(\d+)", splitted[1])
						splitted2 = (each.split("\t")[1]).split("->")[0].replace(" ", "")
						splitted3 = (each.split("\t")[1]).split("->")[1].replace(" ", "")
						
						if "ins" not in each and "del" not in each and "delins" not in each:
							#print(each)
							dict_mutations[splitted[0].lower()].append(str(position[0]) + "_" + str(splitted2) + "r")##wild type
							temp_dict[splitted[0].lower()].append(str(position[0]) + "_" + str(splitted2) + "_" + str(splitted3)+"r")	

						else:
							if "del" in each:
								#print(each)
								if len(position) > 1:
									for l in range(int(position[0]), int(position[1])+1):
										dict_mutations[splitted[0].lower()].append(str(l) + "_" + str(splitted2) + "r")##wild type
										temp_dict[splitted[0].lower()].append(str(l))
								else:
									dict_mutations[splitted[0].lower()].append(str(position[0]) + "_" + str(splitted2) + "r")##wild type
									temp_dict[splitted[0].lower()].append(str(position[0]))
									
							else:
								dict_mutations[splitted[0].lower()].append(str(position[0]) + "_" + str(splitted2) + "r")
								temp_dict[splitted[0].lower()].append(str(position[0]))
								
											
					else:
						print("unexpected mutation", each)
								
									
			sample = item.split("_")[0]
			all_samples[sample].append(temp_dict)

		
	results = open(out_path, "w")

	dict_mutations2 = collections.defaultdict(list)
	total = []
	for each in dict_mutations:
		uniq_res = list(set(list(dict_mutations[each])))
		dict_mutations2[each].append(uniq_res)
		for i in dict_mutations[each]:
			total.append(i)

	all_aa = list(set(all_aa))##unique amino acid
	 
	for k in all_samples:
		results.write(str(k))
		results.write("\t")
		for g in dict_mutations2:
			if g in list(all_samples[k][0]):
				for each in dict_mutations2[g][0]:
					if each.split("_")[0] in [i.split("_")[0] for i in all_samples[k][0][g]]:
						index_list = []
						for x in all_samples[k][0][g]:
							index_list.append(x.split("_")[0])
						target_aa = index_list.index(each.split("_")[0])
						aa = all_samples[k][0][g][target_aa]
						if each[-1] != "n" and each[-1] != "r":
							if len(aa.split("_")) > 2:
								mutation = aa.split("_")[2]
								wild = aa.split("_")[1]
								if (wild, mutation) in blosum:
									score = blosum[(str(wild), str(mutation))]
								elif (mutation, wild) in blosum:
									score =  blosum[(str(mutation), str(wild))]
								else:
									score = -5
							else: 
								score = -5

						elif each[-1] == "n":
							if len(aa.split("_")) > 2:
								mutation = aa.split("_")[2][0:-1]
								wild = aa.split("_")[1]
								if (wild.upper(), mutation.upper()) in nucleotide:
									score = nucleotide[(str(wild).upper(), str(mutation).upper())]
								elif (mutation.upper(), wild.upper()) in nucleotide:
									score = nucleotide[(str(mutation).upper(), str(wild).upper())]
								else:
									# print("nucleotide")
									# print(mutation, wild)
									pass
							else:					
								score = -5
						elif each[-1] == "r":
							if len(aa.split("_")) > 2:
								mutation = aa.split("_")[2][0:-1]
								wild = aa.split("_")[1]
								if (wild.upper(), mutation.upper()) in nucleotide:
									score = nucleotide[(str(wild).upper(), str(mutation).upper())]
								elif (mutation.upper(), wild.upper()) in nucleotide:
									score = nucleotide[(str(mutation).upper(), str(wild).upper())]
								else:
									# print("RNA")
									# print(mutation, wild)
									pass
							else:
								score = -5

						results.write(str(score))
						results.write("\t")
						if mutcol==True:
							results.write(str(1))
							results.write("\t")
						# for currentArgument, currentValue in arguments:
						# 	if currentArgument in ("-m","--mutcol"):
						# 		results.write(str(1))
						# 		results.write("\t")

					else:
						if each[-1] != "n" and each[-1] != "r":
							if "*" != each.split("_")[1] and "?" != each.split("_")[1]:
								score = blosum[each.split("_")[1], each.split("_")[1]]
							else:
								# print(each)
								score = -5
							results.write(str(score))

						elif each[-1] == "n":
							score = 1
							results.write(str(score))
						elif each[-1] == "r":
							score = 1
							results.write(str(score))

						if each.split("_")[1][0:-1].upper() == "N":
							print("check that again")

						results.write("\t")
						if mutcol==True:
							results.write(str(-1))
							results.write("\t")
						# for currentArgument, currentValue in arguments:
						# 	if currentArgument in ("-m","--mutcol"):
						# 		results.write(str(-1))
						# 		results.write("\t")
			else:
				if g in dict_mutations2:
					for f in dict_mutations2[g][0]:
						if f[-1] != "r" and f[-1] != "n":
							if "*" != f.split("_")[1] and "?" != f.split("_")[1]:
								score = blosum[f.split("_")[1], f.split("_")[1]]
							else:
								# print(f.split("_")[1])
								score = -5
							results.write(str(score))
						elif f[-1] == "r":
							score = 1
							results.write(str(score))
						elif f[-1] == "n":
							score = 1
							results.write(str(score))
						results.write("\t")
						if mutcol==True:
							results.write(str(-1))
							results.write("\t")
						# for currentArgument, currentValue in arguments:
						# 	if currentArgument in ("-m","--mutcol"):
						# 		results.write(str(-1))
						# 		results.write("\t")
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
	parser.add_argument("-m", "--mutcol", dest='mutcol',
						help='Add the binary mutation column. The scored representation merges with the scored and binary representation.',
						action='store_true', )  # default:false

	parsedArgs = parser.parse_args()
	# parser.print_help()
	# print(parsedArgs)
	extract_info(parsedArgs.respath,parsedArgs.list,parsedArgs.output,parsedArgs.mutcol)


if __name__ == '__main__':
    main()	

