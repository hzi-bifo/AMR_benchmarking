#!/usr/bin/env/python
import getopt, sys
import warnings
import numpy as np
import argparse

# if not sys.warnoptions:
#     warnings.simplefilter("ignore")




def extract_info(list,res_path,output):

	##this script aims to transfer resfinder data into a matrix


	# snps = np.genfromtxt(point_path, dtype = "str")
	# print(snps.shape)

	list_sample = np.genfromtxt(list, dtype='str')
	genes = np.genfromtxt(res_path, dtype= "str")
	print(genes.shape)


	uniq_genes = []
	for each in genes[:,1]:
		uniq_genes.append(each)

	uniq_genes = list(set(uniq_genes))#gene string,.e.g. aadA5
	uniq_genes.sort()

	file_w = open(output, "w")
	for each in list_sample:#sample+feature
		# for e in each:
		# 	file_w.write(e)
		# 	file_w.write("\t")# SNPs finished loading
		if each in genes[:,0]:#a specific sample in Gene sample list(samples without AMR gene not listed)
			# print(each[0])

			gene_index = [i for i, x in enumerate(genes[:,0]) if x == each]#genes[:,0]: sample list
			#curent sample related gene presence list, e.g. [1,2,4,45]
			# print(gene_index)
			acquired_genes = []
			coverage = []
			for g in gene_index:#for each present gene
				tem = []
				acquired_genes.append(uniq_genes.index(genes[g,1]))#genes[g,1]: the gene name
				# print(genes[g,1])
				# print(acquired_genes)
				coverage.append(genes[g, 2])

			#index w.r.t. uniq_genes.
			# acquired_genes = list(set(acquired_genes))#rm duplicates

			gap = 0
			acquired_genes_uniq = list(set(acquired_genes))
			for a in sorted(acquired_genes_uniq):
				b = a - gap
				for i in range(0, b):
					file_w.write("-1")
					file_w.write("\t")
				ind = acquired_genes.index(a)
				file_w.write(str(coverage[ind]))
				file_w.write("\t")
				gap = a + 1
			if sorted(acquired_genes)[-1] != len(uniq_genes) - 1:#e.g. [33,45]. the last index,
				for m in range(sorted(acquired_genes)[-1] + 1, len(uniq_genes)):
					file_w.write("-1")
					file_w.write("\t")
		else:
			for i in range(0, len(uniq_genes)):
				file_w.write("-1")
				file_w.write("\t")
		file_w.write("\n")
	file_w.close()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-l","--list", default=None, type=str, required=True,
						help='path to sample list( metadata)')
	# parser.add_argument("-p","--pointfin", default=None, type=str, required=True,
						# help='PointFinder results.')
	parser.add_argument("-r","--resfin", default=None, type=str, required=True,
						help='ResFinder results.')
	parser.add_argument("-o","--output", default=None, type=str, required=True,
						help='Output file names')

	parsedArgs = parser.parse_args()
	# parser.print_help()
	# print(parsedArgs)
	extract_info(parsedArgs.list,parsedArgs.resfin,parsedArgs.output)

if __name__ == '__main__':
    main()