#!/usr/bin/env/python
import getopt, sys
import warnings
import numpy as np
import argparse
import pandas as pd
# if not sys.warnoptions:
#     warnings.simplefilter("ignore")


def extract_info(id_list,feature,pheno,output):
	##the script output is the same ordered input and output files without sample IDS, and sample names
	##will be produced as a separate file. 
	#todo, add id_list
	id_list=np.genfromtxt(id_list, dtype="str")

	# data_x = np.genfromtxt(feature, dtype="str")

	data_y = np.genfromtxt(pheno, dtype="str")
	print('data_y.shape',data_y.shape)
	data_y_list = []


	columns = len(data_y[0])

	for each in data_y:
		data_y_list.append(each[0])


	#force the data_x according to id_list
	feature = np.genfromtxt(feature, dtype="str")
	print('feature shape', feature.shape)
	n_feature = feature.shape[1] - 1
	df_feature = pd.DataFrame(feature[:, 1:], index=feature[:, 0],
							  columns=np.array(np.arange(n_feature), dtype='object'))
	df_feature=df_feature.reindex(id_list)
	df_feature=df_feature.reset_index()
	# print(df_feature)
	data_x=df_feature.to_numpy()


	data_x_new = open(output+"data_x.txt", "w")
	data_y_new = open(output+"data_y.txt", "w")
	sample_list = []
	all_names = open(output+"data_names.txt", "w")


	for each in data_x[0:]:
		# print(each)
		if each[0] in data_y_list:
			ind = data_y_list.index(each[0])
			sample_list.append(each[0])
			all_names.write(str(each[0]))
			all_names.write("\n")
			for i in each[1:]:
				data_x_new.write(i)
				data_x_new.write("\t")
			data_x_new.write("\n")
			for i in range(1,columns):
				data_y_new.write(data_y[ind,i])
				if i != columns-1:
					data_y_new.write("\t")
				elif i == columns-1:
					data_y_new.write("\n")


	data_x_new.close()
	data_y_new.close()
	all_names.close()

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-id", "--id_list", default=None, type=str, required=True,
						help='Used as the criteria for sample order in 3 final files for model.')
	parser.add_argument("-f","--feature", default=None, type=str, required=True,
						help='File should include sample ID and features per sample')
	parser.add_argument("-p","--phenotype", default=None, type=str, required=True,
						help='File should include sample ID and y(0,1) per sample.')
	parser.add_argument("-o","--output", default=None, type=str, required=True,
						help='Output file names')

	parsedArgs = parser.parse_args()
	# parser.print_help()
	# print(parsedArgs)
	extract_info(parsedArgs.id_list,parsedArgs.feature,parsedArgs.phenotype,parsedArgs.output)

if __name__ == '__main__':
    main()