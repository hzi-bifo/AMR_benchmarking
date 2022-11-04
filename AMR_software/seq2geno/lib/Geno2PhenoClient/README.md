# GenoPheno Data Validation

GenoPheno part of the Seq2Geno2Pheno is supposed to perform the machine learning predictive modeling and biomarker extraction
and generate the publication quality figures after following the seq2geno part. However, this part is not limited to the features created by Seq2Geno and
you have already created your data features you may run your data on our server application. However, in order to save your time and avoid incorrect submissions to the server you can verify
your data and config prior to the submission using a validator code and modify your config.

After preparion of the configuration and successfull running of the validator you can run your experiment on the server:
The server is available at: https://galaxy.bifo.helmholtz-hzi.de/galaxy


## Run the validation and zip generation


In order to create the environment in anaconda for the config/data verification create an anaconda environment:

```
conda env create -n YOUR_ENV_NAME -f environment.yml
conda activate YOUR_ENV_NAME
```

To check the config file and create the zip file for the GenoPheno server run the following command:

```
python validator.py --config data/config.yml --log log.txt --zip myproj.zip
```

where:
`data/config.yml` is the path to the config file in which every path is explicit.
`log.txt` is the path to the log file generated in checking. Please read this file carefully after generation to ensure the project is created according to your wishes.
`myproj.zip` is the generated zip file which you can submit to the GenoPheno server.


## Config modification

The config of GenoPheno has three main parts: (1) metadata, (2) genotype tables, (3) prediction block

### metadata

In this part the metadata of the project are provided, which are the followings:

```
  `project`: MyProject            --------> <string> the project name
  `phylogenetic_tree`: tree.txt   --------> <string> the path to the phylogenetic tree
  `phenotype_table`: phentype.csv --------> <string> the path to the phenotype table in tabular format,please see the example files)

  NOTE:
  ==================================================================
  Please note that the phenotype table can contain multiple schemes for the phenotype.
  The first column is the name of instances, the headers are the phenotypes, and the labels can be any string

  To learn more please see the example:
  validation/data/phenotypes.txt
  ==================================================================


  `output_directory`: `results/` --------> <string> please do not change this path.
  `number_of_cores`: 20          --------> <int> the integer number of cores (<20)
```

### genotype tables

In this part, the genotype tables are provided, including the tabular features and the k-mers needed to be extracted from the sequence contigs.

```
z:
  tables:
    - table:
          name: 'genexp'            --------> define a name for this feature
          path: "data/genexp.csv"   --------> tabular file of features where the first column is the instance IDs
          preprocessing: none       --------> preprocessing on the matrix of features, possible options are ['l1','l2','percent','zero2one','none','std','binary']
          delimiter: ","            --------> delimiter for parsing the table
          datatype: numerical       --------> the data type, possible options are: ['numerical','text']
    ..
    ..
    ..
    - sequence:                   -------->  the second type is sequence features
          name: '6mer'            -------->  a name for the feature
          path: "data/sequences/" -------->  a directory where each file follows the pattern of instanceID.fasata
          preprocessing: l1       --------> preprocessing on the matrix of features, possible options are ['l1','l2','percent','zero2one','none','std','binary']
          k_value: 6              --------> k of k-mer feature (1<k<9)
```


### prediction block

This part contains information needed for the machine learning models and validation

```
predictions:
  - prediction:              -------->  we would have a list of prediction block (maybe a user wants to perform differnt label mappings)
        name: "amr_pred"     -------->  a name for this prediction scheme
        label_mapping:       -------->  since the software currently support only the binary classification, in case of having mutliple phentypes all needs to be mapped to either 0 or 1 and the rest would be ignored
          1: 1
          0: 0
        optimized_for: "f1_macro"                     -------->  the score that the model on the validation data is optimized for. The possiblities are ['accuracy',  'auc_score_macro',  'auc_score_micro',  'f1_macro',  'f1_micro',  'f1_neg',  'f1_pos',  'p_neg',  'p_pos',  'precision_macro',  'precision_micro',  'r_neg',  'r_pos',  'recall_macro',  'recall_micro']
        reporting: ['accuracy', 'f1_pos', 'f1_macro'] -------->  in the final report multiple scores can be reported, again from the list of ['accuracy',  'auc_score_macro',  'auc_score_micro',  'f1_macro',  'f1_micro',  'f1_neg',  'f1_pos',  'p_neg',  'p_pos',  'precision_macro',  'precision_micro',  'r_neg',  'r_pos',  'recall_macro',  'recall_micro']
        features:                                     -------->  users can specify different feature sets by combining the names used in `genotype_tables` block.
          - feature: "GenExpKmer"
            list: ['genexp','6mer'] -------->  combining features
            validation_tuning:      -------->  `validation_tuning` is a block to define the cross-validation and test scheme
              name: "cv_tree"       -------->  a name for the cv so that it can be reused
              train:                -------->  the cv training scheme
                method: "treebased" -------->  the cv folds can be either selected based on tree or random, i.e., ['treebased','random']
                folds: 10           -------->  # of folds
              test:                 -------->  the test structure
                method: "treebased" -------->  the selection method
                ratio: 0.1          -------->  the ration of data for testing in the beginning
              inner_cv: 10          -------->  number of folds for the nested CV
          - feature: "K-mer"
            list: ["6mer"]
            use_validation_tuning: "cv_tree" --------> a defined CV can be used with recalling
          - feature: "GPA"
            list: ["gpa"]
            use_validation_tuning: "cv_tree"
        classifiers:   --------> possible classifiers
          - lsvm       --------> linear SVM
          - svm        --------> SVM
          - lr         --------> Logistic regression
          - rf         --------> Random forests
```

# Run on the server

The steps for submitting your experiments are detailed at the BIFO galaxy: https://galaxy.bifo.helmholtz-hzi.de/galaxy
You first need to register on the server to be able to track your experiment properly.

## Upload the zip file
Upload the data under the "Get Data" menu.
<img width="1434" alt="Screen Shot 2020-12-10 at 8 53 26 AM" src="https://user-images.githubusercontent.com/8551117/101737849-b71c8500-3ac5-11eb-9eb7-8c6715534e8a.png">
## Run the experiment
Run the experiment under the Geno2Pheno.
<img width="1434" alt="Screen Shot 2020-12-10 at 8 53 46 AM" src="https://user-images.githubusercontent.com/8551117/101737869-bc79cf80-3ac5-11eb-8941-fc2191286782.png">


