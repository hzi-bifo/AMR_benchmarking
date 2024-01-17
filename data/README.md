# Data preprocessing


- [2. ML-based methods evaluation](#evaluation1)
  - [2.1 Input](#Input1)
  - [2.2 Feature building](#feature)
  - [2.3 Nested cross-evaluation](#nCV)
  - [2.4 Run the workflow and outputs](#outputs)
- [3. Rule-based methods evaluation](#evaluation2)
  - [3.1 Input](#Input2)
  - [3.2 AMR reports generation](#report)
  - [3.3 Iterative evaluation](#iter)
  - [3.4 Run the workflow and outputs](#outputs)
    
## <a name="setup"></a>1. Setup
### 1.1 Dependencies
  -    Linux OS and `conda`. Miniconda2 4.8.4 was used by us
### 1.2 Installation
  - Create two conda environments
  - Install packages with conda  
  ```
  conda env create -n  amr_env  -f ./install/amr_env.yml python=3.7
  conda env create -n resfiner_env  -f ./install/res_env.yml  
  ```
## <a name="evaluation1"></a>2. ML-based methods evaluation
  
 Output: 
 
- Feature profiles
- Summary reports of metrics including F1-macro, precision-positive, recall-positive, F1-positive, precision-negative, recall-negative, F1-negative, accuracy, and clinical-oriented F1-negative and precision-negative, recall-negative.
  
