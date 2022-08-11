The sample names in each folds can be loaded by following commands
```
import pickle

species='Escherichia coli'
anti='amoxicillin'
'''
Can change to any of the following species and corresponding antibiotics:
'Escherichia coli' 'Staphylococcus aureus' 'Salmonella enterica' 'Enterococcus faecium' 'Campylobacter jejuni' 'Neisseria gonorrhoeae' 'Klebsiella pneumoniae' \
'Pseudomonas aeruginosa' 'Acinetobacter baumannii' 'Streptococcus pneumoniae' 'Mycobacterium tuberculosis' 
'''

path=str(species.replace(" ", "_"))+"/"+ \
              str(anti.translate(str.maketrans({'/': '_', ' ': '_'})))+"_random_cv.pickle"
names = pickle.load(open(path, "rb"))
print(names)
```
