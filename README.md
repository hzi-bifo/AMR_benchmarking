# AMR tools Benchmarking

- [Patric data set](#data)
- [ResFinder](#p)
- [Multi-species](#m)

### <a name="data"></a>Patric data set
1. Quality control criteria: strict
    - genome status : not Plasmid
    - genome_quality : Good
    - contigs <= 100
    - fine_consistency>= 97
    - coarse_consistency >= 98
    - checkm_completeness >= 98
    - checkm_contamination <= 2
    - |genome length - mean length| <= mean_length/20

Resulting species and antibiotics:
![plot](./Patric data set/species_strict.png)




2. Quality control criteria: loose
![plot](./Patric data set/species_loose.png)


### <a name="p"></a>ResFinder

Bortolaia, Valeria, et al. "ResFinder 4.0 for predictions of phenotypes from genotypes." Journal of Antimicrobial Chemotherapy 75.12 (2020): 3491-3500.

https://bitbucket.org/genomicepidemiology/resfinder/src/master/



### <a name="m"></a>Multi-species
Nguyen, Marcus, et al. "Predicting antimicrobial resistance using conserved genes." PLoS computational biology 16.10 (2020): e1008319.

https://bitbucket.org/deaytan/data_preparation
https://bitbucket.org/deaytan/neural_networks/src/master/




