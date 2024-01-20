#!/bin/bash

species="$1"
feature_path="$2"
n_jobs="$3"



readarray -t Anti_List <  ${feature_path}/${species}/anti_list

for anti in ${Anti_List[@]};do
    for j in {0..9};do
        ##########################################################################################
        #### Software-specific command here to train the 9 training folds and test on the test fold
        ##########################################################################################


    done
    wait
done
