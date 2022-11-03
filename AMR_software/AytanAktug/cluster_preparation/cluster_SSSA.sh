#!/bin/bash


species="$1"
log_path="$2"

echo ${species}
echo "-----------"

readarray -t Anti_List <  ${log_path}log/software/AytanAktug/software_output/SSSA/${species}/anti_list
array_contains () {
    local seeking=$1; shift
    local in=1
    for element; do
        if [[ $element == "$seeking" ]]; then
            in=0
            break
        fi
    done
    return $in
}
arr=('amikacin' 'ethambutol' 'ethiomide' 'ethionamide' 'kanamycin' 'ofloxacin')




for anti in ${Anti_List[@]};do
  (
    echo ${anti}
    if [ ${species} == 'Neisseria_gonorrhoeae' ]
    then
#       echo ${species}
       h_value=0.98
    elif [ ${species} == 'Mycobacterium_tuberculosis' ]
    then
       if array_contains ${anti} "${arr[@]}"
       then
#         echo ${species}_${anti}
         h_value=0.99
       else
         h_value=0.98
        fi

    else
       h_value=0.9
    fi

  cat ${log_path}log/software/AytanAktug/software_output/SSSA/${species}/cluster/temp/${anti}_all_strains_assembly.txt | \
  kma_clustering -i -- -k 16 -Sparse - -ht ${h_value} -hq ${h_value} -NI \
  -o ${log_path}log/software/AytanAktug/software_output/SSSA/${species}/cluster/temp/clustered_90_${anti} &> \
  ${log_path}log/software/AytanAktug/software_output/SSSA/${species}/cluster/${anti}_clustered_90.txt
  )&

done
wait
