in_tabs= config['in_tab'].split(',')

rule all:
    input: 
        expand('{in_tab}_GROUPS', in_tab= in_tabs),
        expand('{in_tab}_NONRDNT', in_tab= in_tabs)
    
rule compress_feat_table:
    input: 
        F='{in_tab}'
    output: 
        GROUPS='{in_tab}_GROUPS',
        NONRDNT='{in_tab}_NONRDNT'
    conda: 'cmpr_env.yaml'
    script: 'featCompress.py'
