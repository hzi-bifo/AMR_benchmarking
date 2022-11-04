# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Purpose:
# - Quantify the expression levels and generate the features matrix
# Materials:
# - RNA-seq reads
# - reference genome
# - adaptor file (optional)
# Methods:
# - Reads mapped using BWA backtrack
# Output:
# - numeric expression table
#
import pandas as pd
import sys
def LoadFile(f):
    # The files might have different encoding methods
    # To void the problem, this is a generalized loader to create file handels
    encoding_set = ['utf-8', 'latin1', 'windows-1252']
    right_encoding = encoding_set.pop()
    fh = open(f, 'r', encoding=right_encoding)
    found = False
    while (not found) and (len(encoding_set) > 0):
        try:
            # test whether the file can be read
            fh.readlines()
            found = True
        except UnicodeDecodeError:
            # shift the decoding to the next
            right_encoding = encoding_set.pop()
        finally:
            # open the file with either the same or another decoding method
            fh.close()
            fh = open(f, 'r', encoding=right_encoding)

    if found:
        return(fh)
    else:
        raise UnicodeDecodeError(
            'The encodings of {} is not recognizable'.format(f))

list_f = config['list_f']
rna_reads = {}
with LoadFile(list_f) as list_fh:
    for l in list_fh.readlines():
        if re.match('#', l):
            continue
        d = l.strip().split('\t')
        try:
            assert len(d) == 2
        except AssertionError:
            raise SyntaxError(
                '{} has an incorrectly formatted line:\n{}'.format(
                list_f, l.strip()))
        else:
            rna_reads[d[0]] = d[1]

strains = list(rna_reads.keys())
out_table = config['out_table']
out_log_table = config['out_log_table']
ref_fasta = config['ref_fasta']
ref_gbk = config['ref_gbk']
annot_tab = config['annot_tab']
r_annot = config['r_annot']
arx_file = 'rna_mapping.tar.gz'


rule:
    input:
        arx_file,
        out_table,
        out_log_table


rule archive_data:
    # archive the intermediate data 
    input:
        check_file = out_table,
        target_files = expand('{strain}.{types}', strain=strains,
            types=['sam', 'sai','sin', 'rpg'])
    output:
        arx = arx_file
    shell:
        '''
        tar -czvf {output.arx} {input.target_files} --remove-files
        '''


rule collect_rpg:
    # integrate the per-region coverages and form the matrix
    input:
        rpg_files = expand('{strain}.rpg', strain= strains)
    output:
        rpg_tab = out_table,
        log_rpg_tab = out_log_table
    wildcard_constraints:
        strain='^[^\/]+$'
    run:
        import pandas as pd
        import re
        series_to_collect = []
        for f in input.rpg_files:
            # import all files
            s = re.search('([^\/]+).rpg', f).group(1)
            df = pd.read_csv(f, sep='\t', header= None)
            col_dict = {0: 'locus', 1: s, 2: 'antisense_count'}
            df = df.rename(columns=col_dict).set_index('locus')
            series_to_collect.append(df.loc[:, s])
        rpg_df = pd.DataFrame(series_to_collect)
        rpg_df.to_csv(output[0], sep='\t', index_label='Isolate')

        log_rpg_df = pd.np.log(rpg_df+1)
        log_rpg_df.to_csv(out_log_table, sep="\t", index_label='Isolate')


rule bwa_pipeline:
    # reads mapped to the reference genome with BWA backtrack algorithm
    # count the coverages for each coding region
    input:
        infile = lambda wildcards: rna_reads[wildcards.strain],
        reffile = ref_fasta,
        ref_index_bwa = ref_fasta+'.bwt',
        annofile = annot_tab,
        Rannofile = r_annot
    output:
        sam = temp('{strain}.sam'),
        sai = '{strain}.sai',
        sin = '{strain}.sin'
    threads: 1
    conda: 'snps_tab_mapping.yml'
    shell:
        """
        bwa aln {input.reffile} {input.infile} > {output.sai}
        bwa samse -r $( echo "@RG\\tID:snps\\tSM:snps" ) \
 {input.reffile} {output.sai} {input.infile} > {output.sam}
        sam2art.py -s 2 -l --sam {output.sam} > {output.sin}
        """


rule calc_genecounts:
    # calculate the counts per gene
    input:
        sin = '{strain}.sin',
        annofile = annot_tab
    output:
        rpg = '{strain}.rpg'
    conda: 'snps_tab_mapping.yml'
    shell:
        '''
        art2genecount.py --art {input.sin} \
 -t tab -r {input.annofile} \
 > {output.rpg} 
        '''


rule create_annot:
    # determine the coding regions to quantify the expression levels
    input:
        ref_gbk = ref_gbk
    output:
        anno_f = annot_tab
    params:
        ref_name = 'reference'
    shell:
        '''
        create_anno.py -r {input.ref_gbk} -n {params.ref_name} -o {output.anno_f}
        '''


rule create_r_annot:
    # determine the coding regions to quantify the expression levels
    input:
        ref_gbk = ref_gbk
    output:
        R_anno_f = r_annot
    shell:
        '''
        create_R_anno.py -r {input.ref_gbk} -o {output.R_anno_f}
        '''


rule index_ref:
    input:
        reffile = ref_fasta
    output:
        ref_fasta+'.bwt',
        ref_fasta+'.fai'
    conda: 'snps_tab_mapping.yml'
    shell:
        '''
        bwa index -a bwtsw {input.reffile}
        samtools faidx {input}
        '''
