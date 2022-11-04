# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Purpose:
# - Calculate SNPs matrix from mapping results of DNA-seq reads
# Materials:
# - DNA-seq reads
# - reference genome
# - adaptor file (optional)
# Methods:
# - Reads mapped using BWA-MEM
# - Variant sites called with samtools mpileup
# Output:
# - binary SNPs table
# - SNPs effects (i.e. synnymous and non-synonymous)
# - vcf files

import pandas as pd
from snakemake.utils import validate
import re
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

# parse the list of reads
list_f = config['list_f']
dna_reads = {}
with LoadFile(list_f) as list_fh:
    for l in list_fh.readlines():
        if re.match('#', l):
            continue
        d = l.strip().split('\t')
        try:
            assert ((len(d) == 2) and (len(d[1].split(',')) == 2))
        except AssertionError:
            raise SyntaxError(
                '{} has an incorrectly formatted line:\n{}'.format(
                list_f, l.strip()))
        else:
            dna_reads[d[0]] = d[1].split(',')

strains = list(dna_reads.keys())
ref_fasta = config['ref_fasta']
ref_gbk = config['ref_gbk']
annot_tab = config['annot_tab']
r_annot = config['r_annot']
mapping_results_dir = config['mapping_results_dir']
snps_table = config['snps_table']
snps_aa_table = config['snps_aa_table']
nonsyn_snps_aa_table = config['nonsyn_snps_aa_table']
snps_aa_bin_mat = config['snps_aa_bin_mat']
nonsyn_snps_aa_bin_mat = config['nonsyn_snps_aa_bin_mat']
adaptor_f = config['adaptor']
new_reads_dir = config['new_reads_dir']
table_subset_num = 30
arx_file = 'dna_var_calling.tar.gz'


rule all:
    input:
        # enabling reuse for the phylo workflow
        expand('{}/{{strain}}.{{file_ext}}'.format(mapping_results_dir),
               strain=strains,
               file_ext=['bam', 'bam.bai']),
        arx_file,
        snps_aa_bin_mat,
        nonsyn_snps_aa_bin_mat,
        expand('{in_tab}_{info}',
               in_tab=[snps_aa_bin_mat, nonsyn_snps_aa_bin_mat],
               info=['GROUPS', 'NONRDNT'])


rule archive_data:
    # archive the intermediate data 
    input:
        snps_aa_bin_mat = snps_aa_bin_mat,
        nonsyn_snps_aa_bin_mat = nonsyn_snps_aa_bin_mat,
        target_files = expand('{strain}.{types}',
                        strain=strains,
                        types=['flatcount','flt.vcf', 'raw.bcf'])
    output:
        arx = arx_file
    shell:
        '''
        tar -czvf {output.arx}  {input.target_files} --remove-files
        '''


rule remove_redundant_feat:
    # features grouped by patterns
    input:
        F = '{in_tab}'
    output:
        GROUPS = '{in_tab}_GROUPS',
        NONRDNT = '{in_tab}_NONRDNT'
    conda: 'cmpr_env.yaml'
    script: 'featCompress.py'


rule create_binary_table:
    # create the binary matrix for the variant sites
    input:
        snps_aa_table = snps_aa_table,
        nonsyn_snps_aa_table = nonsyn_snps_aa_table
    output:
        snps_aa_bin_mat = snps_aa_bin_mat,
        nonsyn_snps_aa_bin_mat = nonsyn_snps_aa_bin_mat
    conda: 'py27.yml'
    params:
        parse_snps_tool = 'parse_snps.py'
    threads: 1
    shell:
        '''
        {params.parse_snps_tool} {input.snps_aa_table} \
{output.snps_aa_bin_mat}
        {params.parse_snps_tool} {input.nonsyn_snps_aa_table} \
{output.nonsyn_snps_aa_bin_mat}
        '''


rule create_table:
    # print the lsit of all variant sites 
    input:
        flt_vcf = expand('{strain}.flt.vcf', strain= strains),
        flatcount = expand('{strain}.flatcount', strain= strains),
        dict_file = 'dict.txt',
        ref_gbk = ref_gbk,
        annofile = annot_tab
    output:
        snps_table = snps_table
    conda: 'snps_tab_mapping.yml'
    params:
        split_tool = 'split_for_mutation_table.py',
        isol_subset_num = lambda wildcards: min(table_subset_num,len(strains)),
        isol_subset_top = lambda wildcards: min(table_subset_num,len(strains)) - 1,
        isol_subset_dir = 'isols_subset',
        # mut_tab_tool= 'mutation_table.py',
        mut_tab_tool = 'mutation_table_py3.py',
        snps_subset_dir = 'snps_subset'
    threads: 16 
    shell:
        '''
        if [ ! -d {params.isol_subset_dir} ]; then
          mkdir -p {params.isol_subset_dir};
        fi
        if [ ! -d {params.snps_subset_dir} ]; then
          mkdir -p {params.snps_subset_dir};
        fi
        {params.split_tool} {input.dict_file} \
 {params.isol_subset_num} \
 {params.isol_subset_dir}

        for i in {{0..{params.isol_subset_top}}}; \
        do \
          echo "{params.mut_tab_tool} -f <(cat {input.dict_file} ) \
           -a {input.annofile} \
           -o {params.snps_subset_dir}/all_snps_$i.txt \
           --restrict_samples {params.isol_subset_dir}/isols_${{i}}.txt\
           --force_homozygous"; \
        done \
        | parallel -j {threads} --joblog {params.snps_subset_dir}/joblog_mutation_table.txt

        i=$(for i in {{0..{params.isol_subset_top}}}; \
 do 
 if [ -f {params.snps_subset_dir}/all_snps_${{i}}.txt ]; then
 echo -n " <(cut -f5-  {params.snps_subset_dir}/all_snps_${{i}}.txt)"; fi; done)

        echo "paste <(cut -f1-4  {params.snps_subset_dir}/all_snps_0.txt )  $i\
> {output.snps_table}" | bash
        '''


rule include_aa_into_table:
    # predict SNPs effect by translating the codons
    input:
        ref_gbk = ref_gbk,
        snps_table = snps_table
    output:
        snps_aa_table = snps_aa_table,
        nonsyn_snps_aa_table = nonsyn_snps_aa_table
    params:
        to_aa_tool = 'Snp2Amino_py3.py'
    threads: 1
    shell:
        """
        {params.to_aa_tool} -f {input.snps_table} -g {input.ref_gbk} \
-n all -o {output.snps_aa_table}
        awk '{{if($6 != "none" && $5 != $6){{print $0}}}}' {output.snps_aa_table} > \
{output.nonsyn_snps_aa_table}
        """


rule isolate_dict:
    # ensure no empty file
    input:
        flt_vcf = expand('{strain}.flt.vcf', strain=strains),
        flatcount = expand('{strain}.flatcount', strain=strains)
    output:
        dict_file = 'dict.txt'
    threads:1
    wildcard_constraints:
        strain = '^[^\/]+$'
    params:
        strains = strains
    run:
        import re
        import os
        # list and check all required files
        try:
            empty_files= [f for f in input if os.path.getsize(f)==0]
            if len(empty_files) > 0:
                raise FileNotFoundError(
                    '{} should not be empty'.format(
                    ','.join(empty_files)))
        except FileNotFoundError as e:
            sys.exit(str(e))

        with open(output[0], 'w') as out_fh:
            out_fh.write('\n'.join(params.strains))


rule move_mapping_result:
    # allow phylo and snps workflow to reuse those done by each other workflow
    input:
        bam = '{strain}.bam',
        bam_index = '{strain}.bam.bai'
    output:
        moved_bam = '{}/{{strain}}.bam'.format(mapping_results_dir),
        moved_bam_index = '{}/{{strain}}.bam.bai'.format(mapping_results_dir)
    shell:
        '''
        ln -s $( realpath {input.bam} ) {output.moved_bam}
        ln -s $( realpath {input.bam_index} ) {output.moved_bam_index}
        '''


rule call_var:
    # variant calling with samtools mpileup
    input:
        bam = '{strain}.bam',
        bam_index = '{strain}.bam.bai',
        reffile = ref_fasta
    output:
        raw_bcf = '{strain}.raw.bcf',
        flt_vcf = '{strain}.flt.vcf'
    threads:1
    params:
        min_depth = 0
    conda: 'snps_tab_mapping.yml'
    shell:
        """
        samtools mpileup -uf {input.reffile} {input.bam} | \
bcftools view -bvcg - > {output.raw_bcf}
        bcftools view {output.raw_bcf} | \
vcfutils.pl varFilter -d {params.min_depth} > {output.flt_vcf}
        """


rule samtools_SNP_pipeline:
    # mapping the reads 
    input:
        sam = '{strain}.sam',
        reffile = ref_fasta
    output:
        bam = '{strain}.bam',
        bam_index = '{strain}.bam.bai'
    threads:1
    conda: 'snps_tab_mapping.yml'
    shell:
        """
        samtools view -bS {input.sam} > {output.bam}
        samtools sort {output.bam} {wildcards.strain}
        samtools index {output.bam}
        """


rule bwa_pipeline_PE:
    # reads mapped to the reference genome with BWA-MEM
    input:
        infile1 = lambda wildcards: os.path.join(
          new_reads_dir, '{}.cleaned.1.fq.gz'.format(wildcards.strain)),
        infile2 = lambda wildcards: os.path.join(
          new_reads_dir, '{}.cleaned.2.fq.gz'.format(wildcards.strain)),
        reffile = ref_fasta,
        ref_index_bwa = ref_fasta+'.bwt',
        annofile = annot_tab,
        Rannofile = r_annot
    output:
        sam = temp('{strain}.sam'),
        flatcount = '{strain}.flatcount'
    params:
        sam2art_bin = 'sam2art.py'
    threads: 1
    conda: 'snps_tab_mapping.yml'
    shell:
        '''
        bwa mem -v 2 -M \
-t 1 -R $( echo "@RG\\tID:snps\\tSM:snps" ) \
{input.reffile} {input.infile1} {input.infile2} > {output.sam}

        {params.sam2art_bin} -f -s 2 -p --sam {output.sam} > {output.flatcount}
        '''


rule redirect_and_preprocess_reads:
    # reads processing before mapped to the reference 
    input:
        infile1 = lambda wildcards: dna_reads[wildcards.strain][0],
        infile2 = lambda wildcards: dna_reads[wildcards.strain][1]
    output:
        log_f = os.path.join(new_reads_dir, '{strain}.log'),
        f1 = os.path.join(new_reads_dir, '{strain}.cleaned.1.fq.gz'),
        f2 = os.path.join(new_reads_dir, '{strain}.cleaned.2.fq.gz')
    params:
        adaptor_f= adaptor_f,
        tmp_f1= lambda wildcards: os.path.join(
            new_reads_dir, '{}.cleaned.1.fq'.format(wildcards.strain)),
        tmp_f2= lambda wildcards: os.path.join(
            new_reads_dir, '{}.cleaned.2.fq'.format(wildcards.strain))
    conda: 'eautils_env.yml'
    shell:
        '''
        if [ -e "{params.adaptor_f}" ]
        then
            fastq-mcf -l 50 -q 20 {params.adaptor_f} {input.infile1} {input.infile2} \
  -o {params.tmp_f1} -o {params.tmp_f2} > {output.log_f}
            gzip -9 {params.tmp_f1}
            gzip -9 {params.tmp_f2}
        else
            echo 'Reads not trimmed'
            echo 'No trimming' > {output.log_f}
            echo $(readlink {input.infile1}) >> {output.log_f}
            echo $(readlink {input.infile2}) >> {output.log_f}
            cp {input.infile1} {output.f1}
            cp {input.infile2} {output.f2}
        fi
        '''


rule create_annot:
    input:
        ref_gbk = ref_gbk
    output:
        anno_f = annot_tab
    params:
        creator_script = 'create_anno.py',
        ref_name = 'reference'
    shell:
        '''
        {params.creator_script} -r {input.ref_gbk} -n {params.ref_name} -o {output.anno_f}
        '''


rule create_r_annot:
    input:
        ref_gbk=ref_gbk
    output:
        R_anno_f=r_annot
    params:
        creator_script = 'create_R_anno.py'
    shell:
        '''
        {params.creator_script} -r {input.ref_gbk} -o {output.R_anno_f}
        '''


rule index_ref:
    input:
        reffile=ref_fasta
    output:
        ref_fasta+'.bwt',
        ref_fasta+'.fai'
    conda: 'snps_tab_mapping.yml'
    shell:
        '''
        bwa index -a bwtsw {input.reffile}
        samtools faidx {input}
        '''
