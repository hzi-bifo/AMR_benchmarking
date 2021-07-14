# Purpose:
# - Compute phylognetic tree from mapping results of DNA-seq reads
# Materials:
# - DNA-seq reads
# - reference genome
# - adaptor file (optional)
# Methods:
# - Reads mapped using BWA-MEM
# - Variant sites called with freebayes
# - Sequences computed with bcftools consensus
# - MSA calculated with MAFFT
# - Phylogenetic tree inferred using fasttree
# Output:
# - phylogenetic tree
# - mapping reults
# - vcf files

import os
import pandas as pd
# parse the list of reads
list_f=config['list_f']
dna_reads = {}
with open(list_f, 'r') as list_fh:
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

REF_FA=config['REF_FA']
REF_GFF=config['REF_GFF']
results_dir=config['results_dir']
seq_list=config['seq_list']
families_seq_dir=config['families_seq_dir']
aln_f=config['aln_f']
tree_f=config['tree_f']
adaptor_f= config['adaptor']
new_reads_dir= config['new_reads_dir']
mapping_results_dir= config['mapping_results_dir']
fam_stats_file= 'fam_stats.txt'

rule all:
    input:
        tree_f

rule tree:
    # infer the phylogenetic tree
    input:
        alignment = aln_f
    output:
        tree = tree_f
    threads: 12
    conda: 'fasttree_env.yml'
    shell:
        '''
        set +u
        export OMP_NUM_THREADS={threads}
        FastTreeMP -nt -gtr -gamma \
-log {output.tree}.log -out {output.tree} {input.alignment}
        set -u
        '''


rule process_aln:
    # prune the invariant sites (<2 alternative state or >90% gaps)
    input:
        out_aln = 'OneLarge.aln'
    output:
        out_var_aln = aln_f
    params:
        tmp_aln1 = 'OneLarge.gapReplaced.aln',
        tmp_aln2 = 'OneLarge.gapReplaced.var2.aln'
    conda: 'concatenate_seq_env.yml'
    shell:
        """
        cat {input} | sed '/^>/!s/[^ATCGatcg]/-/g' > {params.tmp_aln1}
        removeInvariant.py --in {params.tmp_aln1} --out {params.tmp_aln2} --cn 2
        trimal -gt 0.90 -in {params.tmp_aln2} -out {output.out_var_aln}
        """


rule concatenate:
    #' concatenate the family-wise alignments that pass the length filter
    input:
        fam_list = 'aln_to_concatenate'
    output:
        out_aln = 'OneLarge.aln'
    conda: 'concatenate_seq_env.yml'
    params:
        conc_script = 'concatenateAln.py'
    shell:
        '''
        {params.conc_script} --l {input.fam_list} --o {output.out_aln}
        '''


rule list_families:
    #' filter the gene families by the length diversity
    input:
        out_seq = dynamic(os.path.join(families_seq_dir, '{fam}.aln'))
    output:
        out_fam_list = 'aln_to_concatenate'
    params:
        fam_stats = fam_stats_file,
        indel_cutoff = 0.1
    run:
        from os.path import basename
        target_fam = []
        with open(params.fam_stats, 'r') as fam_fh:
            families= [l.strip().split('\t') for l in fam_fh.readlines()]
            target_fam= [fam[0] for fam in families 
                if (int(fam[2])-int(fam[1]))/int(fam[2]) <= params.indel_cutoff]
        with open(output.out_fam_list, 'w') as out_fh:
            for f in input.out_seq:
                fam= re.sub('\.aln$', '', basename(f))
                if fam in target_fam:
                    out_fh.write(f+'\n')


rule alignment:
    input:
        fam_fa = os.path.join(families_seq_dir, '{fam}.fa')
    output:
        fam_aln = os.path.join(families_seq_dir, '{fam}.aln')
    conda: 'mafft_7_310_env.yml'
    params: 
        mafft_params = '--nuc --maxiterate 0 --retree 2 --parttree'
    threads: 2
    shell:
        '''
        mafft --thread {threads} {params.mafft_params} {input.fam_fa} > {output.fam_aln}
        '''


rule sort:
    # calculate the length diverty of sequences for each family
    input:
        seq_list_f = seq_list
    output:
        out_seq = dynamic(os.path.join(families_seq_dir, '{fam}.fa'))
    conda: 'concatenate_seq_env.yml'    
    params:
        fam_stats = fam_stats_file,
        out_dir = families_seq_dir
    shell:
        '''
        geneStats.py --l {input.seq_list_f} --d {params.out_dir} \
--o {params.fam_stats}
        '''


rule list_cons_sequences:
    input:
        expand('{tmp_d}/{strain}/target_regions.fa', tmp_d= results_dir, strain=list(dna_reads.keys()))
    output:
        seq_list_f = seq_list
    params: 
        tmp_d = results_dir,
        strains = list(dna_reads.keys())
    shell:
        '''
        parallel 'echo {{}}"\t"{params.tmp_d}/{{}}/target_regions.fa' ::: {params.strains} \
> {output.seq_list_f}
        '''


rule consensus_seqs:
    # introduce the variant sites to the reference 
    input:
        ref_region_seqs = REF_GFF+'.gene_regions.fa',
        vcf_gz = '{tmp_d}/{strain}/bwa.vcf.gz',
        vcf_gz_index = '{tmp_d}/{strain}/bwa.vcf.gz.tbi'
    output:
        cons_fa = '{tmp_d}/{strain}/target_regions.fa'
    conda:'bcftools_1_6_env.yml'
    shell:
        '''
        bcftools consensus -f {input.ref_region_seqs} \
-o {output.cons_fa} {input.vcf_gz}
        '''

rule ref_regions:
    # determine the coding regions to infer the phylogeny with
    input:
        ref_gff = REF_GFF,
        ref_fa = REF_FA
    output:
        ref_regions = REF_GFF+".gene_regions"
    conda: 'py27.yml'
    shell:
        '''
        make_GeneRegions.py --g {input.ref_gff} --f gene > {output.ref_regions}
        '''


rule ref_regions_seq:
    input:
        ref_fa = REF_FA,
        ref_regions = REF_GFF+".gene_regions"
    output:
        ref_region_seqs = REF_GFF+".gene_regions.fa"
    conda: 'phylo_bwa_env.yml'
    shell:
        '''
        parallel -j 1 \'samtools faidx {input.ref_fa} {{}}\' \
 :::: {input.ref_regions} > {output.ref_region_seqs}
        '''


rule index_ref:
    input:
        REF_FA  
    output:
        REF_FA+".bwt",
        REF_FA+".fai"
    threads:1
    conda: 'phylo_bwa_env.yml'
    shell:
        """
        samtools faidx {input}
        bwa index {input}
        """


rule move_mapping_result:
    # allow phylo and snps workflow to reuse those done by each other workflow
    input:
        sorted_bam = "{}/{{strain}}/bwa.sorted.bam".format(results_dir),
        sorted_bam_index = "{}/{{strain}}/bwa.sorted.bam.bai".format(results_dir)
    output:
        moved_bam = '{}/{{strain}}.bam'.format(mapping_results_dir),
        moved_bam_index = '{}/{{strain}}.bam.bai'.format(mapping_results_dir)
    shell:
        '''
        mv {input.sorted_bam} {output.moved_bam}
        mv {input.sorted_bam_index} {output.moved_bam_index}
        '''


rule mapping:
    #' reads mapping using bwa-mem
    input:
        ref = REF_FA,
        ref_index = REF_FA+".bwt",
        FQ1 = os.path.join(new_reads_dir, '{strain}.cleaned.1.fq.gz'),
        FQ2 = os.path.join(new_reads_dir, '{strain}.cleaned.2.fq.gz')
    output:
        # because directly piping is unavialble with this version of compiled bamtools, 
        # sam file as an intermediate is needed
        sam = temp("{tmp_d}/{strain}/bwa.sam"),
        bam = temp("{tmp_d}/{strain}/bwa.bam"),
        sorted_bam = temp("{tmp_d}/{strain}/bwa.sorted.bam"),
        sorted_bam_index = temp("{tmp_d}/{strain}/bwa.sorted.bam.bai")
    threads: 1
    conda: 'phylo_bwa_env.yml'
    shell:
        """
        bwa mem -v 2 -M -R \'@RG\\tID:snps\\tSM:snps\' -t {threads} \
 {input.ref} {input.FQ1} {input.FQ2} > {output.sam}
        samtools view -b -S {output.sam} -o {output.bam} -@ {threads}
        bamtools sort -in {output.bam} -out {output.sorted_bam}
        samtools index {output.sorted_bam}
        """


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
        adaptor_f = adaptor_f,
        tmp_f1 = lambda wildcards: os.path.join(
            new_reads_dir, '{}.cleaned.1.fq'.format(wildcards.strain)),
        tmp_f2 = lambda wildcards: os.path.join(
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
            ln  -s {input.infile1} {output.f1}
            ln  -s {input.infile2} {output.f2}
        fi
        '''


rule call_var:
    # variant calling using freebayes
    input:
        ref = REF_FA,
        ref_index = REF_FA+".fai",
        moved_bam = '{}/{{strain}}.bam'.format(mapping_results_dir),
        moved_bam_index = '{}/{{strain}}.bam.bai'.format(
            mapping_results_dir)
    output:
        vcf = '{tmp_d}/{strain}/bwa.vcf',
        vcf_gz = '{tmp_d}/{strain}/bwa.vcf.gz',
        vcf_gz_index = '{tmp_d}/{strain}/bwa.vcf.gz.tbi'
    params:
        freebayes_params = '-p 1'
    threads: 16
    conda: 'freebayes_1_1_env.yml'
    shell:
        """
        freebayes-parallel <(fasta_generate_regions.py {input.ref_index} 100000) \
 {threads} -f {input.ref} {params.freebayes_params} {input.moved_bam} \
 > {output.vcf}
        bgzip -c {output.vcf} > {output.vcf_gz}
        tabix -p vcf {output.vcf_gz}
        """
