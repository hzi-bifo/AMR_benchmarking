# SPDX-FileCopyrightText: 2021 Tzu-Hao Kuo
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Purpose:
# - De novo procedures of assemblies, annotation and orthologous clustering
# Materials:
# - DNA-seq reads
# - adaptor file (optional)
# Methods:
# - De novo assemblies with SPAdes 
# - Annotation with Prokka
# - Orthologous clustering with Roary
# - Indels from MSA with msa2vcf
# Output:
# - de novo assemblies
# - binary GPA table
#
import os
import shutil
import pandas as pd
import logging
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
            dna_reads[d[0]]= d[1].split(',')

out_prokka_dir = config['out_prokka_dir']
out_roary_dir = config['out_roary_dir']
out_spades_dir = config['out_spades_dir']
extracted_proteins_dir = config['extracted_proteins_dir']
out_gpa_f = config['out_gpa_f']
out_indel_f = config['out_indel_f']
REF_GFF = config['REF_GFF']
ref_gbk= config['ref_gbk']
annot_tab = config['annot_tab']
awk_script_f = os.path.join(os.environ['TOOL_HOME'], 'lib', 'filter.awk')
adaptor_f = config['adaptor']
new_reads_dir = config['new_reads_dir']

# enable utility of precomputed assemblies
if 'assemblies' in config:
    if not os.path.isdir(out_spades_dir):
        os.makedirs(out_spades_dir)
    assem_list_f = config['assemblies']
    with LoadFile(assem_list_f) as assem_list_fh:
        for l in assem_list_fh.readlines():
            if re.match('#', l):
                continue
            d = l.strip().split('\t')
            strain = d[0]
            assem_f = d[1]
            assert os.path.isfile(assem_f), (
                'Assembly file {} not found'.format(assem_f))
            out_spades_strain_dir = os.path.join(out_spades_dir, strain)
            contigs_f = os.path.join(out_spades_strain_dir, 'contigs.fasta')
            if os.path.isfile(contigs_f):
                logging.info('{} exists, so {} will be ignored')
            else:
                logging.info('Use precomputed assembly file {}. '
                    'De novo assemly will be skipped for {}'.format(
                    assem_f, strain))
                if not os.path.isdir(out_spades_strain_dir):
                    os.makedirs(out_spades_strain_dir)
                shutil.copyfile(assem_f, contigs_f)


rule all:
    input:
        indel_bin_mat = out_indel_f,
        gpa_bin_mat = out_gpa_f,
        non_redundant = expand('{in_tab}_{info}',
            in_tab=[out_indel_f, out_gpa_f],
            info=['GROUPS', 'NONRDNT'])


rule remove_redundant_feat:
    input: 
        F = '{in_tab}'
    output: 
        GROUPS = '{in_tab}_GROUPS',
        NONRDNT = '{in_tab}_NONRDNT'
    conda: 'cmpr_env.yml'
    script: 'featCompress.py'


rule make_annotation:
    input:
        gpa_csv = os.path.join(out_roary_dir, 'gene_presence_absence.csv'),
        ref_gff = REF_GFF,
        anno_f = annot_tab
    output:
        tmp_annotation_map = '{roary_dir}/annotation_mapped.txt',
        tmp_gff = '{roary_dir}/tmp.gff'
    conda: 'indel_py37_env.yml'
    params:
        creator_script = 'mapping_tab_and_gff.py'
    shell:
        '''
        {params.creator_script} \
 -g {input.ref_gff} -t {input.anno_f} \
 -out_gff {output.tmp_gff} -out_annot {output.tmp_annotation_map}
        '''

rule abricate_dict:
    # Determine family names using the reference gff file
    input:
        tmp_annotation_map = '{roary_dir}/annotation_mapped.txt',
        roary_clustered_proteins = os.path.join(out_roary_dir,
            "clustered_proteins"),
        ref_gff = REF_GFF,
        anno_f = annot_tab
    output:
        # tmp_annotation_map = '{roary_dir}/annotation_mapped.txt',
        # tmp_gff = '{roary_dir}/tmp.gff',
        rename_dict = '{roary_dir}/roary_abricate.txt',
        tmp_refined_map = '{roary_dir}/refined_mapping.txt'
    params:
        refine_mapping_script = 'refine_mapping.py',
        match_cluster_script = 'match_clusters.py'
    conda:'indel_env.yml'
    shell:
        '''
        {params.refine_mapping_script} {input.tmp_annotation_map} \
> {output.tmp_refined_map}
        {params.match_cluster_script} {output.tmp_refined_map} \
{input.roary_clustered_proteins} > {output.rename_dict}
        '''


rule gpa_bin_mat:
    # Encoding the roary output into the matrix of binary states
    input:
        rename_dict_f = os.path.join(out_roary_dir, 'roary_abricate.txt'),
        gpa_csv = os.path.join(out_roary_dir, 'gene_presence_absence.csv')
    output:
        gpa_bin_mat = out_gpa_f
    params:
        strains = list(dna_reads.keys())
    run:
        import pandas as pd
        import textwrap

        strains = params['strains']
        gpa_f = input['gpa_csv']
        output_f = output['gpa_bin_mat']

        # new name
        name_dict = {}
        in_fh = open(input.rename_dict_f, 'r')
        for l in in_fh:
            d = l.strip().split('\t')
            name_dict[d[0]] = d[1]

        # read roary result and identify single-copy genes
        df = pd.read_csv(gpa_f, sep= ',',
                header=0, index_col=0, quotechar='"', low_memory=False)

        # filter and convert the states
        bin_df = df.loc[:, strains].applymap(lambda x: '0' if pd.isna(x) else '1')
        bin_df = bin_df.transpose() # strains in rows
        bin_df.rename(columns=name_dict, inplace=True)
        bin_df.to_csv(output_f, sep='\t', header=True, index=True,
            index_label= 'Isolate')


rule indel_select_core_genes:
    # Detect indels
    input:
        gpa_csv = os.path.join('{roary_dir}', 'gene_presence_absence.csv'),
        prot_tab = os.path.join('{roary_dir}', 'clustered_proteins')
    output:
        core_gene_list = '{roary_dir}/core_genes_50.txt'
    params:
        min_num_strains = 2,
        filter_awk_script = awk_script_f
    conda:'indel_env.yml'
    threads: 20
    shell:
        '''
        awk -v threshold="{params.min_num_strains}" \
-f {params.filter_awk_script} < {input.gpa_csv} \
| sed 's/\W/_/' \
> {output.core_gene_list} 
        '''


rule indel_align_families:
    #' Compute family-wise alignments
    input:
        ffn_files = expand(os.path.join(
            out_prokka_dir, '{strain}', '{strain}.ffn'),
            strain=list(dna_reads.keys())),
        core_gene_list = os.path.join(out_roary_dir,'core_genes_50.txt'),
        prot_tab = os.path.join(out_roary_dir, 'clustered_proteins')
    output:
        fam_aln_files = dynamic(os.path.join(
            extracted_proteins_dir, '{fam}.aln'))
    params:
        gene_cluster2multi_script = 'gene_clusters2multi_fasta.py',
        extracted_proteins_dir = extracted_proteins_dir,
        parallel_log= 'mafft.log'
    threads: 20
    conda: 'indel_cluster_env.yml'
    shell:
        '''
        core_genes={input.core_gene_list}
        #number of core genes
        core_genes_wc=$(wc -l $core_genes | cut -f1 -d" ")
        #extract fasta sequences for each gene family
        {params.gene_cluster2multi_script} \
{params.extracted_proteins_dir} \
<(head -n $core_genes_wc {input.prot_tab}) \
{input.ffn_files}
        # align
        cd {params.extracted_proteins_dir}
        parallel --joblog {params.parallel_log} -j {threads} \
'mafft {{}}.fasta > {{}}.aln' ::: `ls| grep 'fasta'| sed 's/\.fasta//'`
        '''


rule indel_msa2vcf:
    # Detect indels from each family-wise alignment
    input:
        indel_msa = os.path.join(
            extracted_proteins_dir, '{fam}.aln')
    output:
        indel_vcf = 'indels/{fam}.vcf'
    conda: 'indel_env.yml'
    shell:
        '''
        msa2vcf < {input.indel_msa} > {output.indel_vcf}
        '''
      

rule indel_vcf2bin:
    # Encode the indel vcf files into the matrix of binary states
    input:
        indel_vcf='indels/{fam}.vcf'
    output:
        indel_indels= 'indels/{fam}_indels.txt',
        indel_gff= 'indels/{fam}_indels.gff',
        indel_stats= 'indels/{fam}_indel_stats.txt'
    conda: 'indel_env.yml'
    params:
        vcf2indel_script= 'vcf2indel.py',
        prefix= lambda wildcards: 'indel/{}'.format(wildcards.fam) 
    shell:
        '''
        {params.vcf2indel_script} {input.indel_vcf} \
 {params.prefix} {output.indel_indels} {output.indel_gff} {output.indel_stats}
        '''

rule indel_integrate_indels:
    # Combine the data of all families
    input:
        indel_all_indels = dynamic('indels/{fam}_indels.txt'),
        core_gene_list = os.path.join(out_roary_dir,'core_genes_50.txt'),
        gpa_rtab = os.path.join(out_roary_dir, 'gene_presence_absence.Rtab'),
        annot = out_gpa_f,
        roary_abricate = os.path.join(out_roary_dir, 'roary_abricate.txt')
    output:
        indel_annot = out_indel_f,
        indel_annot_stats = out_indel_f+'.stats'
    params:
        generate_feature_script='generate_indel_features.py',
        w_dir= 'indels'
    conda: 'indel_env.yml'
    shell:
        '''
        cd {params.w_dir}
        # In 'clustered_proteins', roary never quotes gene names; 
        # in the .Rtab, space-included names are quoted
        {params.generate_feature_script} \
<(cut -f1 ../{input.gpa_rtab} | tail -n+2 | sed 's/"//g') \
../{input.annot} \
../{output.indel_annot} \
../{output.indel_annot}.stats \
../{input.roary_abricate}
        '''

rule roary:
    #' Run Roary to compute orthologous groups
    input:
        gff_files= expand(os.path.join(out_prokka_dir, '{strain}', '{strain}.gff'),
            strain= list(dna_reads.keys()))
    output:
        gpa_csv = '{roary_dir}/gene_presence_absence.csv',
        gpa_rtab = '{roary_dir}/gene_presence_absence.Rtab',
        prot_tab = '{roary_dir}/clustered_proteins'
    conda: 'perl5_22_env.yml'
    params:
        check_add_perl_env_script = 'install_perl_mods.sh',
        check_add_software_script = 'set_roary_env.sh',
        roary_bin = 'roary'
    threads: 20
    shell:
        '''
        set +u
        ROARY_HOME=$(dirname $(dirname $(which roary)))
        # required perl modules
        {params.check_add_perl_env_script}

        export PATH=$ROARY_HOME/build/fasttree:\
$ROARY_HOME/build/mcl-14-137/src/alien/oxygen/src:\
$ROARY_HOME/build/mcl-14-137/src/shmcl:\
$ROARY_HOME/build/ncbi-blast-2.4.0+/bin:\
$ROARY_HOME/build/prank-msa-master/src:\
$ROARY_HOME/build/cd-hit-v4.6.6-2016-0711:\
$ROARY_HOME/build/bedtools2/bin:\
$ROARY_HOME/build/parallel-20160722/src:$PATH
        export PERL5LIB=$ROARY_HOME/lib:\
$ROARY_HOME/build/bedtools2/lib:$PERL5LIB
        which perl
        echo $PERL5LIB
        echo $PERLLIB
        rm -r {wildcards.roary_dir}
        {params.roary_bin} -f {wildcards.roary_dir} \
-v {input.gff_files} -p {threads} -e -g 100000 -z
        set -u
        '''


rule create_gff:
    # Detect coding regions 
    input:
        contigs = os.path.join(out_spades_dir,'{strain}', 'contigs.fasta')
    output:
        os.path.join(out_prokka_dir, '{strain}', '{strain}.gff'),
        os.path.join(out_prokka_dir, '{strain}', '{strain}.ffn')
    threads: 20
    #conda: 'perl_for_prokka.yml'
    conda: 'prokka_env.yml'
    shell:
        '''
        set +u
        which prokka
        export PROKKA_BIN=$( which prokka )
        export PERL5LIB=$( dirname $PROKKA_BIN )/../perl5/:$PERL5LIB
        echo $PERL5LIB
        prokka --locustag {wildcards.strain} \
--prefix  {wildcards.strain} \
--force  --cpus {threads} --metagenome --compliant \
--outdir prokka/{wildcards.strain} {input.contigs}
        set -u
        '''


rule create_annot:
    # Collect the annotations from the reference for updating 
    # the labels of gene families
    input:
        ref_gbk = ref_gbk
    output:
        anno_f = annot_tab
    params:
        ref_name = 'reference',
        creator_script = 'create_anno.py'
    shell:
        '''
        {params.creator_script} -r {input.ref_gbk} -n {params.ref_name} -o {output.anno_f}
        '''


rule spades_create_assembly:
    # Compute de novo assemlies with sapdes
    # The computational resources is increased when 
    # the process is crashed and rerun
    input:
        READS = lambda wildcards: [
          os.path.join(new_reads_dir,'{}.cleaned.{}.fq.gz'.format(
                wildcards.strain, str(n))) for n in [1,2]]
    output: os.path.join(out_spades_dir,'{strain}', 'contigs.fasta')
    threads: 1
    resources:
        mem_mb = lambda wildcards, attempt: (2**attempt) * 2000
    params:
        spades_outdir = os.path.join(out_spades_dir, '{strain}'),
        SPADES_OPT = '--careful',
        SPADES_BIN = 'spades.py'
    conda: 'spades_3_10_env.yml'
    shell:
        '''
        spades.py \
{params.SPADES_OPT} \
--threads {threads} \
--memory $( expr {resources.mem_mb} / 1000 ) \
-o {params.spades_outdir} \
-1 {input.READS[0]} -2 {input.READS[1]}
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

