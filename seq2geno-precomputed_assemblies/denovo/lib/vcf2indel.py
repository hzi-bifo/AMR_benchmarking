#!/usr/bin/env python2
import vcf as VCF
import pandas as pd
def vcf2indel(vcf, gene_name, out, out_gff, out_stats):
    no_indels = 0
    uq_indels = 0
    inserts = 0
    dels = 0
    overall_hits = []
    with open(vcf) as vcf_open:
        #skip meta info
        for l in vcf_open:
            if not l.startswith("##"):
                break 
        #read header
        columns = l.strip("#").split("\t")
        _, _, _, ref, alt, _, _, _, _ = columns[:9]
        sample_names = pd.Series(["_".join(i.split(" ")[0].split("_")[:-1]) for i in columns[9:]])
        sample_has_indel = dict([(i, 1) for i in sample_names])
        #read and process variants
        for l in vcf_open:
            columns = pd.Series(l.strip().split("\t"))
            _, pos, _, ref, alt, _, _, _, _ = columns[:9]
            #insertion at beginning of gene
            hits_temp = {}
            samples = pd.Series([i.split("/")[0] for i in columns[9:]])
            #compare length of ref and alternative alleles to extract indels
            alts = alt.split(",")
            #begin counting alt alleles from 1  reference allele  = 0
            for i, alt in zip(range(1, len(alts) + 1), alts) :
                if not alt == "":
                    if abs(len(alt) - len(ref)) >= 8:
                        if ref == ".":
                            if len(sample_names.loc[samples == "."]) > sample_names.shape[0]/2:
                                hits = sample_names.loc[samples != "."]
                            else:
                                hits = sample_names.loc[samples == "."]
                        elif alt == ".":
                            if len(sample_names.loc[samples == "."]) > sample_names.shape[0]/2:
                                hits = sample_names.loc[samples != "."]
                            else:
                                hits = sample_names.loc[samples == "."]
                        #reference allele is "." but covered by at least one insertion and deletion
                        elif len(sample_names.loc[samples == "."]) > sample_names.shape[0]/2:
                            hits = sample_names.loc[(str(i) == samples) & (samples != ".")]
                        else:
                            print i
                            hits = sample_names.loc[(str(i) == samples) | (samples == ".")]
                        if len(hits) != 0:
                            uq_indels += 1
                        if len(alt) < len(ref):
                            inserts += 1
                        if len(alt) > len(ref):
                            dels += 1
                        for hit in hits:
                            if hit in hits_temp:
                                if len(alt) > len(hits_temp[hit][2]):
                                    hits_temp[hit] = (hit, ref, alt, pos)
                                    print hit, alt, "replace"
                            else:
                                hits_temp[hit] = (hit, ref, alt, pos)
                            sample_has_indel[hit] = 0 
            for hit, ref, alt, pos in hits_temp.values():
                no_indels += 1
                overall_hits.append((hit, ref, alt, pos))
    with open(out_gff, 'w') as og:
        og.write("\tsample\tref\talt\tpos\n") 
        for hit, ref, alt, pos in overall_hits:
            og.write("%s\t%s\t%s\t%s\n" % (hit, ref, alt, pos)) 
    with open(out_stats, 'w') as os:
        os.write("no_indels\tuq_indels\tinserts\tdels\n") 
        os.write("%s\t%s\t%s\t%s\t%s\n" % (gene_name, no_indels, uq_indels, inserts, dels)) 
    pd.Series(sample_has_indel, name = gene_name).to_csv(out, sep = "\t",
                                                         header = True,
                                                         index_label= 'Isolate')
    #print sample_has_indel

    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("parse vcf and detect indels per sample")
    parser.add_argument("vcf", help="vcf produced by msa2vcf")
    parser.add_argument("gene_name", help="name of target gene e.g. ampD")
    parser.add_argument("out", help="sample2indel")
    parser.add_argument("out_gff", help="bed file with indel position")
    parser.add_argument("out_stats", help="sample2indel")
    args = parser.parse_args()
    vcf2indel(**vars(args))
