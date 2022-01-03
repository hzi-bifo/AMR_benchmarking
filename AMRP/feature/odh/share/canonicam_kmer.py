import itertools
from Bio.Seq import Seq


def canonical_kmer(k=3):
    vocab = 'ACGT'

    kmers = (''.join(nucleotides)
             for nucleotides in itertools.product(vocab, repeat=k))

    canonical_kmer = set()

    canonical_kmer_map = {}

    for kmer in kmers:
        rc_kmer = str(Seq(kmer).reverse_complement())
        if kmer <= rc_kmer:
            canonical_kmer.add(kmer)
            canonical_kmer_map[kmer] = kmer
        else:
            canonical_kmer.add(rc_kmer)
            canonical_kmer_map[kmer] = rc_kmer

    return sorted(canonical_kmer), canonical_kmer_map


# kmer_set, kmer_map = canonical_kmer(k=3)

# print(len(kmer_set))
# print(len(kmer_map))
# for kmer, canonical_kmer in kmer_map.items():
#     print(kmer, canonical_kmer)
