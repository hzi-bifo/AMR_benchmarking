#!/usr/bin/python
import sys
import os
sys.path.insert(0, os.path.abspath('../'))
import itertools

def vocab_build(canonical,k):
    if canonical == True:
        vocab = [''.join(xs) for xs in itertools.product('ACGT', repeat=k)]
        # print(len(vocab))
        cano_v = []
        for i in vocab:  # gg. i='AA'
            # create another list of canonical pair, in the same position.
            j = ''
            for alphabet in i:
                if alphabet == 'A':
                    cano_alphabet = 'T'
                elif alphabet == 'T':
                    cano_alphabet = 'A'
                elif alphabet == 'C':
                    cano_alphabet = 'G'
                else:
                    cano_alphabet = 'C'
                j = j + cano_alphabet
            # reverse j

            j_re = j[::-1]
            cano_v.append(j_re)

        for i in range(len(vocab)):
            # if cano_v[i] exist in vocab[j],j<i
            vocab_before_i = vocab[0:i]  # vocab[j]
            if cano_v[i] in vocab_before_i:
                # remove cano_v[i] later, replace it with a mark first. Marker :'delete'
                vocab[i] = 'delete'
        # print(vocab_before_i)

        vocab = list(filter(lambda x: x != 'delete', vocab))  # filter the markers in vocab.
        # vocab = [''.join(xs) for xs in itertools.product('ACGT', repeat=k)]
        print('vocab size(just check)', len(vocab))
    else:
        vocab = [''.join(xs) for xs in itertools.product('ACGT', repeat=k)]
        print('vocab size(just check)', len(vocab))

    return vocab