#!/usr/bin/env python2

def split(isolates, splits, out_dir):
    isolate_list = []
    with open(isolates) as f:
        for l in f:
            isolate_list.append(l.strip())
    even = len(isolate_list) / splits
    odd = len(isolate_list) % splits
    isolate_splits = []
    i = 0
    for s in range(splits - odd):
        isolate_splits.append(isolate_list[i:(i+even)])
        i += even
    for s in range(odd):
        isolate_splits.append(isolate_list[i:(i+even+1)])
        i += (even + 1)
    print isolate_splits
    import os
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for i, split in zip(range(len(isolate_splits)), isolate_splits):
        with open("%s/isols_%s.txt" %(out_dir, i), 'w') as f:
            f.write("%s\n" % "\n".join(split))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("process indel calling into per sample table")
    parser.add_argument("isolates", help='full list of isolates to be processed')
    parser.add_argument("splits", type = int, help='number of chunks')
    parser.add_argument("out_dir", help = 'output directory for isolate list splits')
    args = parser.parse_args()
    split(**vars(args))
