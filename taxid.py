from ete3 import NCBITaxa
ncbi = NCBITaxa()
#for each file name in the data storage directory
with open('i.txt') as input_file:
    for line in input_file:
        print line
        if '__NC'in line:
            id=line[line.find('taxid__')+7:line.find('__NC')]
        else:
            id = line[line.find('taxid__') + 7:line.find('__AC')]
        #lineage w.r.t. each taxa id
        lineage = ncbi.get_lineage(id)
        #rank=ncbi.get_rank(lineage)
        print lineage
        #print rank
        taxid2name  = ncbi.get_taxid_translator(lineage)
        print taxid2name
        out_file = open("lineage.txt", "a")
        out_file.write(str(taxid2name)+"\n")
        out_file.close()
