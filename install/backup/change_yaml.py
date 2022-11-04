
import argparse

''' change contents of yml files. e.g. libffi=3.3=he6710b0_ -->  libffi=3.3'''

def extract_info(old,new):
    open(new, "w")
    with open(new, 'a') as outfile:
        with open(old) as infile:
            line_list = infile.readlines()
            for line in line_list:
                if '=' in line:
                    temp=line.split('=')
                    if len(temp)==3:
                        line_new =line.rsplit('=',1)[0]
                        outfile.write(line_new)
                        outfile.write("\n")
                    else:
                        outfile.write(line)


                else:
                    outfile.write(line)


if __name__== '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--old_file', type=str, required=True,
                        help=' =to be changed')
    parser.add_argument('-n', '--new_file', type=str, required=True,
                        help='changed.')
    parsedArgs=parser.parse_args()
    extract_info(parsedArgs.old_file,parsedArgs.new_file)

