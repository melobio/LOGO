import os,sys
import pandas as pd


inputfile = sys.argv[1]
outputfile = sys.argv[2]
marknum = sys.argv[3]

markcol = [i for i in range(int(marknum))]
varcol = ['chrom','pos','rs','ref','alt']
colnames = varcol + ['leadingSNP','r2'] + markcol

def main():
    evalue = readevalue(inputfile)

    evalue_num = evalue.loc[:,markcol]
    sig_num = evalue_num[evalue_num<=0.00001].count(axis=1)
    evalue.insert(7, 'sig_mark', sig_num)
    mins = evalue_num.min(axis=1).to_frame()
    mins = mins[mins[0] <= 0.00001]
    l_sig = mins.index.values
    sig = evalue.loc[l_sig]

    writedf(sig, outputfile)
    return


def writedf(df,O):
    df.to_csv(O, sep = '\t', index = False, float_format = '%f')
    return

def readevalue(F):
    df = pd.read_csv(F, names=colnames,header=0)
    return df

if __name__ =="__main__":
    main()
