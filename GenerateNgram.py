from DivideDataintoTrainTesT import *

import json


def Trainngram(inputfile,IsQuery,outputfilename):
    Gramone=[]
    Gramtwo=[]
    Gramtri=[]
    Data=[]
    f=open(inputfile,'r')
    for line in f:
        line = line.strip('\n')
        Data.append(line)

    for item in Data:
           #ngram

           tokens=item.split(' ')

           paddbi = []
           paddbi.extend('p')
           paddbi.extend(tokens)
           paddbi.extend('p')
           paddtr = []
           paddtr.extend(['p','p'])
           paddtr.extend(tokens)
           paddtr.extend(['p','p'])
           bi_tokens = nltk.bigrams(paddbi)
           tri_tokens = nltk.trigrams(paddtr)
           bi_tokens =[(token[0]+token[1]).strip('\n') for token in bi_tokens]
           tri_tokens =[(token[0]+token[1]+token[2]).strip('\n') for token in tri_tokens]
           Gramone.extend(tokens)
           Gramtwo.extend(bi_tokens)
           Gramtri.extend(tri_tokens)
    threshold=0
    if IsQuery:
        threshold =2
    else:
        threshold =5

    Gram_Numone={}
    for item in Gramone:
         if Gram_Numone.get(item)==None:
             Gram_Numone[item]=1
         else :
             Gram_Numone[item]+=1
    deleteItem = []
    for item in Gram_Numone.keys():
        if Gram_Numone[item]<1:
            deleteItem.append(item)
    for item in deleteItem:
        del Gram_Numone[item]

    #generate dict
    i=0
    for item in Gram_Numone.keys():
        Gram_Numone[item]=i
        i+=1

    file = outputfilename+"gram_1_all.txt"
    json.dump(Gram_Numone, open(file,'w'))

    Gram_Numtwo={}
    for item in Gramtwo:
         if Gram_Numtwo.get(item)==None:
             Gram_Numtwo[item]=1
         else :
             Gram_Numtwo[item]+=1
    deleteItem = []
    for item in Gram_Numtwo.keys():
        if Gram_Numtwo[item]<1:
            deleteItem.append(item)
    for item in deleteItem:
        del Gram_Numtwo[item]

    #generate dict
    i=0
    for item in Gram_Numtwo.keys():
        Gram_Numtwo[item]=i
        i+=1

    file = outputfilename+"gram_2_all.txt"
    json.dump(Gram_Numtwo, open(file,'w'))

    Gram_Numtri={}
    for item in Gramtri:
         if Gram_Numtri.get(item)==None:
             Gram_Numtri[item]=1
         else :
             Gram_Numtri[item]+=1
    deleteItem = []
    for item in Gram_Numtri.keys():
        if Gram_Numtri[item]<1:
            deleteItem.append(item)
    for item in deleteItem:
        del Gram_Numtri[item]

    #generate dict
    i=0
    for item in Gram_Numtri.keys():
        Gram_Numtri[item]=i
        i+=1

    file = outputfilename+"gram_3_all.txt"
    json.dump(Gram_Numtri, open(file,'w'))

    return

def ReadDictFromFile(filename):
   f =open(filename,'r')
   Gram=[]
   for line in f:
       Gram.append(line)

   return Gram

