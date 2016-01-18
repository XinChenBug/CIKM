from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
stopWords = stopwords.words('english')
from GenerateNgram import *
import json
def GeneratedefaultsvmTrainingFile():
    lable,Query_Len,Query_Freq,Query_First,Query_Last = Read_Training_feature()
    onegramVectorizerArray,twogramVectorizerArray,trigramVectorizerArray=Build_Query_WordVector("D:\\Featureusedfortest\\Query_pig_train.txt")
    oneTgramVectorizerArray,twoTgramVectorizerArray = Build_Title_WordVector("D:\\Featureusedfortest\\Query_pig_train.txt")#,triTgramVectorizerArray
    print("training finished !")
    featuret=hstack((mat(Query_Freq).T,mat(Query_Len).T))
    featuret=hstack((featuret,mat(Query_First).T))
    featuret=hstack((featuret,mat(Query_Last).T))
    featuret=hstack((featuret,onegramVectorizerArray))
    featuret=hstack((featuret,twogramVectorizerArray))
    featuret=hstack((featuret,trigramVectorizerArray))
    featuret=hstack((featuret,oneTgramVectorizerArray))
    featuret=hstack((featuret,twoTgramVectorizerArray))
    #featuret=hstack((featuret,triTgramVectorizerArray))

    return featuret,lable
def GeneratedefaultsvmtestingFile():
    tlable,tQuery_Len,tQuery_Freq,tQuery_First,tQuery_Last = Read_pig_Test_feature()
    tonegramVectorizerArray,ttwogramVectorizerArray,ttrigramVectorizerArray=Build_Query_WordVector("D:\\Featureusedfortest\\Query_pig_test.txt")
    toneTgramVectorizerArray,ttwoTgramVectorizerArray= Build_Title_WordVector("D:\\Featureusedfortest\\Query_pig_test.txt")#,ttriTgramVectorizerArray
    testingData=hstack((mat(tQuery_Freq).T,mat(tQuery_Len).T))
    testingData=hstack((testingData,mat(tQuery_First).T))
    testingData=hstack((testingData,mat(tQuery_Last).T))
    testingData=hstack((testingData,tonegramVectorizerArray))
    testingData=hstack((testingData,ttwogramVectorizerArray))
    testingData=hstack((testingData,ttrigramVectorizerArray))
    testingData=hstack((testingData,toneTgramVectorizerArray))
    testingData=hstack((testingData,ttwoTgramVectorizerArray))
    #testingData=hstack((testingData,ttriTgramVectorizerArray))

    return testingData,tlable

def GeneratelibsvmTrainingFile():
    lable,Query_Len,Query_Freq,Query_First,Query_Last = Read_Training_feature()
    onegramVectorizerArray,twogramVectorizerArray,trigramVectorizerArray=Build_Query_WordVector("D:\\Featureusedfortest\\Query_pig_train.txt")
    oneTgramVectorizerArray,twoTgramVectorizerArray = Build_Title_WordVector("D:\\Featureusedfortest\\Query_pig_train.txt")#,triTgramVectorizerArray

    #for libsvm
    trainfile = open("D:\\Featureusedfortest\\libsvmtrain.txt",'w')
    for i in range(len(lable)):
        tmp=str(lable[i]).strip('\n')
        j=1
        tmp +='  %d:' % j
        tmp+=str(Query_Freq[i]).strip('\n')
        j+=1
        tmp+= '  %d:' % j
        tmp+=str(Query_Len[i]).strip('\n')
        j+=1
        tmp+= '  %d:' % j
        tmp+=str(Query_First[i]).strip('\n')
        j+=1
        tmp+= '  %d:' % j
        tmp+=str(Query_Last[i]).strip('\n')
        j+=1

        for item in onegramVectorizerArray[i]:
             tmp+= '  %d:' % j
             tmp+=str(item).strip('\n')
             j+=1

        for item in oneTgramVectorizerArray[i]:
             tmp+= '  %d:' % j
             tmp+=str(item).strip('\n')
             j+=1
        trainfile.write(tmp)
        trainfile.write('\n')

    trainfile.close()

def GeneratelibsvmTestingFile():
    testfile = open("D:\\Featureusedfortest\\libsvmtest.txt",'w')
    tlable,tQuery_Len,tQuery_Freq,tQuery_First,tQuery_Last = Read_pig_Test_feature()
    tonegramVectorizerArray,ttwogramVectorizerArray,ttrigramVectorizerArray=Build_Query_WordVector("D:\\Featureusedfortest\\Query_pig_test.txt")
    toneTgramVectorizerArray,ttwoTgramVectorizerArray= Build_Title_WordVector("D:\\Featureusedfortest\\Query_pig_test.txt")#,ttriTgramVectorizerArray
    for i in range(len(tlable)):
        tmp=str(tlable[i]).strip('\n')
        j=1
        tmp += '  %d:' % j
        tmp+=str(tQuery_Freq[i]).strip('\n')
        j+=1
        tmp+= '  %d:' % j
        tmp+=str(tQuery_Len[i]).strip('\n')
        j+=1
        tmp+= '  %d:' % j
        tmp+=str(tQuery_First[i]).strip('\n')
        j+=1
        tmp+= '  %d:' % j
        tmp+=str(Query_Last[i]).strip('\n')
        j+=1

        for item in tonegramVectorizerArray[i]:
             tmp+= '  %d:' % j
             tmp+=str(item).strip('\n')
             j+=1

        for item in toneTgramVectorizerArray[i]:
             tmp+= '  %d:' % j
             tmp+=str(item).strip('\n')
             j+=1
        testfile.write(tmp)
        testfile.write('\n')

    testfile.close()
def GenerateDict():
    Trainngram("D:\\Featureusedfortest\\Query_pig_train.txt",True,"D:\\abstractfeatures\\Query_word_vector")
    Trainngram("D:\\Featureusedfortest\\Title_pig_train.txt",False,"D:\\abstractfeatures\\Title_word_vector")

def GenerateDict_inhalf():
    Trainngram("D:\\Featureusedfortest\\Query_for_train.txt",True,"D:\\abstractfeatures\\Half_Query_word_vector")
    Trainngram("D:\\Featureusedfortest\\Title_for_train.txt",False,"D:\\abstractfeatures\\Half_Title_word_vector")

def Build_Query_WordVector(filename):

    '''
    QueryoneGramDict=json.load("D:\\Featureusedfortest\\Query_word_vectorgram_1_all.txt")
    QuerytwoGramDict=json.load("D:\\Featureusedfortest\\Query_word_vectorgram_2_all.txt")
    QuerytriGramDict=json.load("D:\\Featureusedfortest\\Query_word_vectorgram_3_all.txt")
    '''
    QueryoneGramDict=json.load(file("D:\\Featureusedfortest\Query_word_vectorgram_1_all.txt"))
    QuerytwoGramDict=json.load(file("D:\\Featureusedfortest\Query_word_vectorgram_2_all.txt"))
    QuerytriGramDict=json.load(file("D:\\Featureusedfortest\Query_word_vectorgram_3_all.txt"))

    Query=ReadDictFromFile(filename)
    inputlen =len(Query)
    Gramone=[]
    Gramtwo=[]
    Gramtri=[]
    for item in Query:
           #ngram
           item=item.strip('\n')
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
           bi_tokens =[(token[0]+token[1]) for token in bi_tokens]
           tri_tokens =[(token[0]+token[1]+token[2]) for token in tri_tokens]

           onetmp=""
           for token in tokens:
                onetmp+= token
                onetmp+=' '
           Gramone.append(str(onetmp[0:-1]))

           twotmp=""
           for token in bi_tokens:
                twotmp+= token
                twotmp+=' '
           Gramtwo.append(str(twotmp[0:-1]))

           tritmp=""
           for token in tri_tokens:
                tritmp+= token
                tritmp+=' '
           Gramtri.append(str(tritmp[0:-1]))


    if inputlen!=len(Gramone) or inputlen!= len(Gramtwo) :
        print("size don't match !!\n")
    query_vectorizerone =CountVectorizer(stop_words = stopWords)
    query_vectorizerone.vocabulary=QueryoneGramDict
    onegramVectorizerArray = query_vectorizerone.transform(Gramone).toarray()
    #savetxt("D:\\abstractfeatures\\query_tf_idf_featuret.txt",tf_idf_featuret)

    query_vectorizertwo =CountVectorizer(stop_words = stopWords)
    query_vectorizertwo.vocabulary=QuerytwoGramDict
    twogramVectorizerArray = query_vectorizertwo.transform(Gramtwo).toarray()
     #savetxt("D:\\abstractfeatures\\query_tf_idf_featuret.txt",tf_idf_featuret)

    query_vectorizertri =CountVectorizer(stop_words = stopWords)
    query_vectorizertri.vocabulary=QuerytriGramDict
    trigramVectorizerArray = query_vectorizertri.transform(Gramtri).toarray()
     #savetxt("D:\\abstractfeatures\\query_tf_idf_featuret.txt",tf_idf_featuret)

    return onegramVectorizerArray,twogramVectorizerArray,trigramVectorizerArray


def Build_Title_WordVector(filename):
    '''
    QueryoneGramDict=ReadDictFromFile("D:\\Featureusedfortest\\Title_word_vectorgram_1_all.txt")
    QuerytwoGramDict=ReadDictFromFile("D:\\Featureusedfortest\\Title_word_vectorgram_2_all.txt")
    QuerytriGramDict=ReadDictFromFile("D:\\Featureusedfortest\\Title_word_vectorgram_3_all.txt")
    '''

    QueryoneGramDict=json.load(file("D:\\Featureusedfortest\\Title_word_vectorgram_1_all.txt"))
    QuerytwoGramDict=json.load(file("D:\\Featureusedfortest\\Title_word_vectorgram_2_all.txt"))
    QuerytriGramDict=json.load(file("D:\\Featureusedfortest\\Title_word_vectorgram_3_all.txt"))

    Query=ReadDictFromFile(filename)

    inputlen =len(Query)
    Gramone=[]
    Gramtwo=[]
    Gramtri=[]
    for item in Query:
           #ngram
           item=item.strip('\n')
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
           #tri_tokens = nltk.trigrams(paddtr)
           bi_tokens =[(token[0]+token[1]) for token in bi_tokens]
           #tri_tokens =[(token[0]+token[1]+token[2]) for token in tri_tokens]
           onetmp=""
           for token in tokens:
                onetmp+= token
                onetmp+=' '
           Gramone.append(str(onetmp[0:-1]))

           twotmp=""
           for token in bi_tokens:
                twotmp+= token
                twotmp+=' '
           Gramtwo.append(str(twotmp[0:-1]))
           '''
           tritmp=""
           for token in tri_tokens:
                tritmp+= token
                tritmp+=' '
           Gramtri.append(str(tritmp[0:-1]))
           '''

    if inputlen!=len(Gramone) or inputlen!= len(Gramtwo):
        print("train size don't match !!\n")

    query_vectorizerone =CountVectorizer(stop_words = stopWords)
    query_vectorizerone.vocabulary=QueryoneGramDict
    onegramVectorizerArray = query_vectorizerone.transform(Gramone).toarray()
    #savetxt("D:\\abstractfeatures\\query_tf_idf_featuret.txt",tf_idf_featuret)

    query_vectorizertwo =CountVectorizer(stop_words = stopWords)
    query_vectorizertwo.vocabulary=QuerytwoGramDict
    twogramVectorizerArray = query_vectorizertwo.transform(Gramtwo).toarray()
     #savetxt("D:\\abstractfeatures\\query_tf_idf_featuret.txt",tf_idf_featuret)
    '''
    query_vectorizertri =CountVectorizer(stop_words = stopWords)
    query_vectorizertri.vocabulary=QuerytriGramDict
    trigramVectorizerArray = query_vectorizertri.transform(Gramtri).toarray()
     #savetxt("D:\\abstractfeatures\\query_tf_idf_featuret.txt",tf_idf_featuret)
     '''

    return onegramVectorizerArray,twogramVectorizerArray#,trigramVectorizerArray