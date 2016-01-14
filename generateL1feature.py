from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
stopWords = stopwords.words('english')
from GenerateNgram import *
import json

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