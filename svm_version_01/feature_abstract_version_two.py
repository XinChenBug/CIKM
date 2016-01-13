from utility import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
stopWords = stopwords.words('english')
query_vectorizer =CountVectorizer(stop_words = stopWords)
query_transformer = TfidfTransformer()
title_vectorizer = CountVectorizer(stop_words = stopWords)
title_transformer = TfidfTransformer()
Query_Apears_Times = {}


def Features_Abstract_F_2(size):
    f = open('D:\\zizhu\\training data\\train')
    lable = []
    Merge_Query_Title = {}

    Query_Len = []
    Query_Freq = []
    SumValidSample=0
    Query_First =[]
    Query_Last =[]

    for i in range(size):
        cur = f.readline()
        curtoken = cur.strip()
        curtoken = curtoken.split('\t')
        if (len(curtoken)>1):
              #feature abstract
              #ngram
              SumValidSample+=1
              if Merge_Query_Title.get(curtoken[1])==None:
                 Merge_Query_Title[curtoken[1]]=[]
                 #label
                 Merge_Query_Title[curtoken[1]].append(curtoken[0])
                 Query_Apears_Times[curtoken[1]]=0


              Query_Apears_Times[curtoken[1]]+=1
              if len(curtoken) > 2:
                 Merge_Query_Title[curtoken[1]].append(curtoken[2])

    Query=[]
    Title = []
    for Key_Index in Merge_Query_Title.keys():
        lable.append(reorderLable(Merge_Query_Title[Key_Index][0]))
        Query_Len.append(len(Key_Index))
        Query_Apears_Times[Key_Index]=round(float(Query_Apears_Times[Key_Index])/SumValidSample,2)
        Query_Freq.append(Query_Apears_Times[Key_Index])
        Query.append(Key_Index)
        tmpsplit = Key_Index.split(' ')
        Query_First.append(tmpsplit[0])
        Query_Last.append(tmpsplit[-1])
        Title.append(Merge_Query_Title[Key_Index][1:])

    return lable,Query,Title,Query_Len,Query_Freq,Query_First,Query_Last

def TEST_Features_Abstract_F_2(size):
    f = open('D:\\zizhu\\training data\\train')
    lable = []
    Query = []
    Title=[]
    Query_Len = []
    Query_Freq = []
    Query_First =[]
    Query_Last =[]


    for i in range(size):
        cur = f.readline()
        del cur
    size = 5000

    for i in range(size):
        cur = f.readline()
        curtoken = cur.strip()
        curtoken = curtoken.split('\t')
        if (len(curtoken)>1):
              #feature abstract
              #ngram
              Query_Len.append(len(curtoken[1]))
              if Query_Apears_Times.get(curtoken[1])==None:
                  Query_Freq.append(0.0)
              else:
                  Query_Freq.append(Query_Apears_Times[curtoken[1]])
              tmpsplit = curtoken[1].split(' ')
              Query_First.append(tmpsplit[0])
              Query_Last.append(tmpsplit[-1])
              Query.append(curtoken[1])
              Title.append(curtoken[1:])
              lable.append(reorderLable(curtoken[0]))


    return lable,Query,Title,Query_Len,Query_Freq,Query_First,Query_Last


def TF_IDF_2(Merge_Query,IsQuery,IsTraning =True,):
    Query_Title =[]
    # ngram and tf-idf
    i=0

    for item in Merge_Query:
           #ngram
           if IsQuery:
               tokens=item.split(' ')
           else :
               tokens = [token for curitem in item
                 for token in curitem.split(' ')]

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
           bi_tokens =[token[0]+token[1]for token in bi_tokens]
           tri_tokens =[token[0]+token[1]+token[2] for token in tri_tokens]

           tmp=""
           for token in tokens:
                tmp+= token
                tmp+=' '

           for token in bi_tokens:
                tmp+= token
                tmp+=' '

           for token in tri_tokens:
                tmp+= token
                tmp+=' '

           Query_Title.append(str(tmp[0:-1]))

    #tf-idf
    if IsQuery:
       if IsTraning:
        # feature select
          Gram_Num={}
          for item in Query_Title:
             for gram in item.split(' '):
               if Gram_Num.get(gram)==None:
                   Gram_Num[gram]=1
               else :
                   Gram_Num[gram]+=1
          deleteItem = []
          for item in Gram_Num.keys():
              if Gram_Num[item]<10:
                  deleteItem.append(item)
          for item in deleteItem:
              del Gram_Num[item]
          i=0
          for item in Gram_Num.keys():
              Gram_Num[item]=i
              i+=1

          query_vectorizer.vocabulary=Gram_Num
          trainVectorizerArray = query_vectorizer.transform(Query_Title).toarray()
          query_transformer.fit(trainVectorizerArray)
          tfidf = query_transformer.transform(trainVectorizerArray)
          tf_idf_featuret = tfidf.todense()
       else:
          query_testVectorizerArray = query_vectorizer.transform(Query_Title).toarray()
          tfidf = query_transformer.transform(query_testVectorizerArray)
          tf_idf_featuret = tfidf.todense()

    else:
       if IsTraning:
           # feature select
          Gram_Num={}
          for item in Query_Title:
             for gram in item.split(' '):
               if Gram_Num.get(gram)==None:
                   Gram_Num[gram]=1
               else :
                   Gram_Num[gram]+=1
          deleteItem = []
          for item in Gram_Num.keys():
              if Gram_Num[item]<10:
                  deleteItem.append(item)
          for item in deleteItem:
              del Gram_Num[item]
          i=0
          for item in Gram_Num.keys():
             Gram_Num[item]=i
             i+=1

          title_vectorizer.vocabulary=Gram_Num
          title_trainVectorizerArray = title_vectorizer.transform(Query_Title).toarray()
          title_transformer.fit(title_trainVectorizerArray)
          tfidf = title_transformer.transform(title_trainVectorizerArray)
          tf_idf_featuret = tfidf.todense()

       else:
          testVectorizerArray = title_vectorizer.transform(Query_Title).toarray()
          tfidf = title_transformer.transform(testVectorizerArray)
          tf_idf_featuret = tfidf.todense()


    return tf_idf_featuret

trainglines =30000

def Training_preprocess_2():
    lable,Query,Title,Query_Len,Query_Freq,Query_First,Query_Last= Features_Abstract_F_2(trainglines)
    query_tf_idf_featuret=TF_IDF_2(Query,True,True)
    tltle_tf_idf_featuret=TF_IDF_2(Title,False,True)
    featuret=hstack((mat(Query_Freq).T,mat(Query_Len).T))
    featuret=hstack((featuret,mat(Query_First).T))
    featuret=hstack((featuret,mat(Query_Last).T))
    featuret=hstack((query_tf_idf_featuret,featuret))
    featuret=hstack((tltle_tf_idf_featuret,featuret))
    return featuret,lable

def Testing_preprocess_2():
    lable,Query,Title,Query_Len,Query_Freq,Query_First,Query_Last= TEST_Features_Abstract_F_2(trainglines)
    query_tf_idf_featuret=TF_IDF_2(Query,True,False)
    tltle_tf_idf_featuret=TF_IDF_2(Title,False,False)
    featuret=hstack((mat(Query_Freq).T,mat(Query_Len).T))
    featuret=hstack((featuret,mat(Query_First).T))
    featuret=hstack((featuret,mat(Query_Last).T))
    featuret=hstack((query_tf_idf_featuret,featuret))
    featuret=hstack((tltle_tf_idf_featuret,featuret))
    return featuret,lable