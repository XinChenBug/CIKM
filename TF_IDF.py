from DivideDataintoTrainTesT import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
stopWords = stopwords.words('english')
query_vectorizer =CountVectorizer(stop_words = stopWords)
query_transformer = TfidfTransformer()
title_vectorizer = CountVectorizer(stop_words = stopWords)
title_transformer = TfidfTransformer()
Query_Apears_Times = {}

def TF_IDF_1(inputfile,IsQuery,IsTraning,outputfilename):
    Merge_Query=[]
    Query_Title =[]
    f=open(inputfile,'r')
    for line in f:
        Merge_Query.append(line)
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
          #savetxt("D:\\abstractfeatures\\query_tf_idf_featuret.txt",tf_idf_featuret)
          savetxt(outputfilename,tf_idf_featuret)

       else:
          query_testVectorizerArray = query_vectorizer.transform(Query_Title).toarray()
          tfidf = query_transformer.transform(query_testVectorizerArray)
          tf_idf_featuret = tfidf.todense()
          savetxt(outputfilename,tf_idf_featuret)

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
          savetxt(outputfilename,tf_idf_featuret)
       else:
          testVectorizerArray = title_vectorizer.transform(Query_Title).toarray()
          tfidf = title_transformer.transform(testVectorizerArray)
          tf_idf_featuret = tfidf.todense()
          savetxt(outputfilename,tf_idf_featuret)

    return tf_idf_featuret


def TF_IDF_2(Merge_Query,IsQuery,IsTraning =True, filenale =""):
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
           #paddtr = []
           #paddtr.extend(['p','p'])
           #paddtr.extend(tokens)
           #paddtr.extend(['p','p'])
           bi_tokens = nltk.bigrams(paddbi)
           #tri_tokens = nltk.trigrams(paddtr)
           bi_tokens =[token[0]+token[1]for token in bi_tokens]
           #tri_tokens =[token[0]+token[1]+token[2] for token in tri_tokens]

           tmp=""
           for token in tokens:
                tmp+= token
                tmp+=' '

           for token in bi_tokens:
                tmp+= token
                tmp+=' '

           #for token in tri_tokens:
            #    tmp+= token
            #    tmp+=' '

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
          #savetxt("D:\\abstractfeatures\\query_tf_idf_featuret.txt",tf_idf_featuret)
          savetxt(filenale,tf_idf_featuret)

       else:
          query_testVectorizerArray = query_vectorizer.transform(Query_Title).toarray()
          tfidf = query_transformer.transform(query_testVectorizerArray)
          tf_idf_featuret = tfidf.todense()
          savetxt(filenale,tf_idf_featuret)

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
              if Gram_Num[item]<20:
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
          savetxt(filenale,tf_idf_featuret)
       else:
          testVectorizerArray = title_vectorizer.transform(Query_Title).toarray()
          tfidf = title_transformer.transform(testVectorizerArray)
          tf_idf_featuret = tfidf.todense()
          savetxt(filenale,tf_idf_featuret)

    return