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


def Features_Abstract_F_2(size=30000):
    f = open('D:\\zizhu\\training data\\train')
    lable = []
    Merge_Query_Title = {}

    Query_Len = []
    Query_Freq = []
    SumValidSample=0
    Query_First =[]
    Query_Last =[]

   # for cur in f:
    for i in range(1000):
        cur = f.readline()
        curtoken = cur.strip()
        curtoken = curtoken.split('\t')
        if (len(curtoken)>1 and curtoken[0]!='CLASS=UNKNOWN' ):
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
          # paddtr.extend(['p','p'])
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


def Training_saved():
    lable,Query,Title,Query_Len,Query_Freq,Query_First,Query_Last= Features_Abstract_F_2()


    filelabel_pig_train = open("D:\\abstractfeatures\\Lable_pig_train.txt",'w')
    fileQuery_pig_train = open("D:\\abstractfeatures\\Query_pig_train.txt",'w')
    fileTitle_pig_train = open("D:\\abstractfeatures\\Title_pig_train.txt",'w')
    fileQuery_Len_pig_train = open("D:\\abstractfeatures\\Query_Len_pig_train.txt",'w')
    fileQuery_Freq_pig_train = open("D:\\abstractfeatures\\Query_Freq_pig_train.txt",'w')
    fileQuery_First_pig_train = open("D:\\abstractfeatures\\Query_First_pig_train.txt",'w')
    fileQuery_Last_pig_train = open("D:\\abstractfeatures\\Query_Last_pig_train.txt",'w')


    filelabel_pig_test = open("D:\\abstractfeatures\\Lable_pig_test.txt",'w')
    fileQuery_pig_test = open("D:\\abstractfeatures\\Query_pig_test.txt",'w')
    fileTitle_pig_test = open("D:\\abstractfeatures\\Title_pig_test.txt",'w')
    fileQuery_Len_pig_test = open("D:\\abstractfeatures\\Query_Len_pig_test.txt",'w')
    fileQuery_Freq_pig_test = open("D:\\abstractfeatures\\Query_Freq_pig_test.txt",'w')
    fileQuery_First_pig_test = open("D:\\abstractfeatures\\Query_First_pig_test.txt",'w')
    fileQuery_Last_pig_test = open("D:\\abstractfeatures\\Query_Last_pig_test.txt",'w')

    filelabel_test = open("D:\\abstractfeatures\\Lable_test.txt",'w')
    fileQuery_test = open("D:\\abstractfeatures\\Query_test.txt",'w')
    fileTitle_test = open("D:\\abstractfeatures\\Title_test.txt",'w')
    fileQuery_Len_test = open("D:\\abstractfeatures\\Query_Len_test.txt",'w')
    fileQuery_Freq_test = open("D:\\abstractfeatures\\Query_Freq_test.txt",'w')
    fileQuery_First_test = open("D:\\abstractfeatures\\Query_First_test.txt",'w')
    fileQuery_Last_test = open("D:\\abstractfeatures\\Query_Last_test.txt",'w')

    alllen = len(lable)
    trainlen = int(len(lable)/2)
    num=int(trainlen/2)
    j=0
    for i in range(num):
        filelabel_pig_train.writelines(lable[j])
        filelabel_pig_train.write('\n')
        fileQuery_pig_train.writelines(Query[j])
        fileQuery_pig_train.write('\n')
        fileTitle_pig_train.writelines(Title[j])
        fileTitle_pig_train.write('\n')
        fileQuery_Len_pig_train.writelines(str(Query_Len[j]))
        fileQuery_Len_pig_train.write('\n')
        fileQuery_Freq_pig_train.writelines(str(Query_Freq[j]))
        fileQuery_Freq_pig_train.write('\n')
        fileQuery_First_pig_train.writelines(Query_First[j])
        fileQuery_First_pig_train.write('\n')
        fileQuery_Last_pig_train.writelines(Query_Last[j])
        fileQuery_Last_pig_train.write('\n')
        j+=1


    num_pig_test = num
    if trainlen%2==1:
       num_pig_test+=1

    for i in range(num_pig_test):
        filelabel_pig_test.writelines(lable[j])
        filelabel_pig_test.write('\n')
        fileQuery_pig_test.writelines(Query[j])
        fileQuery_pig_test.write('\n')
        fileTitle_pig_test.writelines(Title[j])
        fileTitle_pig_test.write('\n')
        fileQuery_Len_pig_test.writelines(str(Query_Len[j]))
        fileQuery_Len_pig_test.write('\n')
        fileQuery_Freq_pig_test.writelines(str(Query_Freq[j]))
        fileQuery_Freq_pig_test.write('\n')
        fileQuery_First_pig_test.writelines(Query_First[j])
        fileQuery_First_pig_test.write('\n')
        fileQuery_Last_pig_test.writelines(Query_Last[j])
        fileQuery_Last_pig_test.write('\n')
        j+=1


    num_test = trainlen
    if alllen%2==1:
       num_test+=1

    for i in range(num_test):
        filelabel_test.writelines(lable[j])
        filelabel_test.write('\n')
        fileQuery_test.writelines(Query[j])
        fileQuery_test.write('\n')
        fileQuery_Len_test.writelines(str(Query_Len[j]))
        fileQuery_Len_test.write('\n')
        fileQuery_Freq_test.writelines(str(Query_Freq[j]))
        fileQuery_Freq_test.write('\n')
        fileQuery_First_test.writelines(Query_First[j])
        fileQuery_First_test.write('\n')
        fileQuery_Last_test.writelines(Query_Last[j])
        fileQuery_Last_test.write('\n')
        j+=1

    filelabel_pig_train.close()
    fileQuery_pig_train.close()
    fileTitle_pig_train.close()
    fileQuery_Len_pig_train.close()
    fileQuery_Freq_pig_train.close()
    fileQuery_First_pig_train.close()
    fileQuery_Last_pig_train.close()

    filelabel_pig_test.close()
    fileQuery_pig_test.close()
    fileTitle_pig_test.close()
    fileQuery_Len_pig_test.close()
    fileQuery_Freq_pig_test.close()
    fileQuery_First_pig_test.close()
    fileQuery_Last_pig_test.close()

    filelabel_test.close()
    fileQuery_test.close()
    fileTitle_test.close()
    fileQuery_Len_test.close()
    fileQuery_Freq_test.close()
    fileQuery_First_test.close()
    fileQuery_Last_test.close()

   # TF_IDF_2(Query[:num],True,True,"D:\\abstractfeatures\\query_tf_idf_featuret_pig_train.txt")
    #TF_IDF_2(Query[num:],True,False,"D:\\abstractfeatures\\query_tf_idf_featuret_pig_test.txt")
   # TF_IDF_2(Title[:num],False,True,"D:\\abstractfeatures\\tltle_tf_idf_featuret_pig_train.txt")
    #TF_IDF_2(Title[num:],False,False,"D:\\abstractfeatures\\tltle_tf_idf_featuret_pig_test.txt")

    return

def Testing_preprocess_2():
    lable,Query,Title,Query_Len,Query_Freq,Query_First,Query_Last= TEST_Features_Abstract_F_2(2000)
    query_tf_idf_featuret=TF_IDF_2(Query,True,False)
    tltle_tf_idf_featuret=TF_IDF_2(Title,False,False)
    featuret=hstack((mat(Query_Freq).T,mat(Query_Len).T))
    featuret=hstack((featuret,mat(Query_First).T))
    featuret=hstack((featuret,mat(Query_Last).T))
    featuret=hstack((query_tf_idf_featuret,featuret))
    featuret=hstack((tltle_tf_idf_featuret,featuret))
    return featuret,lable