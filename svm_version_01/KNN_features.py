import nltk
from numpy import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
stopWords = stopwords.words('english')
vectorizer = CountVectorizer(stop_words = stopWords)
transformer = TfidfTransformer()


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges, minVals

def Features_Abstract_F(size,Istraning = True):
    f = open('D:\\zizhu\\training data\\train')
    lable = []
    Merge_Query_Title = {}
    Query_Apears_Times = {}
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

    Merge_Query=[]
    for Key_Index in Merge_Query_Title.keys():
        lable.append(Merge_Query_Title[Key_Index][0])
        Query_Len.append(len(Key_Index))
        Query_Freq.append(round(float(Query_Apears_Times[Key_Index])/SumValidSample,2))
        tmpsplit = Key_Index.split(' ')
        Query_First.append(tmpsplit[0])
        Query_Last.append(tmpsplit[-1])
        tmpsplit.extend(Merge_Query_Title[Key_Index][1:])
        Merge_Query.append(tmpsplit)

    return lable,Merge_Query,Query_Len,Query_Freq,Query_First,Query_Last
def TF_IDF(Merge_Query,IsTraning =True):
    Query_Title =[]
    # ngram and tf-idf
    for item in Merge_Query:
           #ngram
           tokens = [token for curitem in item
             for token in curitem.split(' ')]

           tmp=""
           for token in tokens:
                tmp+= token
                tmp+=' '

           Query_Title.append(tmp[0:-1])
    #tf-idf
    if IsTraning:
       trainVectorizerArray = vectorizer.fit_transform(Query_Title).toarray()
       transformer.fit(trainVectorizerArray)
       tfidf = transformer.transform(trainVectorizerArray)
       tf_idf_featuret = tfidf.todense()

    else:
       testVectorizerArray = vectorizer.transform(Query_Title).toarray()
       transformer.fit(testVectorizerArray)
       tfidf = transformer.transform(testVectorizerArray)
       tf_idf_featuret = tfidf.todense()

    return tf_idf_featuret

trainglines =20000

def KNN_Training_preprocess():
    lable,Merge_Query_Title,Query_Len,Query_Freq,Query_First,Query_Last= Features_Abstract_F(trainglines,True)
    tf_idf_featuret=TF_IDF(Merge_Query_Title,True)
    featuret=hstack((mat(Query_Freq).T,mat(Query_Len).T))
    featuret=hstack((featuret,mat(Query_First).T))
    featuret=hstack((featuret,mat(Query_Last).T))
    featuret=hstack((tf_idf_featuret,featuret))

    return featuret,lable

def KNN_Testing_preprocess():
    lable,Merge_Query_Title,Query_Len,Query_Freq,Query_First,Query_Last= Features_Abstract_F(trainglines,False)
    tf_idf_featuret=TF_IDF(Merge_Query_Title,False)
    featuret=hstack((mat(Query_Freq).T,mat(Query_Len).T))
    featuret=hstack((featuret,mat(Query_First).T))
    featuret=hstack((featuret,mat(Query_Last).T))
    featuret=hstack((tf_idf_featuret,featuret))

    return featuret,lable