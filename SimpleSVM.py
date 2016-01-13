from DivideDataintoTrainTesT import*
from TF_IDF import*
from sklearn import svm
from sklearn.externals import joblib
import  numpy

lable,Query,Title,Query_Len,Query_Freq,Query_First,Query_Last = ReadTraining_feature()
Query_train=TF_IDF_1("D:\\Featureusedfortest\\Query_pig_train.txt",True,True,"D:\\abstractfeatures\\train_query_tf_idf_featuret.txt")

featuret=hstack((mat(Query_Freq).T,mat(Query_Len).T))
featuret=hstack((featuret,mat(Query_First).T))
featuret=hstack((featuret,mat(Query_Last).T))
featuret=hstack((Query_train,featuret))

print("training finished !")

clf = svm.SVC()
clf.fit(featuret, lable)
print("SVC finished !")



tlable,tQuery,tQuery_Len,tQuery_Freq,tQuery_First,tQuery_Last = ReadTest_feature()
tQuery_train=TF_IDF_1("D:\\Featureusedfortest\\Query_test.txt",True,False,"D:\\abstractfeatures\\train_query_tf_idf_featuret.txt")
testingData=hstack((mat(tQuery_Freq).T,mat(tQuery_Len).T))
testingData=hstack((featuret,mat(tQuery_First).T))
testingData=hstack((featuret,mat(tQuery_Last).T))
testingData=hstack((tQuery_train,featuret))
print("test finished !")


count = 0
i=0

for test in testingData:
    if clf.predict(test)==tlable[i]:
       count+=1
    i+=1

print(float(count)/len(tlable))

predictlabel =[]
for test in testingData:
    predictlabel.append(clf.predict(test))

from sklearn.metrics import f1_score
score = f1_score(tlable, predictlabel, average='macro')
print(score)





