from DivideDataintoTrainTesT import*
from generateL1feature import*
from GenerateNgram import*
from sklearn import svm
from sklearn.externals import joblib
import  numpy

lable,Query_Len,Query_Freq,Query_First,Query_Last = Read_Training_feature()

onegramVectorizerArray,twogramVectorizerArray,trigramVectorizerArray=Build_Query_WordVector("D:\\Featureusedfortest\\Query_pig_train.txt")
oneTgramVectorizerArray,twoTgramVectorizerArray,triTgramVectorizerArray = Build_Title_WordVector("D:\\Featureusedfortest\\Query_pig_train.txt")
featuret=hstack((mat(Query_Freq).T,mat(Query_Len).T))
featuret=hstack((featuret,mat(Query_First).T))
featuret=hstack((featuret,mat(Query_Last).T))
featuret=hstack((featuret,onegramVectorizerArray))
featuret=hstack((featuret,twogramVectorizerArray))
featuret=hstack((featuret,trigramVectorizerArray))
featuret=hstack((featuret,oneTgramVectorizerArray))
featuret=hstack((featuret,twoTgramVectorizerArray))
featuret=hstack((featuret,triTgramVectorizerArray))
print("training finished !")

clf =joblib.load("train_model.m")
#clf = svm.SVC(C=10000,cache_size=3000)
#clf.fit(featuret, lable)
print("SVC finished !")
#joblib.dump(clf, "train_model.m")


tlable,tQuery,tQuery_Len,tQuery_Freq,tQuery_First,tQuery_Last = Read_pig_Test_feature()
tonegramVectorizerArray,ttwogramVectorizerArray,ttrigramVectorizerArray=Build_Title_WordVector("D:\\Featureusedfortest\\Query_pig_test.txt")
toneTgramVectorizerArray,ttwoTgramVectorizerArray,ttriTgramVectorizerArray = Build_Title_WordVector("D:\\Featureusedfortest\\Query_pig_test.txt")
#tlable,tQuery_Len,tQuery_Freq,tQuery_First,tQuery_Last = Read_Training_feature()
#tonegramVectorizerArray,ttwogramVectorizerArray,ttrigramVectorizerArray=Build_Title_WordVector("D:\\Featureusedfortest\\Query_pig_train.txt")
#toneTgramVectorizerArray,ttwoTgramVectorizerArray,ttriTgramVectorizerArray = Build_Title_WordVector("D:\\Featureusedfortest\\Query_pig_train.txt")
testingData=hstack((mat(tQuery_Freq).T,mat(tQuery_Len).T))
testingData=hstack((testingData,mat(tQuery_First).T))
testingData=hstack((testingData,mat(tQuery_Last).T))
testingData=hstack((testingData,tonegramVectorizerArray))
testingData=hstack((testingData,ttwogramVectorizerArray))
testingData=hstack((testingData,ttrigramVectorizerArray))
testingData=hstack((testingData,toneTgramVectorizerArray))
testingData=hstack((testingData,ttwoTgramVectorizerArray))
testingData=hstack((testingData,ttriTgramVectorizerArray))
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
score = f1_score(tlable, predictlabel)
print(score)





