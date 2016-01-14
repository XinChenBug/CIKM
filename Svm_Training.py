#from feature_abstract_version_two import *
#from savefeature import *
from DivideDataintoTrainTesT import*
from generateL1feature import *
GenerateDict()
#Divide_Data_saved()
print("training finished !")


'''
clf = svm.SVC()#tol=1e-3, cache_size=200C=10000,cache_size=3000
print(shape(trainingData))
print(shape(training_lable))


try:
  clf =joblib.load("train_model.m")
except FileNotFoundError:
  clf.fit(trainingData, training_lable)
  joblib.dump(clf, "train_model.m")

print("SVC finished !")

testingData , testlabel  = Testing_preprocess_2()
print("test finished !")


print(numpy.shape(testingData))
print(numpy.shape(trainingData))
print(numpy.shape(training_lable))
print(numpy.shape(testlabel))
#print(clf.predict(testingData[0]))

count = 0
i=0

for test in testingData:
    if clf.predict(test)==testlabel[i]:
       count+=1
    i+=1

print(float(count)/len(testlabel))

predictlabel =[]
for test in testingData:
    predictlabel.append(clf.predict(test))

print(len(predictlabel))
print(len(testlabel))
from sklearn.metrics import f1_score
score = f1_score(testlabel, predictlabel, average='macro')
print(score)
'''




