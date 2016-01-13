import numpy
from sklearn.neighbors import KNeighborsClassifier

from svm_version_01.KNN_features import *

trainingData ,training_lable= KNN_Training_preprocess()
print("training finished !")



neigh = KNeighborsClassifier(n_neighbors=30)
neigh.fit(trainingData, training_lable)

print(shape(trainingData))
print(shape(training_lable))

'''
try:
  clf =joblib.load("train_model.m")
except FileNotFoundError:
  clf.fit(trainingData, training_lable)
  joblib.dump(clf, "train_model.m")
'''

print("KNN finished !")

testingData , testlabel  = KNN_Testing_preprocess()
print("test finished !")


print(numpy.shape(testingData))
print(numpy.shape(trainingData))
print(numpy.shape(training_lable))
print(numpy.shape(testlabel))
#print(clf.predict(testingData[0]))

count = 0
i=0
predictlabel =[]
for test in testingData:
    predictlabel.append(neigh.predict(test))

print(len(predictlabel))
print(len(testlabel))
from sklearn.metrics import f1_score
score = f1_score(testlabel, predictlabel, average='macro')
print(score)
