from DivideDataintoTrainTesT import*
from generateL1feature import*
from GenerateNgram import*
from sklearn import svm
from sklearn.externals import joblib
from svmutil import *
from numpy import *
from svmutil import *


#GeneratelibsvmTrainingFile()
#GeneratelibsvmTestingFile()
y,x=svm_read_problem("D:\\Featureusedfortest\\libsvmtrain.txt")
m = svm_train(y,x,'-c 3000 -m 10000')
#clf =joblib.load("simple_train_model.m")
#clf = svm.SVC()#Build_Query_WordVector
#clf.fit(featuret, lable)
print("SVC finished !")
#joblib.dump(clf, "simple_train_model.m")
# joblib.dump(m, "libsvm_simple_train_model.m")




y1,x1=svm_read_problem("D:\\Featureusedfortest\\libsvmtest.txt")
print("test finished !")
p_label, p_acc, p_val = svm_predict(y1, x1, m)


print (p_acc)
print ("acc is above\n")



