from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
stopWords = stopwords.words('english')
from GenerateNgram import *


lable,Query,Title,Query_Len,Query_Freq,Query_First,Query_Last = ReadTraining_feature()
#Trainngram("C:\\Users\\v-chexi\\Documents\\CIKM\\Featureusedfortest\\Query_pig_train.txt",True,"C:\\Users\\v-chexi\\Documents\\CIKM\\abstractfeatures\\Query_word_vector")
Trainngram("C:\\Users\\v-chexi\\Documents\\CIKM\\Featureusedfortest\\Title_pig_train.txt",False,"C:\\Users\\v-chexi\\Documents\\CIKM\\abstractfeatures\\Title_word_vector")