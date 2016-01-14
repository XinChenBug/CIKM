from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
stopWords = stopwords.words('english')
from GenerateNgram import *


lable,Query,Title,Query_Len,Query_Freq,Query_First,Query_Last = ReadTraining_feature()
Trainngram("D:\\Featureusedfortest\\Query_pig_train.txt",True,"D:\\abstractfeatures\\Query_word_vector")
Trainngram("D:\\Featureusedfortest\\Title_pig_train.txt",False,"D:\\abstractfeatures\\Title_word_vector")