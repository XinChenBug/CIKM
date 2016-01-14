from utility import *
Query_Apears_Times = {}


def Features_Abstract_F_2():
    f = open('D:\\zizhu\\training data\\train')
    lable = []
    Merge_Query_Title = {}

    Query_Len = []
    Query_Freq = []
    SumValidSample=0
    Query_First =[]
    Query_Last =[]

    for cur in f:
    #for i in range(1000):
        #cur = f.readline()
        curtoken = cur.strip('\n')
        curtoken = curtoken.split('\t')
        if (len(curtoken)>1 and curtoken[0]!='CLASS=UNKNOWN' and curtoken[0]!='CLASS=TEST' ):
              #feature abstract
              #ngram
              SumValidSample+=1
              if Merge_Query_Title.get(curtoken[1])==None:
                 Merge_Query_Title[curtoken[1]]=[]
                 #label
                 Merge_Query_Title[curtoken[1]].append(curtoken[0])
                 Query_Apears_Times[curtoken[1]]=0


              Query_Apears_Times[curtoken[1]]+=1
              if len(curtoken) > 2 and curtoken[-1] is not '-':
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

def Divide_Data_saved():
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
    fileQuery_Len_test = open("D:\\abstractfeatures\\Query_Len_test.txt",'w')
    fileQuery_Freq_test = open("D:\\abstractfeatures\\Query_Freq_test.txt",'w')
    fileQuery_First_test = open("D:\\abstractfeatures\\Query_First_test.txt",'w')
    fileQuery_Last_test = open("D:\\abstractfeatures\\Query_Last_test.txt",'w')

    alllen = len(lable)
    trainlen = int(len(lable)/2)
    num=int(trainlen/2)
    j=0
    for i in range(num):
        filelabel_pig_train.writelines(str(mapLableintoNumber(lable[j])))
        filelabel_pig_train.write('\n')
        fileQuery_pig_train.writelines(Query[j])
        fileQuery_pig_train.write('\n')

        for item in Title[i]:
           fileTitle_pig_train.write(item)
           fileTitle_pig_train.write(' ')
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
    print("Train-pig-size:")
    print(num)
    for i in range(num_pig_test):
        filelabel_pig_test.writelines(str(mapLableintoNumber(lable[j])))
        filelabel_pig_test.write('\n')
        fileQuery_pig_test.writelines(Query[j])
        fileQuery_pig_test.write('\n')

        for item in Title[i]:
            fileTitle_pig_test.write(item)
            fileTitle_pig_test.write(' ')
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
    print("test-pig-size:")
    print(num_pig_test)
    for i in range(num_test):
        filelabel_test.writelines(str(mapLableintoNumber(lable[j])))
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

    print("test-size:")
    print(num_test)

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
    fileQuery_Len_test.close()
    fileQuery_Freq_test.close()
    fileQuery_First_test.close()
    fileQuery_Last_test.close()

    return

def Divide_Data_inHalf__saved():
    lable,Query,Title,Query_Len,Query_Freq,Query_First,Query_Last= Features_Abstract_F_2()


    filelabel_pig_train = open("D:\\abstractfeatures2\\Lable_for_train.txt",'w')
    fileQuery_pig_train = open("D:\\abstractfeatures2\\Query_for_train.txt",'w')
    fileTitle_pig_train = open("D:\\abstractfeatures2\\Title_for_train.txt",'w')
    fileQuery_Len_pig_train = open("D:\\abstractfeatures2\\Query_Len_for_train.txt",'w')
    fileQuery_Freq_pig_train = open("D:\\abstractfeatures2\\Query_Freq_for_train.txt",'w')
    fileQuery_First_pig_train = open("D:\\abstractfeatures2\\Query_First_for_train.txt",'w')
    fileQuery_Last_pig_train = open("D:\\abstractfeatures2\\Query_Last_for_train.txt",'w')


    filelabel_pig_test = open("D:\\abstractfeatures2\\Lable_for_test.txt",'w')
    fileQuery_pig_test = open("D:\\abstractfeatures2\\Query_for_test.txt",'w')
    fileTitle_pig_test = open("D:\\abstractfeatures2\\Title_for_test.txt",'w')
    fileQuery_Len_pig_test = open("D:\\abstractfeatures2\\Query_Len_for_test.txt",'w')
    fileQuery_Freq_pig_test = open("D:\\abstractfeatures2\\Query_Freq_for_test.txt",'w')
    fileQuery_First_pig_test = open("D:\\abstractfeatures2\\Query_First_for_test.txt",'w')
    fileQuery_Last_pig_test = open("D:\\abstractfeatures2\\Query_Last_for_test.txt",'w')


    alllen = len(lable)
    num=int(alllen/2)
    j=0
    for i in range(num):
        filelabel_pig_train.writelines(str(mapLableintoNumber(lable[j])))
        filelabel_pig_train.write('\n')
        fileQuery_pig_train.writelines(Query[j])
        fileQuery_pig_train.write('\n')

        for item in Title[i]:
           fileTitle_pig_train.write(item)
           fileTitle_pig_train.write(' ')
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
    if alllen%2==1:
       num_pig_test+=1
    print("Train-pig-size:")
    print(num)
    for i in range(num_pig_test):
        filelabel_pig_test.writelines(str(mapLableintoNumber(lable[j])))
        filelabel_pig_test.write('\n')
        fileQuery_pig_test.writelines(Query[j])
        fileQuery_pig_test.write('\n')

        for item in Title[i]:
            fileTitle_pig_test.write(item)
            fileTitle_pig_test.write(' ')
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


    print("test-size:")
    print(num_pig_test)

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
    return

def Read_Training_inHalf_feature():

    filelabel_pig_train = open("D:\\Featureusedfortest\\Lable_for_train.txt",'r')
    fileQuery_Len_pig_train = open("D:\\Featureusedfortest\\Query_Len_for_train.txt",'r')
    fileQuery_Freq_pig_train = open("D:\\Featureusedfortest\\Query_Freq_for_train.txt",'r')
    fileQuery_First_pig_train = open("D:\\Featureusedfortest\\Query_First_for_train.txt",'r')
    fileQuery_Last_pig_train = open("D:\\Featureusedfortest\\Query_Last_for_train.txt",'r')

    lable=[]
    for line in filelabel_pig_train:
        lable.append(line)

    Query_Len=[]
    for line in fileQuery_Len_pig_train:
        Query_Len.append(line)

    Query_Freq=[]
    for line in fileQuery_Freq_pig_train:
        Query_Freq.append(line)

    Query_First=[]
    for line in fileQuery_First_pig_train:
        Query_First.append(line)

    Query_Last=[]
    for line in fileQuery_Last_pig_train:
        Query_Last.append(line)

    filelabel_pig_train.close()
    fileQuery_Len_pig_train.close()
    fileQuery_Freq_pig_train.close()
    fileQuery_First_pig_train.close()
    fileQuery_Last_pig_train.close()
    return lable,Query_Len,Query_Freq,Query_First,Query_Last

def Read_Test_inHalf_feature():

    filelabel_test = open("D:\\Featureusedfortest\\Lable_for_test.txt",'r')
    fileQuery_test = open("D:\\Featureusedfortest\\Query_for_test.txt",'r')
    fileQuery_Len_test = open("D:\\Featureusedfortest\\Query_Len_for_test.txt",'r')
    fileQuery_Freq_test = open("D:\\Featureusedfortest\\Query_Freq_for_test.txt",'r')
    fileQuery_First_test = open("D:\\Featureusedfortest\\Query_First_for_test.txt",'r')
    fileQuery_Last_test = open("D:\\Featureusedfortest\\Query_Last_for_test.txt",'r')

    lable=[]
    for line in filelabel_test:
        lable.append(line)

    Query=[]
    for line in fileQuery_test:
        Query.append(line)

    Query_Len=[]
    for line in fileQuery_Len_test:
        Query_Len.append(line)

    Query_Freq=[]
    for line in fileQuery_Freq_test:
        Query_Freq.append(line)

    Query_First=[]
    for line in fileQuery_First_test:
        Query_First.append(line)

    Query_Last=[]
    for line in fileQuery_Last_test:
        Query_Last.append(line)

    filelabel_test.close()
    fileQuery_test.close()
    fileQuery_Len_test.close()
    fileQuery_Freq_test.close()
    fileQuery_First_test.close()
    fileQuery_Last_test.close()
    return lable,Query,Query_Len,Query_Freq,Query_First,Query_Last

def Read_Training_feature():

    filelabel_pig_train = open("D:\\Featureusedfortest\\Lable_pig_train.txt",'r')
    fileQuery_Len_pig_train = open("D:\\Featureusedfortest\\Query_Len_pig_train.txt",'r')
    fileQuery_Freq_pig_train = open("D:\\Featureusedfortest\\Query_Freq_pig_train.txt",'r')
    fileQuery_First_pig_train = open("D:\\Featureusedfortest\\Query_First_pig_train.txt",'r')
    fileQuery_Last_pig_train = open("D:\\Featureusedfortest\\Query_Last_pig_train.txt",'r')

    lable=[]
    for line in filelabel_pig_train:
        lable.append(line)

    Query_Len=[]
    for line in fileQuery_Len_pig_train:
        Query_Len.append(line)

    Query_Freq=[]
    for line in fileQuery_Freq_pig_train:
        Query_Freq.append(line)

    Query_First=[]
    for line in fileQuery_First_pig_train:
        Query_First.append(line)

    Query_Last=[]
    for line in fileQuery_Last_pig_train:
        Query_Last.append(line)

    filelabel_pig_train.close()
    fileQuery_Len_pig_train.close()
    fileQuery_Freq_pig_train.close()
    fileQuery_First_pig_train.close()
    fileQuery_Last_pig_train.close()
    return lable,Query_Len,Query_Freq,Query_First,Query_Last


def Read_Test_feature():

    filelabel_test = open("D:\\Featureusedfortest\\Lable_test.txt",'r')
    fileQuery_test = open("D:\\Featureusedfortest\\Query_test.txt",'r')
    fileQuery_Len_test = open("D:\\Featureusedfortest\\Query_Len_test.txt",'r')
    fileQuery_Freq_test = open("D:\\Featureusedfortest\\Query_Freq_test.txt",'r')
    fileQuery_First_test = open("D:\\Featureusedfortest\\Query_First_test.txt",'r')
    fileQuery_Last_test = open("D:\\Featureusedfortest\\Query_Last_test.txt",'r')

    lable=[]
    for line in filelabel_test:
        lable.append(line)

    Query=[]
    for line in fileQuery_test:
        Query.append(line)

    Query_Len=[]
    for line in fileQuery_Len_test:
        Query_Len.append(line)

    Query_Freq=[]
    for line in fileQuery_Freq_test:
        Query_Freq.append(line)

    Query_First=[]
    for line in fileQuery_First_test:
        Query_First.append(line)

    Query_Last=[]
    for line in fileQuery_Last_test:
        Query_Last.append(line)

    filelabel_test.close()
    fileQuery_test.close()
    fileQuery_Len_test.close()
    fileQuery_Freq_test.close()
    fileQuery_First_test.close()
    fileQuery_Last_test.close()
    return lable,Query,Query_Len,Query_Freq,Query_First,Query_Last

def Read_pig_Test_feature():

    filelabel_test = open("D:\\Featureusedfortest\\Lable_pig_test.txt",'r')
    #fileQuery_test = open("D:\\Featureusedfortest\\Query_pig_test.txt",'r')
    fileQuery_Len_test = open("D:\\Featureusedfortest\\Query_Len_pig_test.txt",'r')
    fileQuery_Freq_test = open("D:\\Featureusedfortest\\Query_Freq_pig_test.txt",'r')
    fileQuery_First_test = open("D:\\Featureusedfortest\\Query_First_pig_test.txt",'r')
    fileQuery_Last_test = open("D:\\Featureusedfortest\\Query_Last_pig_test.txt",'r')

    lable=[]
    for line in filelabel_test:
        lable.append(line)
    '''
    Query=[]
    for line in fileQuery_test:
        Query.append(line)
    '''
    Query_Len=[]
    for line in fileQuery_Len_test:
        Query_Len.append(line)

    Query_Freq=[]
    for line in fileQuery_Freq_test:
        Query_Freq.append(line)

    Query_First=[]
    for line in fileQuery_First_test:
        Query_First.append(line)

    Query_Last=[]
    for line in fileQuery_Last_test:
        Query_Last.append(line)

    filelabel_test.close()
    #fileQuery_test.close()
    fileQuery_Len_test.close()
    fileQuery_Freq_test.close()
    fileQuery_First_test.close()
    fileQuery_Last_test.close()
    return lable,Query_Len,Query_Freq,Query_First,Query_Last


