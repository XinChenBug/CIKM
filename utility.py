import nltk
from numpy import *
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet

def reorderLable(Key_Index):
    if Key_Index =='CLASS=VIDEO | CLASS=GAME':
         Key_Index='CLASS=GAME | CLASS=VIDEO'
    elif Key_Index =='CLASS=VIDEO | CLASS=TRAVEL':
         Key_Index='CLASS=TRAVEL | CLASS=VIDEO'
    elif Key_Index=='CLASS=VIDEO | CLASS=NOVEL':
         Key_Index='CLASS=NOVEL | CLASS=VIDEO'
    elif Key_Index=='CLASS=VIDEO | CLASS=LOTTERY':
         Key_Index='CLASS=LOTTERY | CLASS=VIDEO'
    elif Key_Index=='CLASS=VIDEO | CLASS=ZIPCODE':
         Key_Index='CLASS=ZIPCODE | CLASS=VIDEO'

    elif Key_Index=='CLASS=TRAVEL | CLASS=GAME':
         Key_Index='CLASS=GAME | CLASS=TRAVEL'
    elif Key_Index=='CLASS=NOVEL | CLASS=GAME':
         Key_Index='CLASS=GAME | CLASS=NOVEL'
    elif Key_Index=='CLASS=LOTTERY | CLASS=GAME':
         Key_Index='CLASS=GAME | CLASS=LOTTERY'
    elif Key_Index=='CLASS=ZIPCODE | CLASS=GAME':
         Key_Index='CLASS=GAME | CLASS=ZIPCODE'

    elif Key_Index=='CLASS=TRAVEL | CLASS=NOVEL':
         Key_Index='CLASS=NOVEL | CLASS=TRAVEL'
    elif Key_Index=='CLASS=TRAVEL | CLASS=LOTTERY':
         Key_Index='CLASS=LOTTERY | CLASS=TRAVEL'
    elif Key_Index=='CLASS=TRAVEL | CLASS=ZIPCODE':
         Key_Index='CLASS=ZIPCODE | CLASS=TRAVEL'

    elif Key_Index=='CLASS=NOVEL | CLASS=LOTTERY':
         Key_Index='CLASS=LOTTERY | CLASS=NOVEL'
    elif Key_Index=='CLASS=NOVEL | CLASS=ZIPCODE':
         Key_Index='CLASS=ZIPCODE | CLASS=NOVEL'

    elif Key_Index=='CLASS=ZIPCODE | CLASS=LOTTERY':
         Key_Index='CLASS=LOTTERY | CLASS=ZIPCODE'

    return Key_Index

def mapLableintoNumber(Key_Index):

    value=-1
    if Key_Index=='CLASS=VIDEO':
         value=0
    elif  Key_Index=='CLASS=GAME':
         value=1
    elif  Key_Index=='CLASS=TRAVEL':
         value=2
    elif  Key_Index=='CLASS=NOVEL':
         value=3
    elif  Key_Index=='CLASS=LOTTERY':
         value=4
    elif  Key_Index=='CLASS=ZIPCODE':
         value=5
    elif  Key_Index=='CLASS=OTHER':
         value=6
    elif  Key_Index=='CLASS=GAME | CLASS=VIDEO':
         value=7
    elif Key_Index=='CLASS=TRAVEL | CLASS=VIDEO':
         value=8
    elif Key_Index=='CLASS=NOVEL | CLASS=VIDEO':
         value=9
    elif Key_Index=='CLASS=LOTTERY | CLASS=VIDEO':
         value=10
    elif Key_Index=='CLASS=ZIPCODE | CLASS=VIDEO':
         value=11

    elif Key_Index=='CLASS=GAME | CLASS=TRAVEL':
         value=12
    elif Key_Index=='CLASS=GAME | CLASS=NOVEL':
         value=13
    elif Key_Index=='CLASS=GAME | CLASS=LOTTERY':
         value=14
    elif Key_Index=='CLASS=GAME | CLASS=ZIPCODE':
         value=15

    elif Key_Index=='CLASS=NOVEL | CLASS=TRAVEL':
         value=16
    elif Key_Index=='CLASS=LOTTERY | CLASS=TRAVEL':
         value=17
    elif Key_Index=='CLASS=ZIPCODE | CLASS=TRAVEL':
         value=18

    elif Key_Index=='CLASS=LOTTERY | CLASS=NOVEL':
         value=19
    elif Key_Index=='CLASS=ZIPCODE | CLASS=NOVEL':
         value=20

    elif Key_Index=='CLASS=LOTTERY | CLASS=ZIPCODE':
         value=21

    if value==-1:
        print("error\n")
        print(Key_Index)
        print(value)
    else:
        return value
