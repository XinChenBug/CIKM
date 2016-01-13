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
    return normDataSet, ranges, minVals
def reorderLable(Key_Index):
    if Key_Index =='CLASS=VIDEO | CLASS=GAME':
         Key_Index='CLASS=GAME | CLASS=VIDEO'
    elif Key_Index =='CLASS=VIDEO | CLASS=TRAVEL':
         Key_Index=='CLASS=TRAVEL | CLASS=VIDEO'
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

