def defaultprecsion(testingData,clf,tlable):
    count = 0
    i=0

    for test in testingData:
        if clf.predict(test)==tlable[i]:
           count+=1
        i+=1

    return float(count)/len(tlable)

def F1_score(testlabel,predictlabel):
    from sklearn.metrics import f1_score
    score = f1_score(testlabel, predictlabel, average='macro')
    return score