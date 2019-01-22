from __future__ import division
import pandas as pd
from conll_df import conll_df
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support

class tokenWord:
    def __init__(self, word, feature, count):
        self.word=word
        self.feature=feature
        self.count=count

#def loadData(self,fileList)
filePre='./Data/'
fileList=[['/home/karthik/SP19/CompLing/SigMorph/Data/UD_Chinese-GSD/zh_gsd-um-train.conllu','/home/karthik/SP19/CompLing/SigMorph/Data/UD_Chinese-GSD/zh_gsd-um-dev.conllu'],['/home/karthik/SP19/CompLing/SigMorph/Data/UD_Russian-GSD/ru_gsd-um-train.conllu','/home/karthik/SP19/CompLing/SigMorph/Data/UD_Russian-GSD/ru_gsd-um-dev.conllu'],['/home/karthik/SP19/CompLing/SigMorph/Data/UD_English-EWT/en_ewt-um-train.conllu','/home/karthik/SP19/CompLing/SigMorph/Data/UD_English-EWT/en_ewt-um-dev.conllu'],['/home/karthik/SP19/CompLing/SigMorph/Data/UD_Sanskrit-UFAL/sa_ufal-um-train.conllu','/home/karthik/SP19/CompLing/SigMorph/Data/UD_Sanskrit-UFAL/sa_ufal-um-dev.conllu'],['/home/karthik/SP19/CompLing/SigMorph/Data/UD_Spanish-AnCora/es_ancora-um-train.conllu','/home/karthik/SP19/CompLing/SigMorph/Data/UD_Spanish-AnCora/es_ancora-um-dev.conllu'],['/home/karthik/SP19/CompLing/SigMorph/Data/UD_Turkish-IMST/tr_imst-um-train.conllu','/home/karthik/SP19/CompLing/SigMorph/Data/UD_Turkish-IMST/tr_imst-um-dev.conllu']]
list1=[2,5]
for file in fileList:
    with open(file[0]) as fin:
         rows = ( line.split('\t') for line in fin if line[0]!='#' and line[0]!=' ' )
         data = [ row[0:] for row in rows ]

    countDict={}
    fcountDict={}
    uList=[]
    wordList=[]
    for item in data:
        if(len(item)>5):
            itemList=item[5].split(';')
            itemList.sort()
            feature=';'.join(itemList)
            uList.append((item[2].lower(),feature))

    countList=dict(Counter(uList))

    print(len(countList))

    for i in countList:
        change=0
        for x in wordList:
            if x.word == i[0]:
                change=1
                if(x.count<countList[i]):
                    x.count=countList[i]
                    x.feature=i[1]
                        #print(x.word, i[0],x.feature,x.count,countList[i])
                    break
        if change==0:
            wordList.append(tokenWord(i[0],i[1],countList[i]))

    #filePre='./Data/'
    #fileList='/home/karthik/SP19/CompLing/SigMorph/Data/UD_Chinese-GSD/zh_gsd-um-dev.conllu'
    #list1=[2,5]
    #for file in fileList:
    with open(file[1]) as fin:
         rows = ( line.split('\t') for line in fin if line[0]!='#' and line[0]!=' ' )
         data = [ row[0:] for row in rows ]
    instance,correct=0,0
    uList=[]
    y_true=[]
    y_pred=[]
    for item in data:
        if(len(item)>5):
            instance+=1
            itemList=item[5].split(';')
            itemList.sort()
            item[5]=';'.join(itemList)
            searchResult=next((x for x in wordList if x.word ==item[2].lower() ), None)
            if(searchResult is not None):

                y_true.append(item[5])
                y_pred.append(searchResult.feature)
                if(searchResult.feature==item[5]):
                    correct+=1
                y_true.append(item[5])
                y_pred.append(searchResult.feature)
    print(correct/instance)
    print(precision_recall_fscore_support(y_true, y_pred, average='micro'))
    print(len(y_pred))
    print(len(data))
    
