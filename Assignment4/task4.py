from __future__ import division
from collections import defaultdict
from collections import Counter
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import heapq
import operator
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from sklearn.manifold import TSNE
import matplotlib as plt
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation
from keras.optimizers import Adam


MAX_LENGTH=192

''' To find the 95th percentile
lineLength=[]
file='Data/UD_English-EWT/en_ewt-um-train.conllu'
with open(file) as fin:
		lines = [line[9:len(line)-1]  for line in fin if line!='\n' and line[0:9]=='# text = ' and line[0]!=' ' ]

for rows in lines:
	print(rows)
	print(len(rows))
	lineLength.append(len(rows))

lineLength=np.array(lineLength)
print(np.percentile(lineLength,95))

'''
def readData(file):
	sentences=[]
	sentence_tags=[]
	with open(file) as f:
		 fin = f.readlines()
	i=0
	while i<len(fin):
		if(fin[i])=='\n':
			i+=1
			continue
		if fin[i][0:7]=='# text ':
			i+=1
			tempSent=[]
			tempTag=[]

			while(i<len(fin) and fin[i] is not '\n'):
				lines=fin[i].split('\t')
				if len(tempSent)<MAX_LENGTH:
					tempSent.append(lines[2])
					tempTag.append(lines[5])
				i+=1
			sentences.append(tempSent)
			sentence_tags.append(tempTag)
		else:
			i+=1
	return((sentences,sentence_tags))

def encodeSeq(sequences, categories):
	encodings= []
	for s in sequences:
		enc = []
		for item in s:
			enc.append(np.zeros(categories))
			enc[-1][item] = 1.0
		encodings.append(enc)
	return np.array(encodings)

def trainLSTM(trainSentences_X,trainTags_y,testSentences_X,testTags_y,tag2index,word2index):
	model = Sequential()
	model.add(InputLayer(input_shape=(MAX_LENGTH, )))
	model.add(Embedding(len(word2index), 64))
	model.add(LSTM(64, return_sequences=True))
	model.add(TimeDistributed(Dense(len(tag2index))))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
				  optimizer=Adam(0.001),
				  metrics=['accuracy'])
	model.fit(trainSentences_X, encodeSeq(trainTags_y, len(tag2index)), batch_size=128, epochs=4)#, validation_split=0.2)
	scores = model.evaluate(testSentences_X, encodeSeq(testTags_y, len(tag2index)))
	print(f"{model.metrics_names[1]}: {scores[1] * 100}")

def getLemmaDict(file):
	lemmaDict={}
	lemmaLength=[]
	with open(file) as fin:
		lines = [line.split('\t') for line in fin if line!='\n' and line[0]!='#' and line[0]!=' ' ]
	l=0
	for rows in lines:
		word=str(rows[2])
		lemma=str(rows[3])
		lemmaDict[word]=lemma
		lemmaLength.append(word)
		lemmaLength.append(lemma)
	lemmaLength=np.array(lemmaLength)
	maxLemmaLength=np.percentile(lemmaLength,95
	print(maxLemmaLength)
	return ((lemmaDict,maxLemmaLength))

def getCharDict(dict, flag=False):
	charIndDict={}
	indCharDict={}
	l=-1
	for word in dict:
		for i in range(len(dict)):
			if i not in charDict:
				l+=1
				charIndDict[i]=l
				indCharDict[l]=i
				if flag and len(charDict)==26:
					break
	return((charIndDict,indCharDict))

def encodeLemma(self, C, num_rows):
    x = np.zeros((num_rows, len(self.chars)))
    for i, c in enumerate(C):
        x[i, self.char_indices[c]] = 1
    return x

def decodeLemma(self, x, calc_argmax=True):
    if calc_argmax:
        x = x.argmax(axis=-1)
    return ''.join(self.indices_char[x] for x in x)

def lemmatizeMain():
	testFileList=['UD_English-EWT/en_ewt-um-dev.conllu']
	trainFileList=['UD_English-EWT/en_ewt-um-train.conllu']
	trainDict=getLemmaDict(trainFileList)
	testDict=getLemmaDict(testFileList)
	trainCharacterDict=getCharDict(trainDict)
	testCharacterDict=getCharDict(testDict)


def main():
	#testFileList=['./Data/UD_Spanish-AnCora/es_ancora-um-dev.conllu','./Data/UD_Chinese-GSD/zh_gsd-um-dev.conllu','./Data/UD_Russian-GSD/ru_gsd-um-dev.conllu','./Data/UD_English-EWT/en_ewt-um-dev.conllu','./Data/UD_Sanskrit-UFAL/sa_ufal-um-dev.conllu','./Data/UD_Turkish-IMST/tr_imst-um-dev.conllu']
	#trainFileList=['./Data/UD_Spanish-AnCora/es_ancora-um-train.conllu','./Data/UD_Chinese-GSD/zh_gsd-um-train.conllu','./Data/UD_Russian-GSD/ru_gsd-um-train.conllu','./Data/UD_English-EWT/en_ewt-um-train.conllu','./Data/UD_Sanskrit-UFAL/sa_ufal-um-train.conllu','./Data/UD_Turkish-IMST/tr_imst-um-train.conllu']
	testFileList=['UD_English-EWT/en_ewt-um-dev.conllu']
	trainFileList=['UD_English-EWT/en_ewt-um-train.conllu']
	#for i in range(len(testFileList)):
	trainRet=readData(trainFileList[0])
	trainSentences=trainRet[0]
	trainTags=trainRet[1]
	testRet=readData(testFileList[0])
	testSentences=testRet[0]
	testTags=testRet[1]
	words, tags = set([]), set([])

	for s in trainSentences:
		for w in s:
			words.add(w.lower())

	for ts in trainTags:
		for t in ts:
			tags.add(t)

	word2index = {w: i + 2 for i, w in enumerate(list(words))}
	word2index['PAD'] = 0  # The special value used for padding
	word2index['UNK'] = 1  # The special value used for OOVs

	tag2index = {t: i + 1 for i, t in enumerate(list(tags))}
	tag2index['PAD'] = 0

	trainSentences_X, testSentences_X, trainTags_y, testTags_y = [], [], [], []

	for s in trainSentences:
		s_int = []
		for w in s:
			try:
				s_int.append(word2index[w.lower()])
			except KeyError:
				s_int.append(word2index['UNK'])

		trainSentences_X.append(s_int)

	for s in testSentences:
		s_int = []
		for w in s:
			try:
				s_int.append(word2index[w.lower()])
			except KeyError:
				s_int.append(word2index['UNK'])

		testSentences_X.append(s_int)

	for s in trainTags:
		trainTags_y.append([tag2index[t] for t in s])

	for s in testTags:
		try:
			testTags_y.append([tag2index[t] for t in s])
		except:
			pass

	trainSentences_X = pad_sequences(trainSentences_X, maxlen=MAX_LENGTH, padding='post')
	testSentences_X = pad_sequences(testSentences_X, maxlen=MAX_LENGTH, padding='post')
	trainTags_y = pad_sequences(trainTags_y, maxlen=MAX_LENGTH, padding='post')
	testTags_y = pad_sequences(testTags_y, maxlen=MAX_LENGTH, padding='post')
	trainLSTM(trainSentences_X,trainTags_y,testSentences_X,testTags_y,tag2index,word2index)

	lemmatizeMain()

if __name__ == "__main__":
	main()
