from __future__ import division
from collections import defaultdict
from collections import Counter
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy
import heapq

fileList=['./Data/UD_Spanish-AnCora/es_ancora-um-train.conllu','./Data/UD_Chinese-GSD/zh_gsd-um-train.conllu','./Data/UD_Russian-GSD/ru_gsd-um-train.conllu','./Data/UD_English-EWT/en_ewt-um-train.conllu','./Data/UD_Sanskrit-UFAL/sa_ufal-um-train.conllu','./Data/UD_Turkish-IMST/tr_imst-um-train.conllu']
for file in fileList:
	print(file)
	wordDict={}
	wordDict1={}#=collections.defaultdict(list)
	tagDict={}
	features=[]
	with open(file) as fin:
		lines = [line.split('\t') for line in fin if line!='\n' and line[0]!='#' and line[0]!=' ' ]
	for rows in lines:
		word=str(rows[2])
		tagDim=str(rows[5])
		if word not in wordDict1:
			wordDict1[word]=list()
		wordDict1[word].append(tagDim)
		if word not in wordDict:
			wordLen=len(word)
			#k=0
			#while(k<3):
			#	k+=1
			if(wordLen>0):
				features.append("PRE"+(word[0:1]))
				features.append("SUF"+(word[wordLen-1]))
			if(wordLen>1):
				features.append("PRE"+(word[0:2]))
				features.append("SUF"+(word[wordLen-2:]))
			if(wordLen>2):
				features.append("PRE"+(word[0:3]))
				features.append("SUF"+(word[wordLen-3:]))
			wordDict[word]=len(wordDict)+1
		if tagDim not in tagDict:
			tagDict[tagDim]=len(tagDict)+1
	preFeat=defaultdict(int)
	sufFeat=defaultdict(int)
	featureList=list(features)
	featDict=defaultdict(int)
	counter=Counter(features)
	freqFeat=dict(counter.most_common(300))
	trainVector=numpy.zeros((len(wordDict), 300))
	print(len(wordDict))
	i=-1
	k=-1
	for f in freqFeat:
		i+=1
		k=-1
		for d in wordDict:
			wordLen=len(str(d))
			k+=1
			if len(f)==4 and wordLen>0:
				if(f[0:3]=='PRE' and f[3:]==d[0]):
					trainVector[k][i]=1
				if(f[0:3]=='SUF' and f[3:]==d[wordLen-1]):
					trainVector[k][i]=1
			elif len(f)==5 and wordLen>1:
				if(f[0:3]=='PRE' and f[3:]==d[0:2]):
					trainVector[k][i]=1
				elif(f[0:3]=='SUF' and f[3:]==d[wordLen-2]):
					trainVector[k][i]=1
			elif len(f)==6 and wordLen>2:
				if(f[0:3]=='PRE' and f[3:]==d[0:3]):
					trainVector[k][i]=1
				elif(f[0:3]=='SUF' and f[3:]==d[wordLen-3]):
					trainVector[k][i]=1
	# print(trainVector)
	tagVector=numpy.zeros((len(wordDict1), len(tagDict)))
	i=-1
	k=-1
	for w in wordDict1:
		i+=1
		k=-1
		for t in tagDict:
			k+=1
			if(t in wordDict1[w]):
				tagVector[i][k]=1
	print(tagVector.shape)

	def getNextBatch( data, labels,num):
		'''
		Return a total of `num` random samples and labels.
		'''
		idx = numpy.arange(0 , data.shape[0])
		numpy.random.shuffle(idx)
		idx = idx[:num]
		data_shuffle = [data[ i] for i in idx]
		labels_shuffle = [labels[ i] for i in idx]

		return numpy.asarray(data_shuffle), numpy.asarray(labels_shuffle)


	x = tf.placeholder(tf.float64)
	y = tf.placeholder(tf.float64)
	sess = tf.Session()
	initW = numpy.random.randn(300, 1)
	W = tf.Variable(initW, dtype=tf.float64)
	score = tf.matmul(x, W)
	#y = tf.convert_to_tensor(tagVector[0])#,preferred_dtype=float64)
	#x = tf.convert_to_tensor(trainVector)#,preferred_dtype=float64)
	init = tf.global_variables_initializer()
	sess.run(init)
	scVal = sess.run(score, {x : trainVector})
	print(scVal.shape)
	prob = 1 / (1 + tf.exp(-score))
	logPY = y * tf.log(prob) + (1-y) *tf.log(1 - prob)

	logLikelihood = -tf.reduce_sum(logPY)
	meanLL = -tf.reduce_mean(logPY)
	dw = tf.gradients(meanLL, [W,])

	batch_size=32
	eta = .01
	loss_batch=[]
	wValue = sess.run(W)
	train_op = tf.train.GradientDescentOptimizer(0.01).minimize(meanLL,var_list=[W,])#,prob,score])
	model = tf.global_variables_initializer()
	print(trainVector.shape)
	print(tagVector[:,0].shape)
	print(len(tagVector[0]))
	# print(tagVector[1].shape)

	for kl in range(len(tagVector[0])):
		for i in range(1000):
			subfeats, subtags = getNextBatch(trainVector, tagVector[:,kl], 32)
			#fixW = tf.assign(W, wValue)
			#sess.run(fixW)
			mll, gradVal = sess.run([train_op,meanLL],{x : subfeats, y : subtags})
			#gradVal = gradVal#[0] #return value is a list
			#wValue -= eta * gradVal

			if(i+1) % 1000 == 0:
				#print('Step #', str(i), 'W = ', str(sess.run(W)))
				print('Loss = ', gradVal)
				loss_batch.append(gradVal)
		
	#plt.plot(range(0, 420, 25), loss_batch, 'r--', label='Batch Loss for tagsets')
	print(sum(loss_batch)/float(len(loss_batch)))
	x1 = np.linspace(0, len(tagVector[0]), len(tagVector[0]), endpoint=False)
	plt.scatter(x1,loss_batch, marker='o', c='b')
	plt.savefig(file+'.png')
