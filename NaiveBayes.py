import os
import glob
import re
import nltk
from nltk.stem.lancaster import LancasterStemmer

#Naive Bayes Implementaion
#Paths for files
path_stoplist = os.path.join(os.path.dirname(__file__), 'stoplist.txt')
path_arxiv = os.path.join(os.path.dirname(__file__), 'articles/arxiv/*.txt')
path_jdm = os.path.join(os.path.dirname(__file__), 'articles/jdm/*.txt')
path_plos = os.path.join(os.path.dirname(__file__), 'articles/plos/*.txt')


#Reading stoplist file and storing it in stoplist_data
stoplist_data = []
with open(path_stoplist, 'r', errors='ignore') as fobj:
	for l in fobj:
		stoplist_data.append(l.strip('\n'))


#Reading files from plos folder and storing it data_plosfiles
data_plosfiles = []
for fpath in glob.glob(path_plos):
	with open(fpath, 'r', errors='ignore') as fobj:
		wordlist = []
		for l in fobj:
			for w in l.split():
				wordlist.append(w.lower())
		data_plosfiles.append(wordlist)


#Reading files from arxiv folder and storing it in data_arxivfiles
data_arxivfiles = []
for fpath in glob.glob(path_arxiv):
	with open(fpath, 'r', errors='ignore') as fobj:
		wordlist = []
		for l in fobj:
			for w in l.split():
				wordlist.append(w.lower())
		data_arxivfiles.append(wordlist)


#Reading files from jdm folder and storing it data_jdmfiles
data_jdmfiles = []
for fpath in glob.glob(path_jdm):
	with open(fpath, 'r', errors='ignore') as fobj:
		wordlist = []
		for l in fobj:
			for w in l.split():
				wordlist.append(w.lower())
		data_jdmfiles.append(wordlist)


#Splitting the data into training and testing sets
trainingSet = [data_arxivfiles[:150], data_jdmfiles[:150], data_plosfiles[:150]]
testingSet = [data_arxivfiles[150:], data_jdmfiles[150:], data_plosfiles[150:]]
length_of_trainingSet = len(trainingSet[0])+len(trainingSet[1])+len(trainingSet[2]) 


#Pre-processing step
#Forming vocabulary
print('\n*****PRE-PROCESSING STEP*****')
print('\n*****FORMING VOCABULARY*****\n')
lancaster_Stemmer = LancasterStemmer()
count_of_word = {}
for ts in trainingSet:
	for doc in ts:
		words_distinct = set(doc)
		for word in words_distinct:
			if word not in stoplist_data and re.match('^[a-zA-Z_-]*$',word):
				w = lancaster_Stemmer.stem(word)
				if w in count_of_word:
					count_of_word[w] += 1
				else:
					count_of_word[w] = 1

vocab=[]
for wrd in count_of_word:
	if count_of_word[wrd] > 1:
		vocab.append(wrd)
vocab.sort()
print('*****VOCABULARY FORMED*****\n')


#Converting the training set into a set of features
print('*****CREATING FEATURE VECTOR*****')
FeatureVector_plos = []
FeatureVector_arxiv = []
FeatureVector_jdm = []


for i in range(len(trainingSet[2])):
	doc = trainingSet[2][i] 
	FeatureVector_plos.append([0]*(len(vocab)+1))
	FeatureVector_plos[i][len(vocab)] = 'P'
	for wrd in doc:
		w = lancaster_Stemmer.stem(wrd)
		try:
			FeatureVector_plos[i][vocab.index(w)] = 1
		except ValueError:
			pass


for i in range(len(trainingSet[0])):
	doc = trainingSet[0][i] 
	FeatureVector_arxiv.append([0]*(len(vocab)+1))
	FeatureVector_arxiv[i][len(vocab)] = 'A'
	for wrd in doc:
		w = lancaster_Stemmer.stem(wrd)
		try:
			FeatureVector_arxiv[i][vocab.index(w)] = 1
		except ValueError:
			pass


for i in range(len(trainingSet[1])): 
	doc = trainingSet[1][i] 
	FeatureVector_jdm.append([0]*(len(vocab)+1))
	FeatureVector_jdm[i][len(vocab)] = 'J'
	for wrd in doc:
		w = lancaster_Stemmer.stem(wrd)
		try:
			FeatureVector_jdm[i][vocab.index(w)] = 1
		except ValueError:
			pass


print('*****CLASSIFICATION STEP*****')

ProbabilityConditional_P = [0]*len(vocab)
ProbabilityConditional_A = [0]*len(vocab)
ProbabilityConditional_J = [0]*len(vocab)


ProbabilityPrior_P = len(data_plosfiles)/length_of_trainingSet
ProbabilityPrior_A = len(data_arxivfiles)/length_of_trainingSet
ProbabilityPrior_J = len(data_jdmfiles)/length_of_trainingSet


for row in FeatureVector_plos:
	for i in range(len(row)):
		if row[i] == 1:
			ProbabilityConditional_P[i] += 1

for row in FeatureVector_arxiv:
	for i in range(len(row)):
		if row[i] == 1:
			ProbabilityConditional_A[i] += 1

for row in FeatureVector_jdm:
	for i in range(len(row)):
		if row[i] == 1:
			ProbabilityConditional_J[i] += 1



ProbabilityConditional_A = [x/len(FeatureVector_arxiv) for x in ProbabilityConditional_A]
ProbabilityConditional_J = [x/len(FeatureVector_jdm) for x in ProbabilityConditional_J]
ProbabilityConditional_P = [x/len(FeatureVector_plos) for x in ProbabilityConditional_P]

# Read test data and convert into feature vectors
FeatureVector_plos_test = []
FeatureVector_arxiv_test = []
FeatureVector_jdm_test = []

for i in range(len(testingSet[2])): 
	doc = testingSet[2][i] 
	FeatureVector_plos_test.append([0]*(len(vocab)))
	for wrd in doc:
		w = lancaster_Stemmer.stem(wrd)
		try:
			FeatureVector_plos_test[i][vocab.index(w)] = 1
		except ValueError:
			pass


for i in range(len(testingSet[0])):
	doc = testingSet[0][i] 
	FeatureVector_arxiv_test.append([0]*(len(vocab)))
	for wrd in doc:
		w = lancaster_Stemmer.stem(wrd)
		try:
			FeatureVector_arxiv_test[i][vocab.index(w)] = 1
		except ValueError:
			pass


for i in range(len(testingSet[1])):
	doc = testingSet[1][i] 
	FeatureVector_jdm_test.append([0]*(len(vocab)))
	for wrd in doc:
		w = lancaster_Stemmer.stem(wrd)
		try:
			FeatureVector_jdm_test[i][vocab.index(w)] = 1
		except ValueError:
			pass


print('*****CLASSIFIED CLASS LABEL*****\n')

A_predictions = 0
for roww in FeatureVector_arxiv_test:
	prob_of_A = ProbabilityPrior_A
	prob_of_J = ProbabilityPrior_J
	prob_of_P = ProbabilityPrior_P
	for i in range(len(roww)):
		if roww[i] == 1:
			prob_of_A = prob_of_A*ProbabilityConditional_A[i]
			prob_of_J = prob_of_J*ProbabilityConditional_J[i]
			prob_of_P = prob_of_P*ProbabilityConditional_P[i]
		else:
			prob_of_A = prob_of_A*(1-ProbabilityConditional_A[i])
			prob_of_J = prob_of_J*(1-ProbabilityConditional_J[i])
			prob_of_P = prob_of_P*(1-ProbabilityConditional_P[i])

	classified_class_label = ''

	if((prob_of_A >= prob_of_J) and (prob_of_A >= prob_of_P)):
		classified_class_label = 'ARXIV'
	elif((prob_of_J >= prob_of_A) and (prob_of_J >= prob_of_P)):
		classified_class_label = 'JDM'
	elif((prob_of_P >= prob_of_A) and (prob_of_P >= prob_of_J)):
		classified_class_label = 'PLOS'
	print('----------------------------')
	print('Actual class: ARXIV')
	print('Classified class: {}'.format(classified_class_label))
	print('----------------------------')
	if(classified_class_label == 'ARXIV'):
		A_predictions+=1


J_predictions = 0
for row in FeatureVector_jdm_test:
	prob_of_A = ProbabilityPrior_A
	prob_of_J = ProbabilityPrior_J
	prob_of_P = ProbabilityPrior_P
	for i in range(len(row)):
		if row[i] == 1:
			prob_of_A = prob_of_A*ProbabilityConditional_A[i]
			prob_of_J = prob_of_J*ProbabilityConditional_J[i]
			prob_of_P = prob_of_P*ProbabilityConditional_P[i]
		else:
			prob_of_A = prob_of_A*(1-ProbabilityConditional_A[i])
			prob_of_J = prob_of_J*(1-ProbabilityConditional_J[i])
			prob_of_P = prob_of_P*(1-ProbabilityConditional_P[i])
	classified_class_label = ''
	if((prob_of_A >= prob_of_J) and (prob_of_A >= prob_of_P)):
		classified_class_label = 'ARXIV'
	elif((prob_of_J >= prob_of_A) and (prob_of_J >= prob_of_P)):
		classified_class_label = 'JDM'
	elif((prob_of_P >= prob_of_A) and (prob_of_P >= prob_of_J)):
		classified_class_label = 'PLOS'
	print('----------------------------')
	print('Actual class: JDM')
	print('Classified class: {}'.format(classified_class_label))
	print('----------------------------')
	if(classified_class_label == 'JDM'):
		J_predictions+=1


P_predictions = 0
for row in FeatureVector_plos_test:
	prob_of_A = ProbabilityPrior_A
	prob_of_J = ProbabilityPrior_J
	prob_of_P = ProbabilityPrior_P
	for i in range(len(row)):
		if row[i] == 1:
			prob_of_A = prob_of_A*ProbabilityConditional_A[i]
			prob_of_J = prob_of_J*ProbabilityConditional_J[i]
			prob_of_P = prob_of_P*ProbabilityConditional_P[i]
		else:
			prob_of_A = prob_of_A*(1-ProbabilityConditional_A[i])
			prob_of_J = prob_of_J*(1-ProbabilityConditional_J[i])
			prob_of_P = prob_of_P*(1-ProbabilityConditional_P[i])
	classified_class_label = ''
	if((prob_of_A >= prob_of_J) and (prob_of_A >= prob_of_P)):
		classified_class_label = 'ARXIV'
	elif((prob_of_J >= prob_of_A) and (prob_of_J >= prob_of_P)):
		classified_class_label = 'JDM'
	elif((prob_of_P >= prob_of_A) and (prob_of_P >= prob_of_J)):
		classified_class_label = 'PLOS'
	
	print('----------------------------')
	print('Actual class: PLOS')
	print('Classified class: {}'.format(classified_class_label))
	print('----------------------------')
	if(classified_class_label == 'PLOS'):
		P_predictions+=1


print('\n*****ACCURACY*****\n')

print('ARXIV Accuracy = {:.2f}%'.format((A_predictions/len(FeatureVector_arxiv_test)*100)))
print('JDM Accuracy = {:.2f}%'.format((J_predictions/len(FeatureVector_jdm_test)*100)))
print('PLOS Accuracy = {:.2f}%'.format((P_predictions/len(FeatureVector_plos_test)*100)))