# -*- coding: utf-8 -*-

#Import packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import timeit
import codecs
from bs4 import BeautifulSoup # $ pip install beautifulsoup4
from os import walk

#Search through the dataset directory to get all of the file directory.
mypath = 'C:\\Users\\Teerapat\\Desktop\\NLPProjects\\corpus\\texts_tagged_200'

def getfilesPath(path):
    files = []
    fileName = []
    d = [mypath]
    for (dirpath, dirnames, filenames) in walk(mypath):
        if(dirpath)!='./': dirpath+="/"
        files.extend([dirpath+s for s in filenames])
        fileName.extend([s for s in filenames])
    return files

files = getfilesPath(mypath)


print(files[0]) #sample
print(len(files))

def xml2text(directory):
    file = codecs.open(directory,'r','utf-8')
    fileString = file.read()
    soup = BeautifulSoup(fileString,'lxml')
    sentences = soup.find_all("se")
    all_sentences = []
    for sentence in sentences:
        words = sentence.findAll('w')
        wordsList = [word.getText() for word in words]
        if(len(wordsList)>3): all_sentences.append(wordsList)
    return all_sentences

sentences = []
start = timeit.default_timer()
for file in files[:1000]:
    all_sentences = xml2text(file)
    sentences.extend(all_sentences)
stop = timeit.default_timer()
print("Execution time : "+str(round(stop - start,2))+"s.")

#------------------------------------------Start Preprocessing the data----------------------------------#

text = ''
isStart = ''
for sentence in sentences[0:10]:
    for word in sentence:
        text+=word
        isStart+="0"*(len(word)-1)+"1"
    text+=" "
    isStart+="0"
charArr = list(text)
isStartArr = list(isStart)

#construct df from text and isstart
timeStep = 3
dfDic = {}
for i in range(timeStep):
    if(i-timeStep+1<0): 
        dfDic[i] = charArr[i:i-timeStep+1]
    else:
        dfDic[i] = charArr[i:]
dfDic['isStart'] = isStartArr[timeStep-1:]
    
#Df to Pandas
df = pd.DataFrame(dfDic)

#X and y
X = df[[0,1,2]]
y = df[['isStart']]

#Label each character
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
X = X.apply(le.fit_transform)

trainPer= 0.8
boundary = int(X.shape[0]*trainPer)
X_train = X.iloc[:boundary].values
X_test = X.iloc[boundary:].values

X_train = np.reshape(X_train, (X_train.shape[0], timeStep,1))
X_test = np.reshape(X_test,(X_test.shape[0],timeStep,1))

#--------------------------- RNN--------------------------------------#
# Importing the Keras libraries and packages
#TBC on Python 3 spyder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()
