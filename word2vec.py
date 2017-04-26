# -*- coding: utf-8 -*-
import pandas as pd

dataset = pd.read_excel("pokemon TW TH search.xlsx",skiprows=1,encoding ='utf-8')

from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

from pythainlp.segment import segment

#For strip accents
import unicodedata
def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

def review_to_wordlist(review):
    review_text = review
    review_text = review_text.lower()
    review_text = review_text.replace(u'é','e')
    pokemonSyn = [u'pokemongo',u'โปเกม่อนโก',u'โปเกมอนโก',u'pokemon go']
    for syn in pokemonSyn: review_text = review_text.replace(syn,u'ม่อน')
    removeCharList = ['rt','!','?']
    for c in removeCharList: review_text = review_text.replace(c,'')
    elimSet = ['http','@','#']
    review_word_list = review_text.split()
    for e in elimSet:
        review_word_list = [word for word in review_word_list if e not in word]
        
    reviewSentence = "".join(review_word_list) 
    wordList = segment(reviewSentence)
    
    return wordList


sentences = []  # Initialize an empty list of sentences
print("Parsing Tweet Text sentences from dataset ")
for row in dataset["Tweet Text"].head(5):
    print(row)
    print(type(review_to_wordlist(row)))
    sentences.append(review_to_wordlist(row))


# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print("Training model...")
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling 
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and 
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "quora_context"
model.save(model_name)
