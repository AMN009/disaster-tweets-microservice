import numpy as np 
import pandas as pd

# text processing libraries
import re
import string
import nltk
from nltk.corpus import stopwords

# text representation libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

#Training data
train = pd.read_csv('train.csv')

# Testing data 
test = pd.read_csv('test.csv')

# Make text lowercase, remove text in square brackets,remove links,remove punctuation and remove words containing numbers.
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Removing stopwords belonging to english language
def remove_stopwords(text):
    words = [w for w in text if w not in stopwords.words('english')]
    return words

# Takes a list of text and combines them into one large chunk of text.
def combine_text(list_of_text):
    combined_text = ' '.join(list_of_text)
    return combined_text

# Text preprocessing function
def text_preprocessing(text):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    
    nopunc = clean_text(text)
    tokenized_text = tokenizer.tokenize(nopunc)
    remove_stopwords = [w for w in tokenized_text if w not in stopwords.words('english')]
    combined_text = ' '.join(remove_stopwords)
    return combined_text

# Applying the preprocessing function to both test and training datasets
train['text'] = train['text'].apply(lambda x: text_preprocessing(x))
test['text'] = test['text'].apply(lambda x: text_preprocessing(x))

# Representing the data using TF-IDF
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
train_tfidf = tfidf.fit_transform(train['text'])
test_tfidf = tfidf.transform(test["text"])

def text_representation(text):
    return tfidf.transform(text)

# Fitting a simple Naive Bayes on TFIDF
clf = MultinomialNB()
clf.fit(train_tfidf, train["target"])