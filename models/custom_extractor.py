import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download(['punkt', 'stopwords', 'wordnet'])

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def replace_url(text):
    detected_urls = re.findall(url_regex, text)

    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    return text

def tokenize(text):
    text = replace_url(text)
    text = re.sub('[^a-zA-Z0-9]', ' ', text).lower()
    tokens = word_tokenize(text)
    token_list = []

    for token in tokens:
        if token not in stopwords.words("english"):
            token_list.append(token)
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in token_list:
        clean_token = lemmatizer.lemmatize(token).strip()
        clean_tokens.append(clean_token)
    return clean_tokens

class DisasterWordExtractor(BaseEstimator, TransformerMixin):

    def disaster_words(self, text):
        words = ['food', 'hunger', 'hungry', 'starving', 'water', 'drink',
                 'eat', 'thirst',
                 'need', 'hospital', 'medicine', 'medical', 'ill', 'pain',
                 'disease', 'injured', 'falling',
                 'wound', 'dying', 'death', 'dead', 'aid', 'help',
                 'assistance', 'cloth', 'cold', 'wet', 'shelter',
                 'hurricane', 'earthquake', 'flood', 'live', 'alive', 'child',
                 'people', 'shortage', 'blocked',
                 'gas', 'pregnant', 'baby'
                 ]
        lemmatized_words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]
        stem_disaster_words = [PorterStemmer().stem(w) for w in lemmatized_words]

        text = replace_url(text)

        clean_tokens = tokenize(text)
        for token in clean_tokens:
            if token in stem_disaster_words:
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_disaster_word = pd.Series(X).apply(self.disaster_words)
        return pd.DataFrame(X_disaster_word)