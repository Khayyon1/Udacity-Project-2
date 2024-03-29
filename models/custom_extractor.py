import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin

nltk.download(['punkt', 'stopwords', 'wordnet'])



def replace_url(text):
    '''
    Returns text without the URL in them

    Parameters
    ----------
        text : str
            text that is being cleaned
    Returns
    -------
    text, str
        Text without URL strings 
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    matches = re.findall(url_regex, text)

    for match in matches:
        text = text.replace(match, 'urlplaceholder')
    return text

def tokenize(text):
    '''
    Returns list of cleaned tokens from the text, function used
    by the CountVectorizer in the ML Pipeline

    Parameters
    ----------
    text : str
        string passed into the function by CountVectorizer

    Returns
    -------
    clean_tokens : List[str]
        Cleaned tokens without punctuation marks, stop words,
        and extra white space
    '''
    text = replace_url(text)
    text = re.sub('[^a-zA-Z0-9]', ' ', text).lower()
    tokens = word_tokenize(text)
    tokens_arr = []

    for token in tokens:
        if token not in stopwords.words("english"):
            tokens_arr.append(token)
    
    lemmatizer = WordNetLemmatizer()

    clean_token_arr = []
    for token in tokens_arr:
        tokenized = lemmatizer.lemmatize(token).strip()
        clean_token_arr.append(tokenized)
    return clean_token_arr

class Extractor(BaseEstimator, TransformerMixin):
    '''
    A class that represents a Custom Estimator for ML model

    Attributes
    ----------
    text : str
        text provided by the model for classification, 
        used in the pipeline during the model creation

    Methods
    -------
    words_of_interest(text)
        cleans the text provided and determines if the text is related to any disaster
        words (returns a boolean)
    fit(X, y=None)
        returns an Estimator with no Transformation
    transform(X)
        returns DataFrame resulting from applying disaster_word method to X
    '''
    
    def words_of_interest(self, text):
        '''
        Returns boolean based on if cleaned text has any disaster words in them

        Parameters
        ----------
        text : str
            text provided by the model
        
        Returns
        -------
        bool
        '''
        disasters = ['food', 'hunger', 'hungry', 'starving', 'water', 'drink',
                 'eat', 'thirst',
                 'need', 'hospital', 'medicine', 'medical', 'ill', 'pain',
                 'disease', 'injured', 'falling',
                 'wound', 'dying', 'death', 'dead', 'aid', 'help',
                 'assistance', 'cloth', 'cold', 'wet', 'shelter',
                 'hurricane', 'earthquake', 'flood', 'live', 'alive', 'child',
                 'people', 'shortage', 'blocked',
                 'gas', 'pregnant', 'baby'
                 ]
        lemmatized = [WordNetLemmatizer().lemmatize(w, pos='v') for w in disasters]
        disaster_words = [PorterStemmer().stem(w) for w in lemmatized]

        text = replace_url(text)

        clean_tokens = tokenize(text)
        for token in clean_tokens:
            if token in disaster_words:
                return True
        return False

    def fit(self, X, y=None):
        '''
        Return an Estimator

        Currently does nothing with X or y

        Parameters
        ----------
        X : DataFrame
        y : Series

        Returns
        -------
        self : Estimator Object
        '''
        return self

    def transform(self, X):
        '''
        Returns a new DataFrame from Provided DataFrame X

        Parameters
        ----------
        X : DataFrame
            DataFrame that is being transformed

        Returns
        -------
        new_X : DataFrame
            DataFrame that had the disaster_word method applied to it
        '''
        new_X = pd.Series(X).apply(self.words_of_interest)
        return pd.DataFrame(new_X)