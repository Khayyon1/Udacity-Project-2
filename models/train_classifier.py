from operator import mod
from os import replace
import sys
import nltk
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix,  classification_report 
from sklearn.utils.multiclass import is_multilabel
from sqlalchemy import create_engine, inspect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from custom_extractor import DisasterWordExtractor

def load_data(database_filepath):
    '''
    Returns the Dependent, Independent variables from DataFrame returned from database_filepath

            Parameters:
                    database_filepath (str): a string that looks like a file path, used to locate DB file

            Returns:
                    X (DataFrame): DataFrame of Independent Variables from DB file
                    y (Series): Series of Dependent Target Variable from DB file
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    inspector = inspect(engine)

    tablename = inspector.get_table_names()[0]

    df = pd.read_sql_table(tablename, con=engine)

    X = df['message'].values
    Y = df[df.columns[4:]]

    return X, Y

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

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
    detected_urls = re.findall(url_regex, text)

    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    return text

def tokenize(text):
    '''
    Returns the clean tokens of the text that was provided
    this function will be used in the tokenizer of the 
    CountVectorizer model

            Parameters:
                    text (str): text data from the DataFrame
            
            Returns:
                    clean_tokens (List[str]): list of clean strings
    '''
    # replace each url in text strings with placeholder
    text = replace_url(text)
    # Case Normalization
    text = text.lower() # convert to lowercase
    # remove puntuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # tokenize text
    tokens = nltk.word_tokenize(text)
    token_list = []
    # remove stop words
    for tok in tokens:
        if tok not in stopwords.words("english"):
             token_list.append(tok)
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # iritate through each token
    clean_tokens = []
    for tok in token_list:
        # lemmatize and remove leading and tailing white space
        clean_tok = lemmatizer.lemmatize(tok).strip()
        
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    '''
    Returns the Machine Learning model and optimizes model based on parameters 
    provided to GridSearch

            Parameters:
                    None
            
            Returns:
                    model (Estimator):  Model used for the predictive results
    '''
    model = Pipeline([
        ('features', FeatureUnion([
            ('nlp_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('disaster_words', DisasterWordExtractor())
        ])),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier(max_depth=15)))
    ])

    params = { 'clf__estimator__n_estimators': [50,100]}

    model = GridSearchCV(model, param_grid=params, scoring='recall_micro')

    return model

def evaluate_model(model, X_test, Y_test):
    '''
    Displays the model's results for each of the predictive columns

            Parameters:
                model (Estimator):  Model used for the predictive results
                X_test (DataFrame):
                Y_test (DataFrame):

            Returns:
                None
    '''
    y_pred=model.predict(X_test)

    i = 0
    for col in Y_test:
        print('Feature {}:{}'.format(i+1,col))
        print(classification_report(Y_test[col],y_pred[:,i]))
        i = i+1
    accuracy = (y_pred == Y_test.values).mean()
    print('The model accuracy score is {:.3f}'.format(accuracy))
    

      
def save_model(model, model_filepath):
    '''
    Creates a .pkl file containing the ML model in the project file directory
    
            Parameters:
                    model (Estimator):  Model used for the predictive results
                    model_filepath (str): file path that determines where the .pkl is created

            Returns:
                    generates a .pkl file in the project directory

    '''
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n   DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train[:1000], Y_train[:1000])        

        print('Evaluating model...')
        evaluate_model(model, X_test[:1000], Y_test[:1000])

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
    else:
        print("Please provide the file path of the disaster messages database as the first argument and the filepath of the pickle file to save the model as the second argument.")

if __name__ == '__main__':
    main()