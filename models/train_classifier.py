from operator import mod
import sys
import nltk
import re
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix,  classification_report 
from sklearn.utils.multiclass import is_multilabel
from sqlalchemy import create_engine, inspect
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    inspector = inspect(engine)

    tablename = inspector.get_table_names()[0]

    df = pd.read_sql_table(tablename, con=engine)

    X = df.message
    Y = df.drop(['id','message','original','genre'], axis=1)
    categories = df.columns[4:]

    return X, Y, categories

def tokenize(text):
   
    stop_words = stopwords.words("english")
    lemmitizer = WordNetLemmatizer()
    
    text = re.sub('[^a-zA-Z0-9]', "", text.lower())
    tokens = nltk.word_tokenize(text)
    tokens = [lemmitizer.lemmatize(word).strip() for word in tokens if word not in stop_words]
    return tokens

def build_model():
    model = Pipeline([
        ('features', FeatureUnion([
            ('nlp_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf',MultiOutputClassifier(RandomForestClassifier(n_estimators=36), n_jobs=-1))
    ])
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred=model.predict(X_test)     
    print(classification_report(Y_test, y_pred, target_names=category_names))
    
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)

      
def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n   DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

        print('building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train[:1000], Y_train[:1000])
        # print(model.score(X_train[:1000], Y_train[:1000]))
        

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
    else:
        print("Please provide the file path of the disaster messages database as the first argument and the filepath of the pickle file to save the model as the second argument.")

if __name__ == '__main__':
    main()