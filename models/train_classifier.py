#Import Libraries

import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import sqlite3
from sqlalchemy import create_engine

def load_data(database_filepath):
    """Read in the Messages_Catagories dataset from the SQL database specified as database_filepath
        function then isolates the predictor data and Y variables - these are returned
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('SELECT * FROM Messages_Catagories', con = engine)
    X = df['message']
    Y = df[list(df.columns)[4:]]
    return X,Y

def tokenize(text):
    """The function takes text as input, this text is then tokenized and cleaned. Cleaned tokens from the text are returned. """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Creates a pipline which applies the word tokenizer, applies the TFidTransformer on the tokens then fits
    a Random forest classifier for multiple classifications. A grid search is also performed to tune the model performance.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__max_df': (0.5, 0.75, 1.0),
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test):
    """Function takes a model , X_test and Y_test data as input and returns the 
        model performance statistics on the test set as dictionaries each instance in the dictionary is the output for each classifier 1-36"""
    
    Y_pred = model.predict(X_test)

    summary_precision = {}
    summary_recall = {}
    summary_fscore = {}

    counter = 0
    for col in Y_test.columns:
        counter = counter+1
        y = Y_test[col]
        pred = Y_pred[:,(counter-1)]
        precision,recall,fscore,support=score(y,pred,average='weighted')

        summary_precision[(counter)] = precision
        summary_recall[(counter)] = recall
        summary_fscore[(counter)] = fscore
        
    return summary_precision, summary_recall, summary_fscore


def save_model(model, model_filepath):
    """saves the model in a specied pkl file """
    joblib_file = "classifier.pkl"
    joblib.dump(model, joblib_file)


def main():
    """Function to run the entire modelling process """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()