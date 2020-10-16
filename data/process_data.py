#Import Libraries

import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Function to read the two csv files from (args):
        1. messages_filepath
        2. categories_filepath
        these two datasets are then merged together by id and a dataframe (df) is returned"""
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on='id')
    return df
    
def clean_data(df):
    """Taking a the dataframe df as input the function performs the following:
    1. splits out message categories into 36 separate columns
    2. drops the orginal categories column
    3. removes duplicates across all rows"""
    categories = df['categories'].str.split(pat=';',expand=True)
    row = categories[:1]
    category_colnames = row.apply(lambda x: x.str.split(pat='-',expand=False)[0][0])
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str.split(pat='-').str[1]
        categories[column] = categories[column].astype(float)   

    df=df.drop(columns='categories')

    df = df.join(categories)

    df = df.drop_duplicates()
    
    return df
    
def save_data(df, database_filename):
    """Taking the a dataframe (df) as input this function saves the data into a sql database named database_filename with filename Messages_Catagories """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Messages_Catagories', engine, index=False) 

def main():
    """ This is the scrpit which runs all the functions for the data extract, transform and load process """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()