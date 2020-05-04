# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import logging
import datetime
import requests
import os
import sys

def download_csv():
    '''
    Description
    This function downloads required csv files for the project from google drive and saves them on the current directory.
    ----------
    
    Parameters
    None
    ----------

    Returns
    None
    ----------
    '''
    categories_link = 'https://drive.google.com/u/0/uc?id=1cm6VF1dTLPxXt_97haeZUALOlqCpcIch&export=download'
    messages_link = 'https://drive.google.com/u/0/uc?id=1OBvKGf2RWVmQSvndI3_mSMjHsBIivoEw&export=download'

    r = requests.get(categories_link)
    with open('categories.csv', 'wb') as f: 
        f.write(r.content)

    r = requests.get(messages_link)
    with open('messages.csv', 'wb') as f: 
        f.write(r.content)


def load_data(messages_csv, categories_csv):
    '''
    Description
    Loads messages and categories data and merges them.
    ----------

    Parameters
    messages_csv: message files path
    categories_csv: categories file path
    ----------

    Returns
    Merged dataframe
    ----------
    '''
    messages = pd.read_csv(messages_csv)
    categories = pd.read_csv(categories_csv)
    df = messages.merge(categories, on = 'id')
    return df


def clean_data(df):
    '''
    Description
    Performs following data cleanins:
        Extracts all the columns with values from categories column
        Deletes unusful columns
        Drops duplicate and null values
        Converts string numbers to numeric
    ----------
    
    Parameters
    df: Merged dataframe
    ----------

    Returns
    A clean dataframe 
    ----------
    '''
    # Extract column names
    cols = []
    for title in df.categories.tolist()[0].split(';'):
        cols.append(title.split('-')[0])

    # Create a new data frame with categories as columns
    df.categories = df.categories.str.split(';')
    categories_new = pd.DataFrame(df.categories.tolist(),columns = cols)
    for c in cols:
        categories_new[c] = categories_new[c].str.split('-')
        categories_new[c] = categories_new[c].apply(lambda x: float(x[1]))

    # concat messages with categories
    df = pd.concat([df,categories_new], axis = 1, sort = False)

    # delete unuseful columns
    df.drop(['categories','original','child_alone'], axis = 1, inplace = True)
    
    # drop duplicates
    df.drop_duplicates(inplace = True)

    # drop null values
    df.dropna(how='any', inplace = True)

    # convert string columns to numeric
    for c in df.columns[3:]:
        df[c] = pd.to_numeric(df[c])

    # replace 2 (indirectly related) with 1 for related column
    df['related'] = df['related'].replace(2,1)
    df.to_csv('all.csv')
    return df

def save_data(df, db_name):
    '''
    Description
    Saves a dataframe into a SQLlite database.
    ----------
    
    Parameters
    df: Dataframe
    db_name: name of the database
    ----------

    Returns
    None
    ----------
    '''
    table_name = 'messages'
    engine = create_engine('sqlite:///'+db_name)
    df.to_sql(table_name, engine, index=False)

def main():

    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[
                        logging.FileHandler("process_data.log"),
                        logging.StreamHandler()
                        ])
    try:
        messages_csv = sys.argv[1]
        categories_csv = sys.argv[2]
        db_name = sys.argv[3]
    except:
        logging.info('Downloading csv files...')
        try:
            download_csv()
        except:
            logging.exception('Failed to download csv files!')
            raise
        messages_csv = 'categories.csv'
        categories_csv = 'messages.csv'
        db_name = 'disaster_tweets.db'


    logging.info('Loading csv files...')
    try:
        df = load_data(messages_csv, categories_csv)
    except:
        logging.exception('Failed to load the csv files!')
        raise


    logging.info('Cleaning data...')
    try:
        df = clean_data(df)
    except:
        logging.exception('Failed in cleaning phase!')
        raise

    logging.info('Saving data...')
    try:
        save_data(df, db_name)
    except:
        logging.exception('Failed while saving the clean data on the database!')
        raise
    
if __name__ == "__main__":
    main()

    