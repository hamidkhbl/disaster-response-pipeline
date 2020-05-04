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
    This method download csv files for the project 
    and stores them on the current directory.
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
    ----------

    Parameters
    ----------

    Returns
    ----------
    '''
    messages = pd.read_csv(messages_csv)
    categories = pd.read_csv(categories_csv)
    df = messages.merge(categories, on = 'id')
    return df


def clean_data(df):
    '''
    Description
    ----------
    
    Parameters
    ----------

    Returns
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
    ----------
    
    Parameters
    ----------

    Returns
    ----------
    '''
    table_name = 'messages'
    engine = create_engine('sqlite:///'+db_name)
    df.to_sql(table_name, engine, index=False)

def main():

    messages_csv = sys.argv[1]
    categories_csv = sys.argv[2]
    db_name = sys.argv[3]

    print('load...')
    df = load_data(messages_csv, categories_csv)

    print('clean...')
    df = clean_data(df)

    print('save...')
    save_data(df, db_name)
    
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler("process_data.log"),
                            logging.StreamHandler()
                            ])
if __name__ == "__main__":
    main()

    