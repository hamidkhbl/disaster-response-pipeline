# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
#import  coloredlogs
import logging
import datetime
import requests
import os

try:
    os.remove('disaster_tweets.db')
except OSError:
    pass

def process_data():
    '''
    This function downloads csv files ,reads data from them, merges and cleans the data.
    Finally, stores the cleaned data in a SQLlite database.

    Inputs: No input.
    Output: A SQLlite database file.
    '''
    # logging
    #coloredlogs.install()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler("process_data.log"),
                            logging.StreamHandler()
    ])

    # download csv files
    categories_link = 'https://drive.google.com/u/0/uc?id=1cm6VF1dTLPxXt_97haeZUALOlqCpcIch&export=download'
    messages_link = 'https://drive.google.com/u/0/uc?id=1OBvKGf2RWVmQSvndI3_mSMjHsBIivoEw&export=download'
    try:
        logging.info('Downloading CSV files ...')
        r = requests.get(categories_link)
        with open('categories.csv', 'wb') as f: 
            f.write(r.content)

        r = requests.get(messages_link)
        with open('messages.csv', 'wb') as f: 
            f.write(r.content)
        logging.info('CSV files downloaded successfully.')
    except:
        logging.exception('Failed to download the CSV files')
        raise

    # load messages and categories dataset
    try:
        messages = pd.read_csv('messages.csv')
        categories = pd.read_csv('categories.csv')
        logging.info('Data imported successfully.')
    except:
        logging.exception('Failed to read csv files.')
        raise
    # clean
    try:
        # merge datasets
        df = messages.merge(categories, on = 'id')

        # Extract column names
        cols = []
        for title in categories.categories.tolist()[0].split(';'):
            cols.append(title.split('-')[0])

        # Create a new data frame with categories as columns
        categories.categories = categories.categories.str.split(';')
        categories_new = pd.DataFrame(categories.categories.tolist(),columns = cols)
        for c in cols:
            categories_new[c] = categories_new[c].str.split('-')
            categories_new[c] = categories_new[c].apply(lambda x: float(x[1]))
        
        # delete dataframe
        del categories

        # concat messages with categories
        df = pd.concat([df,categories_new], axis = 1, sort = False)

        # delete unuseful columns
        df.drop(['categories','original'], axis = 1, inplace = True)

        # count duplicates
        duplicates_count = df.shape[0] - df.drop_duplicates().shape[0]
        
        # drop duplicates
        df.drop_duplicates(inplace = True)

        df.to_csv('all.csv')

        # drop null values
        df.dropna(how='any', inplace = True)

        # convert string columns to numeric
        for c in df.columns[3:]:
            df[c] = pd.to_numeric(df[c])

        # replace 2 (indirectly related) with 1 for related column
        df['related'] = df['related'].replace(2,1)

        logging.info('Data cleaned successfully.')

    except:
        logging.exception('Failed to cleaning the data.')

    db_name = 'disaster_tweets.db'
    table_name = 'messages'

    try:
        engine = create_engine('sqlite:///'+db_name)
        df.to_sql(table_name, engine, index=False)
        logging.info('Clean data stored on {} SQLlite database \n'.format(db_name))
        df.to_csv('all.csv')
    except:
        logging.exception('Not able to create the database')
        raise

process_data()