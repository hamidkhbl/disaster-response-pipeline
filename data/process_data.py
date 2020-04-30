# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import  coloredlogs, logging
import datetime

def process_data():
    '''
    This function reads data from csv files, merges them and cleans the data.
    Finally, stores the cleaned data in a SQLlite database.

    Inputs: No input.
    Output: A SQLlite database file.
    '''
    # logging
    coloredlogs.install()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[
                            logging.FileHandler("process_data.log"),
                            logging.StreamHandler()
    ])

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
            categories_new[c] = categories_new[c].str.split('-',1)
            categories_new[c] = categories_new[c].apply(lambda x: x[1])

        # delete dataframe
        del categories

        # concat messages with categories
        df = pd.concat([df,categories_new], axis = 1, sort = False)

        # delete categories column
        df.drop(['categories'], axis = 1, inplace = True)

        # count duplicates
        duplicates_count = df.shape[0] - df.drop_duplicates().shape[0]

        # drop duplicates
        df.drop_duplicates(inplace = True)

        logging.info('Data cleaned successfully.')
    except:
        logging.exception('Failed to cleaning the data.')

    db_name = 'disaster_tweets.db'
    table_name = 'messages'

    try:
        engine = create_engine('sqlite:///'+db_name)
        df.to_sql(table_name, engine, index=False)
        logging.info('Clean data stored on {} SQLlite database'.format(db_name))
    except:
        logging.exception('Not able to create the database')
        raise

process_data()