#%% import libraries
import numpy as np 
import pandas as pd 
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion

import sys
import pickle
import sqlite3

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# read data from SQLlite 
def read_data(db_name):
    conn = sqlite3.connect(db_name)
    df = pd.read_sql('SELECT * FROM messages', conn)
    return df


def tokenize(text):
    #porter = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    inner_text = re.sub(r"[^a-zA-Z0-9]"," ", text.lower())
    tokens = word_tokenize(inner_text)
    stop_words = stopwords.words('english')
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

def train_model(df):

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=1)),
    ])

    parameters = {  
    #'vect__ngram_range': ((1,1),(1, 2)),
    'tfidf__use_idf': (True, False)
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=1)

    X = df['message']
    Y = df[df.columns[3:]]
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Classification:")
    print(classification_report(y_test, y_pred)) 

    print("Accuracy Score:")
    print(accuracy_score(y_test, y_pred)) 

    return model

def save_model(model):
    # save the model to disk
    pickle.dump(model, open('model.pkl', 'wb'))


      
def main():
    try:
        db_name = argv[1]
    except:
        try:
            db_name = r"..\data\disaster_tweets.db"
        except:
            print('Database not found')

    print('Training the model...')
    model = train_model(read_data(db_name))

    print('Saving the model...')
    save_model(model)

if __name__ == "__main__":
    main()



# %%
