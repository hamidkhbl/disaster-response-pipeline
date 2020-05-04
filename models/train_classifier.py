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
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import pickle
import sqlite3
import nltk
import datetime
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

def train_clf(df):

    X = df['message']
    Y = df[df.columns[3:]]
    pipeline = Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(max_depth=3))
    ])

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 42)


    # parameters = {
    #         'vect__ngram_range': ((1,1),(1, 2)),
    #         'vect__max_df': (0.5, 0.75, 1.0),
    #         'vect__max_features': (None, 1000, 2000),
    #         'tfidf__use_idf': (True, False),
    #         'clf__n_estimators': [20, 100, 200],
    #         'clf__min_samples_split': [2, 3, 4]
    #     }

    # cv = GridSearchCV(pipeline, param_grid=parameters)

    print('Training the model...')
    s_t = datetime.datetime.now()

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print('accuracy_score:',accuracy_score(y_test, y_pred))

    e_t = datetime.datetime.now()
    print(e_t - s_t)

    # save the model to disk
    pickle.dump(pipeline, open('model.pkl', 'wb'))

try:
    import sys
    db_name = sys.argv[1]
except:
    db_name = '..\data\disaster_tweets.db'
    
def main():
    train_clf(read_data(db_name))

main()



# %%
