import numpy as np 
import pandas as pd 
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import sqlite3

# read data from SQLlite 
conn = sqlite3.connect('../data/disaster_tweets.db')
df = pd.read_sql('SELECT * FROM messages', conn)
X = df['message']
Y = df[df.columns[3:]]

lemmatizer = WordNetLemmatizer()
def _tokenize(text):
    inner_text = re.sub(r"[^a-zA-Z0-1]"," ", text.lower())
    tokens = word_tokenize(inner_text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')]
    return tokens
