import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Scatter
from plotly.graph_objs import Figure
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine



app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/disaster_tweets.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    return render_template('master.html')

@app.route('/plots')
def plots():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)


    # Category count
    cat_counts = [] 
    for col in df.columns[3:]:
        cat_counts.append(df[col].sum())

    cat_names = df.columns[3:]

    cat_dict = dict(zip(cat_names, cat_counts))

    cat_counts = [] 
    cat_names = []
    for k, v in sorted(cat_dict.items(), key=lambda item: item[1], reverse=True):
        cat_names.append(k.replace('_',' '))
        cat_counts.append(v)


    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
                 {
            'data': [
                Scatter(
                    x= cat_names,
                    y= cat_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'margin': {'b': 150},
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                     'tickangle' : 45
                }
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genres"
                }
            }
        }


    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('plots.html', ids=ids, graphJSON=graphJSON)

@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[3:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )




def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()