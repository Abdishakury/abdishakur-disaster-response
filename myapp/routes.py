import sys
import json
import plotly
import pandas as pd
import nltk
nltk.download('omw-1.4')

# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
import joblib
from sqlalchemy import create_engine

from myapp import app

# sys.path.insert(0, '/Users/abdis/OneDrive/Desktop/abdishakur-disaster-response/')
# from tokenizer_function import tokenize

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load('models/classifier.pkl')

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    flood_labels=['no flood', 'flood']
    flood_values=list(df.groupby('floods').count()['message'])

    storm_labels = ['no storm', 'storm']
    storm_values = list(df.groupby('storm').count()['message'])

    earthquake_labels = ['no earthquake', 'earthquake']
    earthquake_values = list(df.groupby('earthquake').count()['message'])
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
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
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Pie(
                    labels=flood_labels,
                    values=flood_values,
                )
            ],

            'layout': {
                'title': 'Distribution of Message contain floods',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Floods"
                }
            }
        },

        {
            'data': [
                Pie(
                    labels=storm_labels,
                    values=storm_values,
                )
            ],

            'layout': {
                'title': 'Distribution of Message contain floods',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Floods"
                }
            }
        },

        {
            'data': [
                Pie(
                    labels=earthquake_labels,
                    values=earthquake_values,
                )
            ],

            'layout': {
                'title': 'Distribution of Message contain floods',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Floods"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


