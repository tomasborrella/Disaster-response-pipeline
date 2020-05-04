import json
import plotly
import pandas as pd
import sys

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine

sys.path.append('../helpers')
from utils import tokenize


app = Flask(__name__,
            static_folder='static',
            static_url_path='/static')

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_message_category', engine)

# load model
model = joblib.load('../models/classifier.pkl')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    categories_df = df.iloc[:,4:].sum().sort_values(ascending=False).reset_index()
    categories_df.columns = ['category', 'count']
    categories_counts = categories_df['count']
    categories_names = categories_df['category']

    df['different_categories'] = df.iloc[:, 4:].sum(axis=1)
    df_different_categories_volume = (
        df['different_categories']
            .value_counts()
            .reset_index()
            .rename(columns=dict(index='different_categories', different_categories='volume'))
    )

    Y = df.drop(columns=['id','message','original','genre'])
    Y['sum'] = Y[list(Y.columns)].sum(axis=1)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=categories_counts
                )
            ],

            'layout': {
                'title': 'Distribution of message categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of message genres',
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
                Bar(
                    x=df_different_categories_volume['different_categories'],
                    y=df_different_categories_volume['volume']
                )
            ],

            'layout': {
                'title': 'Number of different categories in messages',
                'yaxis': {
                    'title': "Number of messages"
                },
                'xaxis': {
                    'title': "Number of categories"
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


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
