import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
#from sklearn.externals import joblib
import joblib
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
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('cat_messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # Extract data needed for visuals

    # Genre of messages for Chart No. 1
    # Count messages by genre and genre names 
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Categories of messages for Chart No. 2
    # Count messages in different categories and ordered category names 
    categories=list(df.columns)[4:42]
    category_counts=df[categories].sum().sort_values(ascending=False)
    category_names=list(category_counts.index)

    # Number of Categories per Message for Chart No. 3
    # Count of Categories per Message and ordered category names 
    # Create Copy of DataFrame
    df_num = df[categories]
    
    # Insert column with number of categories per message
    df_num['number_cat']=df[categories].sum(axis=1)

    # Count the number of messages with the same number of categories per message 
    number_cat_counts = df_num['number_cat'].value_counts()
    number_cat_name = list(number_cat_counts.index)

    # Correlation matrix between categories for Chart No. 4
    corr_mat=df[categories].corr()
        
    # Create visuals
    graphs = [
        #Chart No. 1: Distribution of Message Genres
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Chart No. 1: Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
        ,
        #Chart No. 2: Distribution of Categories
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Chart No. 2: Distribution of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'automargin': True
                }
            }
        }
        ,
        #Chart No. 3: Distribution of Number of Categories per Message
        {
            'data': [
                Bar(
                    x=number_cat_name,
                    y=number_cat_counts
                )
            ],

            'layout': {
                'title': 'Chart No. 3: Distribution of Number of Categories per Message',
                'yaxis': {
                    'title': "Count of Messages"
                },
                'xaxis': {
                    'title': "Number of Selected Categories per Message",
                    'dtick': 1
                }
            }
        }
        ,
        #Chart No. 4: Correlation between Categories
        {
            'data': [
                Heatmap(
                    x=categories,
                    y=categories,
                    z=corr_mat
                )
            ],

            'layout': {
                'title': 'Chart No. 4: Correlation between Categories',
                'height': 860,
                'width': 860,
                'xaxis': {
                    'automargin': True
                },
                'yaxis': {
                    'automargin': True
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
    print(classification_results)

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