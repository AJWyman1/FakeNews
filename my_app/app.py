from flask import Flask, request
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from build_model import TextClassifier, get_data
from bs4 import BeautifulSoup
from bs4.element import Comment
from urllib.request import urlopen
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


app = Flask(__name__)

def classify(t):
    with open('static/fake_news_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('static/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    return str(model.predict_proba(vectorizer.transform(t))[0][1])
    
    # if model.predict(vectorizer.transform(t)):
    #     return "Fake News"
    # else:
    #     return "True"

def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
        
    com = vs['compound']
    neu = vs['neu']
    pos = vs['pos']
    neg = vs['neg']
        
    return com, neu, pos, neg

def sentiment_analysis_score(text, title):
    com_text, neu_text, pos_text, neg_text = get_sentiment(text)
    com_title, neu_title, pos_title, neg_title = get_sentiment(title)
    df = pd.DataFrame(np.array([[com_text, neu_text, pos_text, neg_text, com_title, neu_title, pos_title, neg_title]]),
                                columns=['com', 'neu', 'pos', 'neg', 'com_title', 'neu_title', 'pos_title', 'neg_title'])
    
    with open('static/sent_analysis_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print(model.predict_proba(df))
    return str(model.predict_proba(df)[0][1])

def abc_article(url):
    '''
    input: response html from abc

    output: Article body text
    '''
    r  = requests.get(url)
    response_html = r.text
    soup2 = BeautifulSoup(response_html, 'html.parser')
    article = soup2.find('article', class_='Article__Content story')
    article_text = article.get_text()
    return article_text

@app.route('/submit')
def index():
    return '''
        <!DOCTYPE html>
        <html>
            <head>
                <meta charset="utf-8">
                <title>Is this Fake News?</title>
            </head>
          <body>
            <!-- page content -->
            
            <h1>Enter in article text body</h1>
            <form action="/predict" method='POST' >
                <input type="text" name="title" style="width: 300px height: 400px;" />
                <input type="text" name="text" style="width: 300px height: 400px;" />  
                <input type="submit" />
            </form>
          </body>
        </html>
        '''
        #TODO make text box bigger

@app.route("/predict", methods=['POST'])
def text_classification():
    title = [str(request.form['title'])]
    text = [str(request.form['text'])]

    
    sentimet_score = sentiment_analysis_score(text, title)

    classification = classify(text)
    return f'''This article is {classification} scores are {sentimet_score}
            <h1>{str(request.form['title'])}</h1>
            <p> {str(request.form['text'])} </p>
            <form>
                <input type="button" value="Try a new article" onclick="history.back()">
            </form>'''
    


if __name__ == '__main__':
    vectorizer = TfidfVectorizer()
    app.run(host='0.0.0.0', port=8080, debug=True)
