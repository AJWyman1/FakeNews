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
import re


app = Flask(__name__)

def classify(t, n):
    with open('static/model11.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('static/vectorizer11.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    article_for_model = []
    article_for_model.append(t)
    vects = vectorizer.transform(article_for_model)
    
    stop_words = vectorizer.get_stop_words()

    token = vectorizer.build_tokenizer()
    # print(token(str(t).lower()))
    article_tokens = [word for word in token(str(t).lower()) if word not in stop_words]
    # print(article_tokens)
    
    
    return highlighting(vects, model, vectorizer, t, article_tokens, stop_words, n)
    
    # if model.predict(vectorizer.transform(t)):
    #     return "Fake News"
    # else:
    #     return "True"

def article_grams(tokens):
    grams = [' '.join(i) for i in zip(tokens[:-1], tokens[1:])]
    # print(grams)
    # print(type(grams[1]))

    return grams

def highlighting(vects, model, vectorizer, text, tokens, stop_words, n):
    # n=30

    grams = tokens

    feature_words = vectorizer.get_feature_names()
    log_prob_fake = model.feature_log_prob_[1]
    i_top = np.argsort(log_prob_fake)[::-1][:n]
    features_fake = [feature_words[i] for i in i_top]
    # print(f"Top {n} tokens: ", features_fake)

    log_prob_true = model.feature_log_prob_[0]
    i_top = np.argsort(log_prob_true)[::-1][:n]
    features_true = [feature_words[i] for i in i_top]
    # print(f"Top {n} tokens: ", features_true)




    fake_gram_d = dict()
    true_gram_d = dict()

    for gram in grams:
        fake_gram = False
        true_gram = False
        if gram in features_fake:
            fake_gram = True
        if gram in features_true:
            true_gram = True
        if fake_gram and true_gram: 
            fake_gram = False
            true_gram = False
            fake_idx = features_fake.index(gram)
            true_idx = features_true.index(gram)
            if fake_idx < true_idx and ((true_idx - fake_idx) > n**.5):
                fake_gram = True
                # print(f'fake{fake_idx}     true{true_idx}')
            elif true_idx < fake_idx and ((fake_idx - true_idx) > n**.5):
                true_gram = True
                # print(f'fake{fake_idx}     true{true_idx}')

        if fake_gram:
            # print(f"FAKE: {gram} pos:{features_fake.index(gram)}")
            if gram not in fake_gram_d:
                fake_gram_d[gram] = features_fake.index(gram)
        elif true_gram:
            # print(f"True: {gram} pos:{features_true.index(gram)}")
            if gram not in true_gram_d:
                true_gram_d[gram] = features_true.index(gram)
            
    text_list = text.split()

    for idx, word in enumerate(text_list):
        word_no_punc = re.sub(r'[^\w\s]', '', word)
        # print(word)
        # print(idx)

        if word_no_punc.lower() in fake_gram_d:
            text_list[idx] = '<span style="background-color: #' + higlight_intensity(True, fake_gram_d[word_no_punc.lower()], n) + '">' + word + '</span>'

        elif word_no_punc.lower() in true_gram_d:
            text_list[idx] = '<span style="background-color: #' + higlight_intensity(False, true_gram_d[word_no_punc.lower()], n) + '">'  + word + '</span>'



    return ' '.join(text_list)


    # same = [gram for gram in features_true if gram in features_fake]
    # for gram in same:
    #     print(gram)
    #     print(f'True index {features_true.index(gram)}')
    #     print(f'Fake index {features_fake.index(gram)}')

def higlight_intensity(fake, score, n):
    if fake:
        colors = ['ffff00', 'ffff1a', 'ffff33', 'ffff4d', 'ffff66', 'ffff80', 'ffff99', 'ffffb3', 'ffffcc', 'ffffe6']
    else:
        colors = ['00ff00', '1aff1a', '33ff33', '4dff4d', '66ff66', '80ff80', '99ff99', 'b3ffb3', 'ccffcc', 'e6ffe6']
    return colors[int(score/n*10)]

def is_stop_word(word, stop_words):
    return word in stop_words

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
    # print(model.predict_proba(df))
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

@app.route('/')
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
                <input type="text" name="n" style="width: 300px height: 400px;" />  
                <input type="submit" />
            </form>
          </body>
        </html>
        '''
        #TODO make text box bigger

@app.route("/predict", methods=['POST'])
def text_classification():
    title = [str(request.form['title'])]
    text = str(request.form['text'])
    n = int(request.form['n'])

    
    sentimet_score = sentiment_analysis_score(text, title)

    text_w_highlights = classify(text, n)
    return f'''sentiment score is {sentimet_score}
            <h1>{str(request.form['title'])}</h1>
            <p> {text_w_highlights} </p>
            <form>
                <input type="button" value="Try a new article" onclick="history.back()">
            </form>'''
    


if __name__ == '__main__':
    # vectorizer = TfidfVectorizer()
    app.run(host='0.0.0.0', port=8080, debug=True)

    # text = ''' SIOUX FALLS, S.D. -- South Dakota Attorney General Jason Ravnsborg said in a statement late Monday that he realized he had struck and killed a man walking along a rural stretch of highway only after returning to the scene the next day and discovering the body.The state's top law enforcement officer said he initially thought he hit a deer while driving home from a Republican fundraiser on Saturday night. He is under investigation by the South Dakota Highway Patrol.Ravnsborg said he immediately called 911 after the crash on U.S. Highway 14 and that he didn't realize he had hit a man until he returned to the scene the next morning and found him while looking for the animal he thought he had hit.Authorities identified the dead man as 55-year-old Joseph Boever, who had crashed his truck in that area earlier, according to relatives, and was apparently walking toward it near the road when he was hit.Republican Gov. Kristi Noem revealed Sunday that Ravnsborg had been involved in a fatal crash and asked the Department of Public Safety to investigate, but neither she nor the agency provided any details at that point.The Department of Public Safety issued a statement earlier Monday saying only that Ravnsborg told the Hyde County Sheriff’s Office he had hit a deer. Department spokesman Tony Mangan would not confirm whether Ravnsborg called 911, saying it is part of the investigation.The North Dakota Bureau of Criminal Investigation is also participating in the investigation. The South Dakota Division of Criminal Investigation, which would normally be involved, is part of the attorney general's office. It is standard practice to request an outside agency to conduct an investigation when there may be a conflict of interest.Ravnsborg said Sunday he was “shocked and filled with sorrow." He released a second statement on Monday night detailing his account of the accident, saying it was necessary to dispel rumors.Ravnsborg said he was driving from a Republican fundraiser in Redfield to his home some 110 miles (180 kilometers) away when his vehicle hit something he believed was a large animal. Ravnsborg said he called 911 and looked around his vehicle in the dark using a cellphone flashlight. He said all he could see were pieces of his vehicle.After Hyde County Sheriff Mike Volek arrived, the two men surveyed the damage and filled out paperwork for his car to be repaired, the attorney general said.“At no time did either of us suspect that I had been in an accident with a person,” Ravnsborg said.With his car wrecked, Ravnsborg said he borrowed the sheriff's personal car to return to his home in Pierre. The next morning, he and chief of staff Tim Bormann drove back to return the sheriff's car.They stopped at the spot of the accident, where Ravnsborg said he discovered Boever's body in the grass just off the shoulder of the road. He said it was apparent Boever was dead.Ravnsborg said he drove to Volek's house and reported the dead body. They both returned to the accident scene, where Volek said he would handle the investigation and asked Ravnborg to return to Pierre, according to Ravnsborg's statement.Ravnsborg said he was cooperating with the investigation, including providing a blood sample, agreeing to have both of his cellphones searched, and being interviewed by law enforcement agents.Boever's family said Monday they felt frustrated with and suspicious about the investigation, especially after investigators took nearly 22 hours to allow them to identify Boever's body.Boever had crashed his truck into a hay bale near the road on Saturday evening, according to his cousin Victor Nemec. Boever told his cousin that he had been reaching for some tobacco.Nemec had given Boever a ride home, which was about 1.5 miles (2.4 kilometers) away, and made plans to make repairs on Sunday. He left Boever after 9 p.m. The crash that killed him happened around 10:30 p.m. Nemec said “there was no indication whatsoever” that his cousin had been drinking.Boever lived alone and had been separated from his wife, Nick Nemec, another cousin, said.Victor Nemec, the last known person to see Boever, said that besides answering a few brief questions when he identified the body, investigators have not questioned him about what happened.“A human doesn’t look like a deer,” he said. “The whole thing stinks to me.”When Boever’s cousins couldn't find him at his home on Sunday and saw an accident being investigated near where Boever had left his truck, they grew fearful that he was involved. Nemec said he contacted the sheriff around 10 a.m. and was told to wait. As the hours ticked on, they grew more suspicious and called 911 and the Highway Patrol after 5 p.m. They were allowed to identify his body after 8 p.m. on Sunday.“I don’t know if cousin Joe was laying on the highway for 22 hours or if they had bagged him up before that,” Nick Nemec said.Ravnsborg had been at a fundraising dinner hosted by the Spink County Republicans at Rooster’s Bar & Grill. The attorney general is known to be a frequent attendee of the fundraisers known as Lincoln Day Dinners, held by county GOP groups across the state.Bormann said the attorney general drinks occasionally, but has made it a practice not to drink at the Lincoln Day events.“I didn’t see him with anything but a Coke," said state Sen. Brock Greenfield, who also attended the dinner.Ravnsborg has received six traffic tickets for speeding in South Dakota over the last six years. He also received tickets for a seat belt violation and for driving a vehicle without a proper exhaust and muffler system.In 2003, Bill Janklow, a former four-term governor who was a congressman at the time, killed a motorcyclist after running a stop sign at a rural intersection. Janklow was convicted of manslaughter, prompting his resignation.The Department of Public Safety says its investigation into Ravnsborg's crash is ongoing.
    # '''
    # x = classify(text)
    # title = ''' South Dakota's top attorney says found body day after crash '''
    # print(sentiment_analysis_score(text, title))
    # print (x)

