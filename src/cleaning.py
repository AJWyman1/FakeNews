import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud, STOPWORDS

if __name__ == "__main__":
    plt.style.use('ggplot')
    news_df = pd.read_csv('data/fake-news/train.csv')
    fake_df = pd.read_csv('data/Fake.csv')
    true_df = pd.read_csv('data/True.csv')
    print(fake_df.subject.unique())
    # count_vect = CountVectorizer(lowercase=True, tokenizer=None, stop_words='english', analyzer='word')
    # count_vect.fit(fake_df.text)
    # counts = count_vect.transform(fake_df.text)
    # # tfidf_transformer = TfidfTransformer(use_idf=True)
    # # tfidf_transformer.fit(counts)

    # # X_tfidf = tfidf_transformer.transform(counts)

    # # nb_model = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    # # nb_model.fit(X_tfidf, fake_df.text)
    md = true_df.head(2)
    print(md.to_markdown())
    # feature_words = count_vect.get_feature_names()
    n = 7 #number of top words associated with the category that we wish to see
    # print(counts)
    stopwords = set(STOPWORDS)
    stopwords.add('wa')
    stopwords.add('hi') 
    stopwords.add('ha')
    stopwords.add('thi')
    stopwords.add('.com')
    word_string = ''
    word_string+=" ".join(fake_df['title'].str.lower())
    word_string+=" ".join(true_df['title'].str.lower())
    cloud = WordCloud(background_color='white',
                          stopwords=stopwords, 
                          random_state=42).generate(word_string)
    # cloud.generate_from_text(fake_df.text)
    fig = plt.figure(1)
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()
    # print(count_vect.vocabulary_)
    # print(news_df['title'].unique)
    # print(news_df)
    # for key in sorted(count_vect.vocabulary_.keys()):
    #     # print(f'{key} \t {count_vect.vocabulary_[key]}')
    #     print("{0:<20s} {1}".format(key, count_vect.vocabulary_[key]))


    # fake_df.subject.value_counts().plot(kind='bar')
    # plt.tight_layout()
    # plt.show()