import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import text
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import LinearSVC, SVC

np.random.seed(8675309)


'''
    tfidf -> Nieve Bayes

'''

def load_dataframe():
    fake_df = pd.read_csv('data/Fake_title_text_sentiment.csv')
    fake_df['y'] = np.ones(fake_df.shape[0], dtype='int')
    true_df = pd.read_csv('data/True_title_text_sentiment.csv')
    true_df['y'] = np.zeros(true_df.shape[0], dtype='int')
    df = pd.concat([fake_df, true_df])

    return df

def print_feature_ranking():
    importances = rando_forest.feature_importances_

    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(n):
        print(f"{f + 1}. feature {feature_words[indices[f]]} ({importances[indices[f]]})")

def sentiment_analysis_rf(df):
    sentiment_corpus = df[['com', 'neu', 'pos', 'neg','com_title', 'neu_title', 'pos_title', 'neg_title', 'y']]
    X_sent = sentiment_corpus[['com', 'neu', 'pos', 'neg', 'com_title', 'neu_title', 'pos_title', 'neg_title']]
    y_sent = sentiment_corpus.y
    X_train_sent, X_test_sent, y_train_sent, y_test_sent = train_test_split(X_sent, y_sent)

    rf_class = RandomForestClassifier()
    rf_class.fit(X_train_sent, y_train_sent)
    print(f'Sentiment Analysis Acc = {rf_class.score(X_test_sent, y_test_sent)}')


if __name__ == "__main__":

    my_stop_words = text.ENGLISH_STOP_WORDS.union(['said', '21st', 'century', 'wire', 'reuters','21wire', 'www', 'https', 'com', 'pic'])
    
    naive = MultinomialNB()

    vectorizer = TfidfVectorizer(use_idf=True, stop_words=my_stop_words,smooth_idf=False ,ngram_range=(2,3)) 
    count_vect = CountVectorizer(lowercase=True, stop_words=my_stop_words)

    df = load_dataframe()
    
    sentiment_analysis_rf(df)

    corpus = df[['text', 'y']]
    X = corpus.text
    y = corpus.y
    X_train, X_test, y_train, y_test = train_test_split(X, y)


    count_vect.fit(X_train)
    X_train_counts = count_vect.transform(X_train)
    X_test_counts = count_vect.transform(X_test)

    vectorizer.fit_transform(X)
    X_train_tfidf = vectorizer.transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(type(X_train_tfidf))


    print("Start of model training")


    nb_model = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    nb_model.fit(X_train_tfidf, y_train)
    print(f'naive bayes acc:{nb_model.score(X_test_tfidf, y_test)}')


    lin_clf = LinearSVC()
    lin_clf.fit(X_train_tfidf, y_train)
    print(f'Lin SVC acc: {lin_clf.score(X_test_tfidf, y_test)}')
    

    n = 25

    feature_words = vectorizer.get_feature_names()
    log_prob_fake = nb_model.feature_log_prob_[1]
    i_top = np.argsort(log_prob_fake)[::-1][:n]
    features_topn_fake = [feature_words[i] for i in i_top]
    print(f"Top {n} tokens: ", features_topn_fake)

    log_prob_true = nb_model.feature_log_prob_[0]
    i_top = np.argsort(log_prob_true)[::-1][:n]
    features_topn_true = [feature_words[i] for i in i_top]
    print(f"Top {n} tokens: ", features_topn_true)


    rando_forest = RandomForestClassifier()
    rando_forest.fit(X_train_tfidf, y_train)
    print(f'Random Forest acc: {rando_forest.score(X_test_tfidf, y_test)}')
    titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(rando_forest, X_test_tfidf, y_test,
                                    normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    plt.show()

    importances = rando_forest.feature_importances_

    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(n):
        print(f"{f + 1}. feature {feature_words[indices[f]]} ({importances[indices[f]]})")



    # Plot the impurity-based feature importances of the forest
    # plt.figure()
    # plt.title("Feature importances")
    # plt.bar(range(n), importances[indices],
    #         color="r", yerr=std[indices], align="center")
    # plt.xticks(range(n), indices)
    # plt.xlim([-1, n])
    # plt.show()

    # print(feature_words)

    


