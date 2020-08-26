from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

plt.style.use('ggplot')

class sentiment_analysis(object):

    def __init__(self, fake_df, true_df):
        self.fake_df = fake_df
        self.true_df = true_df

    def get_sentiment(self, title, analyzer):
        vs = analyzer.polarity_scores(title)
        
        com = vs['compound']
        neu = vs['neu']
        pos = vs['pos']
        neg = vs['neg']
        
        return com, neu, pos, neg

    def append_four(self, lst1, lst2, lst3, lst4, app1, app2, app3, app4, remove_defaults=True):
        if remove_defaults:
            if app1:
                lst1.append(app1)
            lst2.append(app2)
            if app3:
                lst3.append(app3)
            if app4:
                lst4.append(app4)
        else:
            lst1.append(app1)
            lst2.append(app2)
            lst3.append(app3)
            lst4.append(app4)

    def analyze_and_plot(self, row, analyzer, verbose=False):
        fake_com = []
        fake_neu = []
        fake_pos = []
        fake_neg = []

        true_com = []
        true_neu = []
        true_pos = []
        true_neg = []


        for text in range(0, fake_df.shape[0], 1):
            if verbose:
                print(f'ANALYZING TEXT NUMBER {text}')

            fake_title = self.fake_df[row][text]
            #true_title = self.true_df[row][text]

            f_com, f_neu, f_pos, f_neg = self.get_sentiment(fake_title, analyzer)
            #t_com, t_neu, t_pos, t_neg = self.get_sentiment(true_title, analyzer)
            

            self.append_four(fake_com, fake_neu, fake_pos, fake_neg, f_com, f_neu, f_pos, f_neg)
            #self.append_four(true_com, true_neu, true_pos, true_neg, t_com, t_neu, t_pos, t_neg)

        
        fake_attributes = [fake_com, fake_neu, fake_pos, fake_neg]
        #true_attributes = [true_com, true_neu, true_pos, true_neg]
        col_names = ['com', 'neu', 'pos', 'neg']

        self.fake_df = self.add_columns(self.fake_df, fake_attributes, col_names)
        print(self.fake_df.head())

        fig, axs = plt.subplots(4,2, sharey='row', sharex='row')

        num_bins = 30

        #self.plotter(fake_attributes, true_attributes, axs, num_bins)

    def add_columns(self, df, attributes, cols):
        for col, rows in zip(cols, attributes):
            print(df.shape[0])
            print(len(rows))
            df[col] = rows
        return df



    def plotter(self, fake_attributes, true_attributes, axs, num_bins):

        titles = ['Compound', 'Neutrality', 'Positivity', 'Negativity']

        for ax, (fake_score, true_score, title) in enumerate(zip(fake_attributes, true_attributes, titles)):
            axs[ax, 0].hist(fake_score, bins=num_bins)
            axs[ax, 0].title.set_text(f"Fake News {title}")

            axs[ax, 1].hist(true_score, bins=num_bins)
            axs[ax, 1].title.set_text(f"True News {title}")

        plt.show()


if __name__ == "__main__":
    analyzer = SentimentIntensityAnalyzer()

    fake_df = pd.read_csv('data/Fake.csv')
    true_df = pd.read_csv('data/True.csv')

    print(fake_df.shape)
    print(true_df.shape)

    sa = sentiment_analysis(fake_df, true_df)

    subjects = ['News', 'politics', 'Government News', 'left-news', 'US_News', 'Middle-east']
    

    # sa.analyze_and_plot('title', analyzer, verbose=True)