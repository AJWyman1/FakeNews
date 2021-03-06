from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import sys

plt.style.use('ggplot')

def demo_vader(sentence):
    '''
    Demo VADER sentiment analysis
    '''
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(sentence)
    print("{:-<40} {}".format(sentence, str(score)))

class sentiment_analysis(object):

    def __init__(self, fake_df, true_df, is_vader=True):
        self.fake_df = fake_df
        self.true_df = true_df
        self.is_vader = is_vader

    def get_sentiment(self, title, analyzer):
        vs = analyzer.polarity_scores(title)
        
        com = vs['compound']
        neu = vs['neu']
        pos = vs['pos']
        neg = vs['neg']
        
        return com, neu, pos, neg

    def append_four(self, lst1, lst2, lst3, lst4, app1, app2, app3, app4, remove_defaults=False):
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

    def append_and_save_df(self, df, path, row, analyzer, verbose=False):
        com, neu, pos, neg = self.analyze_df(df, row, analyzer, verbose=verbose)
        # breakpoint()
        df['com_title'] = com
        df['neu_title'] = neu
        df['pos_title'] = pos
        df['neg_title'] = neg

        df.to_csv(path)

    def blob_blob(self, df):
        blob = TextBlob
        
        df['polarity'] = df.apply(lambda row: TextBlob(row['text']).sentiment.polarity, axis=1)
        df['subjectivity'] = df.apply(lambda row: TextBlob(row['title']).sentiment.subjectivity, axis=1)

        return df

    def analyze_df(self, df, row, analyzer, verbose=False):
        com = []
        neu = []
        pos = []
        neg = []

        for text in range(0, df.shape[0]):
                if verbose:
                    print(f'\rANALYZING TEXT NUMBER {text}')

                title = df[row][text]
                
                _com, _neu, _pos, _neg = self.get_sentiment(title, analyzer)
                # breakpoint()
                self.append_four(com, neu, pos, neg, _com, _neu, _pos, _neg)

        return com, neu, pos, neg

    def analyze_and_plot(self, row, analyzer, verbose=False):
        if self.is_vader:
            fake_com = []
            fake_neu = []
            fake_pos = []
            fake_neg = []

            true_com = []
            true_neu = []
            true_pos = []
            true_neg = []


            for text in range(0, 20000, 1):
                if verbose:
                    print(f'\rANALYZING TEXT NUMBER {text}')

                fake_title = self.fake_df[row][text]
                true_title = self.true_df[row][text]

                f_com, f_neu, f_pos, f_neg = self.get_sentiment(fake_title, analyzer)
                t_com, t_neu, t_pos, t_neg = self.get_sentiment(true_title, analyzer)
                

                self.append_four(fake_com, fake_neu, fake_pos, fake_neg, f_com, f_neu, f_pos, f_neg)
                self.append_four(true_com, true_neu, true_pos, true_neg, t_com, t_neu, t_pos, t_neg)
                sys.stdout.flush()

            
            fake_attributes = [fake_com, fake_neu, fake_pos, fake_neg]
            true_attributes = [true_com, true_neu, true_pos, true_neg]
            col_names = ['com', 'neu', 'pos', 'neg']

            #self.fake_df = self.add_columns(self.fake_df, fake_attributes, col_names)
            print(self.fake_df.head())

            fig, axs = plt.subplots(4,2, sharey=True, sharex='row')

            num_bins = 30

            self.plotter(fake_attributes, true_attributes, axs, num_bins)

    def add_columns(self, df, attributes, cols):
        for col, rows in zip(cols, attributes):
            print(df.shape[0])
            print(len(rows))
            df[col] = rows
        return df

    def boot_strap(self, df):        
        return np.random.choice(df.shape[0])

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

    fake_df = pd.read_csv('data/Fake_sentiment.csv')
    true_df = pd.read_csv('data/True_sentiment.csv')

    print(fake_df.shape)
    print(true_df.shape)

    sa = sentiment_analysis(fake_df, true_df)

    subjects = ['News', 'politics', 'Government News', 'left-news', 'US_News', 'Middle-east']

    fake_df = sa.blob_blob(fake_df)
    true_df = sa.blob_blob(true_df)

    print(fake_df.describe())
    print(true_df.describe())
    

    # sa.analyze_and_plot('text', analyzer, verbose=True)

    # # sa.append_and_save_df(df_test, 'data/test_sentiment.csv', 'col1', analyzer, verbose=True)

    # sa.append_and_save_df(fake_df, 'data/Fake_title_text_sentiment.csv', 'title', analyzer, verbose=True)
    # sa.append_and_save_df(true_df, 'data/True_title_text_sentiment.csv', 'title', analyzer, verbose=True)