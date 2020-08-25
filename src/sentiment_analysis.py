from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

plt.style.use('ggplot')

class sentiment_analysis(object):

    def __init__(self):
        pass

    def three_d_plotter(self, ax, text, vs):
        z2 = vs['neu']
        x2 = vs['neg']
        y2 = vs['pos']
        z = vs['neu']
        x = vs['neg']
        y = vs['pos']
        ax.scatter3D(x, y, z, color='m')
        ax.scatter3D(x2, y2, z2, color='g')
        zdata.append(vs['neg'])
        xdata.append(vs['neu'])
        ydata.append(vs['pos'])
        ax.scatter(vs['neg'], vs['pos'], color=subject_color[fake_df['subject'][text]])


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
            if app3 < .98:
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


        for text in range(0, 20000, 1):
            if verbose:
                print(f'ANALYZING TEXT NUMBER {text}')

            fake_title = fake_df[row][text]
            true_title = true_df[row][text]

            f_com, f_neu, f_pos, f_neg = self.get_sentiment(fake_title, analyzer)
            t_com, t_neu, t_pos, t_neg = self.get_sentiment(true_title, analyzer)

            self.append_four(fake_com, fake_neu, fake_pos, fake_neg, f_com, f_neu, f_pos, f_neg)
            self.append_four(true_com, true_neu, true_pos, true_neg, t_com, t_neu, t_pos, t_neg)

        
        fake_attributes = [fake_com, fake_neu, fake_pos, fake_neg]
        true_attributes = [true_com, true_neu, true_pos, true_neg]

        fig, axs = plt.subplots(4,2, sharey='row', sharex='row')

        num_bins = 30

        self.plotter(fake_attributes, true_attributes, axs, num_bins)


    def plotter(self, fake_attributes, true_attributes, axs, num_bins):

        titles = ['Compound', 'Neutrality', 'Positivity', 'Negativity']

        for ax, (fake_score, true_score, title) in enumerate(zip(fake_attributes, true_attributes, titles)):
            axs[ax, 0].hist(fake_score, bins=num_bins)
            axs[ax, 0].title.set_text(f"Fake News {title}")

            axs[ax, 1].hist(true_score, bins=num_bins)
            axs[ax, 1].title.set_text(f"True News {title}")

        plt.show()

if __name__ == "__main__":
    sa = sentiment_analysis()
    analyzer = SentimentIntensityAnalyzer()
    fake_df = pd.read_csv('data/Fake.csv')
    true_df = pd.read_csv('data/True.csv')
    zdata = []
    xdata = []
    ydata = []

    # ax = plt.axes(projection='3d')
    # ax = plt.axes()


    subjects = ['News', 'politics', 'Government News', 'left-news', 'US_News', 'Middle-east']
    subject_color = {'News': 'b', 'politics':'g', 'Government News':'r', 'left-news':'m', 'US_News':'y', 'Middle-east':'k'}
    fake_com = []
    fake_neu = []
    true_com = []
    true_neu = []

    sa.analyze_and_plot('title', analyzer, verbose=True)

    # fig, axs = plt.subplots

    # for text in range(0, 20000, 1):
    #     print(f'ANALYZING TEXT NUMBER {text}')
    #     vs = analyzer.polarity_scores(fake_df['title'][text])
    #     z = vs['neu']
    #     x = vs['neg']
    #     y = vs['pos']
    #     fake_neu.append(vs['neu'])
    #     if vs['compound']:
    #         fake_com.append(vs['compound'])

    #     vs = analyzer.polarity_scores(true_df['title'][text])
    #     z2 = vs['neu']
    #     x2 = vs['neg']
    #     y2 = vs['pos']
    #     if vs['compound']:
    #         true_com.append(vs['compound'])
    #     true_neu.append(vs['neu'])
        

    # print(f'FAKE NEWS AVG COMPOUND SCORE: {np.mean(fake_com)}')
    # print(f'TRUE NEWS AVG COMPUND SCORE: {np.mean(true_com)}')

    # print(f'FAKE NEWS AVG neu SCORE: {np.mean(fake_neu)}')
    # print(f'TRUE NEWS AVG neu SCORE: {np.mean(true_neu)}')


    # fig, axs = plt.subplots(1,2, sharey=True, figsize=(15,8))

    # axs[0].hist(fake_neu, bins=90)
    # axs[1].hist(true_neu, bins=90)
    # axs[0].title.set_text(f" {len(fake_neu)} Fake News Titles")
    # axs[1].title.set_text(f" {len(true_neu)} True News Titles")

    # plt.show()
    
    # ax.scatter3D(xdata, ydata, zdata)
    # ax.set_xticks(np.linspace(0,1,5))
    # ax.set_yticks(np.linspace(0,1,5))
    # ax.set_zticks(np.linspace(0,1,5))
    # ax.set_zlabel('Neutrality Score')
    # ax.set_ylabel('Positivity Score')
    # ax.set_xlabel('Negativity Score')
    # plt.show()