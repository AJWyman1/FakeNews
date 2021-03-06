from bs4 import BeautifulSoup
from bs4.element import Comment
from urllib.request import urlopen
import requests

class scraper():

    def __init__(self):
        pass

    def tag_visible(element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        return True

    def text_from_html(body):
        soup = BeautifulSoup(body, 'html.parser')
        texts = soup.findAll(text=True)
        visible_texts = filter(tag_visible, texts)  
        return u" ".join(t.strip() for t in visible_texts)
    def abc_article(self, html):
        '''
        input: response html from abc

        output: Article body text
        '''
        soup2 = BeautifulSoup(html, 'html.parser')
        article = soup2.find('article', class_='Article__Content story')
        article_text = article.get_text()
        return article_text

    def bbart_article(self, html):
        '''
        input: response html from abc

        output: Article body text
        '''
        soup2 = BeautifulSoup(html, 'html.parser')
        article = soup2.find(class_='entry-content')
        article_text = article.get_text()
        return article_text

    def daily_wire(self, html):
        soup2 = BeautifulSoup(html, 'html.parser')
        article = soup2.find(id="post-body-text")
        print(article)
        print(article.get_text())
        print(type(article))
        pass

    def basic_article(self, html):
        pass

if __name__ == "__main__":
    url = "https://www.cnn.com/2020/09/13/media/donald-trump-rally-coronavirus-safety-reliable-sources/index.html"
    url_d = "https://www.dailywire.com/news/larry-elder-thanks-blm-obama-dnc-cnn-msnbc-sharpton-after-la-cops-are-shot-james-woods-to-elder-thank-you"
    url_a = "https://abcnews.go.com/US/wireStory/south-dakota-ag-drinking-fatal-crash-73001889"
    url_h = "https://www.haaretz.com/israel-news/.premium-israel-could-have-made-an-effort-to-avoid-a-lockdown-1.9155934"
    url_a2 = "https://abcnews.go.com/US/settlement-reached-fatal-kentucky-police-shooting-breonna-taylor/story?id=73019106&cid=clicksource_4380645_1_heads_hero_live_hero_hed"
    url_d2 = "https://www.dailywire.com/news/violent-riots-looting-hit-pennsylvania-after-cop-shoots-black-man-suspect-charged-at-cop-with-knife-video-shows"
    url_ny = "https://www.nytimes.com/2020/09/15/books/booker-prize-shortlist.html"
    url_b = "https://www.breitbart.com/politics/2020/09/17/soros-backed-coalition-preparing-for-post-election-day-chaos-were-going-to-fight-like-hell/"
    url_d3 = "https://www.dailywire.com/news/biden-blasted-for-beyond-patronizing-incident-trying-to-appeal-to-latinos-by-playing-spanish-sex-song"
    reut = "https://www.reuters.com/article/us-health-coronavirus-usa-farmers-aid/u-s-details-up-to-14-billion-in-new-aid-for-farmers-idUSKBN269261"
    r  = requests.get(url_b)
    response_html = r.text

    s = scraper()
    text = s.bbart_article(response_html)
    print(text)
    text = s.text_from_html(response_html)

    # print(s.bbart_article(response_html))

    # print(text)

    # print(text)

    # title = s.get_title(text)
    # print(title)
