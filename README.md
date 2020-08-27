# Fake News

### INTRO
        fake news and problem intro

### Introduce the data set

Dataset came in two separate `csv` files one containing only fake news and the other with true news. 

## Word Clouds & EDA
---
### Fake news word cloud from title
![](img/fake_news_wordcloud.png)
### True news word cloud from title
![](img/true_news_wordcloud.png)

### Exploring the test with VADER sentiment analysis 

![](img/all_four_with_zeros.png)

## TFIDF

Stop words added `'reuters' '21st', 'century', 'wire', '21wire', 'www', 'https', 'com', 'pic'`

### Naive Bayes
--- 

                  with n_grams = 1

                    0.930 Accuracy

Word Rank | Fake News | True News
---------|----------|---------
 1 | Trump | said
 2 | Clinton | Trump
 3 | Obama | president
 4 | people | state
 5 | president | house
 6 | Hillary | government



Word Rank | Fake News | True News
---------|----------|---------
 7 | just | Washington
 8 | said | republican
 9 | like | united
 10 | Donald | states
 11 | twitter | north
 12 | news | new

 ---

                         n_grams = 2


                        0.953 Accuracy

Word Rank | Fake News | True News
---------|----------|---------
 1 | Donald Trump | United States
 2 | Hillary Clinton | North Korea
 3 | featured image | white house
 4 | white house | Donald Trump
 5 | president Trump | president Donald
 6 | president Obama | prime minister

 Word Rank | Fake News | True News
---------|----------|---------
 7 | United States | said statement
 8 | getty images | Islamic state
 9 | fox news | told reporters 
 10 | New York| Trump said
 11 | 21st century| New York
 12 | century wire| Washington President


                    n_grams = 3
                    0.971 Accuracy

Word Rank | Fake News | True News
---------|----------|---------
 1 | Trump | said
 2 | Clinton | Trump
 3 | Obama | president
 4 | people | state
 5 | president | house
 6 | Hillary | government



Word Rank | Fake News | True News
---------|----------|---------
 7 | just | Washington
 8 | said | republican
 9 | like | united
 10 | Donald | states
 11 | twitter | north
 12 | news | new

### Random Forest
---

                    0.979 Accuracy

 | Feature      | Feature Importances
----------------|-----------------
 | said         | 0.042
 | washington   | 0.012
 | featured     | .010
 | image        | 0.010
 | minister     | 0.007