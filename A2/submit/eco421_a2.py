# -*- coding: utf-8 -*-
"""ECO421 A2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-gLy92xXbAfXvPFSoAiWc3wM3Ax6oKOw
"""

# from google.colab import files
# uploaded = files.upload()

import pandas as pd
import numpy as np
import string
import nltk
import json
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.download('wordnet')
from gensim.models import CoherenceModel
from gensim.models import LdaModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# !pip install vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime


def main_headlines():
    # Q1.
    data = pd.read_csv('NYT_headlines.csv')
    data.drop_duplicates()

    for index, row in data.iterrows():
        if 'Feb.' in row['date']:
            row['date'] = datetime.strptime(row['date'], "%b. %d").replace(year=2021).date()
            data.loc[index, 'date'] = row['date']
        else:
            row['date'] = datetime.strptime(row['date'], "%B %d").replace(year=2021).date()
            data.loc[index, 'date'] = row['date']

            # Q2.
    covid_words = ['Pandemic', 'Coronavirus', 'Vaccine', 'Cases', 'Distancing',
                   'Quarantine', 'Covid-19']

    # Preprocessing

    # Q3.
    # Combine headlines by day
    comb_headlines = data.groupby('date')['Headlines'].apply(' '.join).reset_index()

    # Q4.
    # Stopwords
    stop = STOPWORDS
    # stop=stop.union(['said','im','person'])

    # Punctuation
    punct = set(string.punctuation)
    punct.update(['"', '``'])

    # stemmer/lemmatizer

    lemma = WordNetLemmatizer()
    ps = PorterStemmer()

    # Function, input is string, output is clean string
    def clean(doc, method):
        numb_free = ''.join([i for i in doc if not i.isdigit()])
        punc_free = ''.join([i for i in numb_free if i not in punct])
        stop_free = " ".join([i for i in punc_free.lower().split() if i not in stop])
        if method == 'lemma':
            normalize = [lemma.lemmatize(word) for word in stop_free.split()]
        elif method == 'porter':
            normalize = [ps.stem(word) for word in stop_free.split()]
        else:
            print('Only "lemma" or "porter" available')
            return
        final = " ".join([word for word in normalize if len(word) != 1])
        return (final)

    # Clean each article, and return as *list* of clean words
    # (this is why we use .split())

    clean_doc = [clean(t, 'lemma').split() for t in data['Headlines']]  ########################
    clean_doc2 = [clean(t, 'porter').split() for t in data['Headlines']]  ###########################

    # -----------------------------------
    # Bigrams-trigrams with lemmatizer
    # -----------------------------------

    bigram = Phrases(clean_doc, min_count=5, threshold=20)  # Generates a model
    trigram = Phrases(bigram[clean_doc], min_count=5, threshold=20)

    bigram_model = Phraser(bigram)  # Stores model in an easier to use way
    trigram_model = Phraser(trigram)

    # Find all trigrams based on model above
    norm_corpus_trigrams = [trigram_model[bigram_model[doc]] for doc in clean_doc]

    # The list of all unique words (including trigrams) in the corpus
    dictionary = Dictionary(norm_corpus_trigrams)

    dictionary.filter_extremes(no_below=20, no_above=0.99)

    # Looking only at words in dictionary, create bag-of-words
    # for each article
    bow_corpus = [dictionary.doc2bow(trigram_model[bigram_model[doc]]) for doc in clean_doc]

    # ----------------------------
    # Same, but use porter text!
    # -----------------------------

    bigram2 = Phrases(clean_doc2, min_count=5, threshold=20)
    trigram2 = Phrases(bigram2[clean_doc2], min_count=5, threshold=20)

    bigram_model2 = Phraser(bigram2)
    trigram_model2 = Phraser(trigram2)

    norm_corpus_trigrams2 = [trigram_model2[bigram_model2[doc]] for doc in clean_doc2]

    dictionary2 = Dictionary(norm_corpus_trigrams2)

    dictionary2.filter_extremes(no_below=20, no_above=0.9)

    bow_corpus2 = [dictionary2.doc2bow(trigram_model2[bigram_model2[doc]]) for doc in clean_doc2]

    # Finding optimal K

    coherence_lst = []
    model_lst = []
    coherence_lst2 = []
    model_lst2 = []
    for K in range(1, 15, 1):
        # lemmatized
        lda_model = LdaModel(corpus=bow_corpus, id2word=dictionary, chunksize=1700,
                             random_state=42, iterations=50, num_topics=K, passes=20)
        cv_coherence = CoherenceModel(model=lda_model, corpus=bow_corpus,
                                      texts=norm_corpus_trigrams,
                                      dictionary=dictionary, coherence='c_v')

        avg_coherence = cv_coherence.get_coherence()
        coherence_lst.append(avg_coherence)

        # porter
        lda_model2 = LdaModel(corpus=bow_corpus2, id2word=dictionary2, chunksize=1700,
                              random_state=42, iterations=50, num_topics=K, passes=20)

        cv_coherence2 = CoherenceModel(model=lda_model2, corpus=bow_corpus2,
                                       texts=norm_corpus_trigrams2,
                                       dictionary=dictionary2, coherence='c_v')
        avg_coherence2 = cv_coherence2.get_coherence()

        model_lst.append(K)
        coherence_lst2.append(avg_coherence2)
        # model_lst2.append(K)

    # limit=15; start=1; step=1
    # x = range(start, limit, step)
    plt.figure(1)
    plt.plot(model_lst, coherence_lst)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence Score lemmatized")

    plt.figure(2)
    plt.plot(model_lst, coherence_lst2)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence Score porter")

    plt.show()

    ##### optimal K happens to be 4 for lemma and 6 for porter #####

    # ----------------------------------------
    # LDA Model with lemmatized text
    # ----------------------------------------
    K_lemmatized = 4
    lda_model = LdaModel(corpus=bow_corpus, id2word=dictionary, chunksize=1700,
                         random_state=42, iterations=50, num_topics=K_lemmatized, passes=20)

    # # Plot topics

    for t in range(K_lemmatized):
        plt.figure()
        plt.imshow(WordCloud().fit_words(dict(lda_model.show_topic(t, 200))))
        plt.axis('off')
        plt.title('Topic # ' + str(t))

    topics = ['Pandemic', 'Politics', '2016 Election', 'Vaccination']

    # -------------------------------
    # LDA Model now with porter
    # -------------------------------

    K_porter = 6
    lda_model2 = LdaModel(corpus=bow_corpus2, id2word=dictionary2, chunksize=1700,
                          random_state=42, iterations=50, num_topics=K_porter, passes=20)

    # Plot topics

    for t in range(K_porter):
        plt.figure()
        plt.imshow(WordCloud().fit_words(dict(lda_model2.show_topic(t, 200))))
        plt.axis('off')
        plt.title('Topic # ' + str(t))

    topics2 = ['Presidential Election', 'Republican Trump', 'Protest',
               'Democratic Biden', 'Vaccination and Trump\'s impeachment']

    # --------------------------------------
    # Measure topic coherence of both
    # (Allows us to see which does 'better')
    # ---------------------------------------
    cv_coherence1 = CoherenceModel(model=lda_model, corpus=bow_corpus,
                                   texts=norm_corpus_trigrams,
                                   dictionary=dictionary, coherence='c_v')
    avg_coherence = cv_coherence1.get_coherence()
    cv_coherence2 = CoherenceModel(model=lda_model2, corpus=bow_corpus2,
                                   texts=norm_corpus_trigrams2,
                                   dictionary=dictionary2, coherence='c_v')
    avg_coherence2 = cv_coherence2.get_coherence()

    print('avg_coherence lemmatized', avg_coherence)
    print('avg_coherence porter', avg_coherence2)

    # avg_coherence lemmatized 0.6376855111025308
    # avg_coherence porter 0.6083734758195751
    # therefore, lemmatized performs better

    # Q5.
    covid_related = data[data['Headlines'].str.contains('|'.join(covid_words))]

    covid_related_count = covid_related["date"].value_counts()
    total_articles_c = data["date"].value_counts()

    covid_uncertainty_index = pd.DataFrame()
    for idx, d in enumerate(covid_related_count.index.tolist()):
        for idx2, d2 in enumerate(total_articles_c.index.tolist()):
            if d == d2:
                ratio = covid_related_count[idx] / total_articles_c[idx]
                a = pd.DataFrame({"date": [d], "ratio": [ratio]})
                covid_uncertainty_index = covid_uncertainty_index.append(a)
                covid_uncertainty_index_reset = covid_uncertainty_index.reset_index()
    print("Covid Uncertainty Index:\n", covid_uncertainty_index_reset)

    # Q6.
    economic_policy_words = ["uncertainty", "uncertain", "economics", "economy",
                             "Congress", "deficit", "federal reserve", "legislation",
                             "regulation", "white house", "uncertainties",
                             "regulatory", "the fed"]

    economic_related = data[data['Headlines'].str.contains('|'.join(economic_policy_words))]
    economic_related_count = economic_related["date"].value_counts()
    total_articles_e = data["date"].value_counts()

    coarse_economic_policy_uncertainty_index = pd.DataFrame()
    for idx, d in enumerate(economic_related_count.index.tolist()):
        for idx2, d2 in enumerate(total_articles_e.index.tolist()):
            if d == d2:
                ratio = economic_related_count[idx] / total_articles_e[idx]
                a = pd.DataFrame({"date": [d], "ratio": [ratio]})
                coarse_economic_policy_uncertainty_index = coarse_economic_policy_uncertainty_index.append(a)
                coarse_economic_policy_uncertainty_index_reset = coarse_economic_policy_uncertainty_index.reset_index()
    print("Coarse Economic Policy Uncertainty Index:\n", coarse_economic_policy_uncertainty_index_reset)

    # #Q7.
    data_snp = pd.read_csv('SP500 (3).csv')
    data_snp_1 = data_snp[['Date', 'Adj Close**']]
    data_snp_1['Adj Close**'] = data_snp_1['Adj Close**'].str.replace(',', '').astype(float)
    data_snp_1['Adj Close**'].apply(pd.to_numeric)
    data_snp_1['returns'] = data_snp_1['Adj Close**'].pct_change()
    data_snp_1 = data_snp_1.rename(columns={"Date": "date"})
    for index, row in data_snp_1.iterrows():
        row['date'] = datetime.strptime(row['date'], "%d-%b-%y").date()
        data_snp_1.loc[index, 'date'] = row['date']
    print('SNP500 daily returns:\n', data_snp_1)

    # #Q8.
    covid_data = covid_uncertainty_index
    economic_data = coarse_economic_policy_uncertainty_index
    snp_data = data_snp_1

    covid_economic_na = pd.merge(covid_data, economic_data, how='outer', on='date')
    covid_economic = covid_economic_na.dropna()
    covid_snp_na = pd.merge(covid_data, snp_data, how='outer', on='date')
    covid_snp = covid_snp_na.dropna()
    economic_snp_na = pd.merge(economic_data, snp_data, how='outer', on='date')
    economic_snp = economic_snp_na.dropna()

    covid_economic.plot(x='ratio_x', y='ratio_y', style='o')
    covid_snp.plot(x='ratio', y='returns', style='o')
    economic_snp.plot(x='ratio', y='returns', style='o')

    # Q9.
    # (a)
    daily_covid_related = covid_related.groupby('date')['Headlines'].apply(' '.join).reset_index()
    analyzer = SentimentIntensityAnalyzer()

    daily_sentiment_index = pd.DataFrame()
    for i, r in daily_covid_related.iterrows():
        polarity = analyzer.polarity_scores(r["Headlines"])
        a = pd.DataFrame({"date": [r["date"]], "polarity_neg": [polarity['neg']], "polarity_neu": [polarity['neu']],
                          "polarity_pos": [polarity['pos']], "polarity_neg-pos": [polarity['neg'] - polarity['pos']]})
        daily_sentiment_index = daily_sentiment_index.append(a)
    print("Daily Sentiment Index:\n", daily_sentiment_index)

    daily_sentiment_index.plot(x='date', y='polarity_neg')
    daily_sentiment_index.plot(x='date', y='polarity_neu')
    daily_sentiment_index.plot(x='date', y='polarity_pos')
    daily_sentiment_index.plot(x='date', y='polarity_neg-pos')

    # (b)
    aggregate_sentiment_index = pd.DataFrame()
    polarity = analyzer.polarity_scores(daily_covid_related["Headlines"])
    b = pd.DataFrame(
        {"polarity_neg": [polarity['neg']], "polarity_neu": [polarity['neu']], "polarity_pos": [polarity['pos']],
         "polarity_neg-pos": [polarity['neg'] - polarity['pos']]})
    aggregate_sentiment_index = aggregate_sentiment_index.append(b)
    print("Aggregate Sentiment Index:\n", aggregate_sentiment_index)
    return


if __name__ == "__main__":
    main_headlines()
