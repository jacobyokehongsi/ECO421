
import pandas as pd
import numpy as np
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models.phrases import Phrases, Phraser
from gensim.corpora import Dictionary
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def main():
    data = pd.read_csv('NYT_headlines.csv')
    data.drop_duplicates()
    # Preprocessing
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

    clean_doc = [clean(t, 'lemma').split() for t in data['Headlines']]
    clean_doc2 = [clean(t, 'porter').split() for t in data['Headlines']]

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

    var = [(dictionary[word], freq) for word, freq in bow_corpus[1][:10]]

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


    # Combine headlines by day ??????????????????
    
    ######################################
    
    
    # ----------------------------------------
    # LDA Model with lemmatized text
    # ----------------------------------------

    from gensim.models import LdaModel

    K = 15
    lda_model = LdaModel(corpus=bow_corpus, id2word=dictionary, chunksize=1700,
                         random_state=42, iterations=50, num_topics=K, passes=20)

    for topic_id, topic in lda_model.print_topics(num_topics=10, num_words=20):
        print('topic #' + str(topic_id + 1) + ':')
        print(topic)
        print('-----------------------------------')
        print('')

    # Plot topics


    for t in range(K):
        plt.figure()
        plt.imshow(WordCloud().fit_words(dict(lda_model.show_topic(t, 200))))
        plt.axis('off')
        plt.title('Topic # ' + str(t))

    topics = ['2016 Election', 'Politics', 'Trump', 'Health', 'Law/Police', 'German', 'Science',
              'Spanish', 'China', 'Family', 'Foreign Affairs', 'Business', 'People?', 'Russia',
              'Clinton email']

    # -------------------------------
    # LDA Model now with porter
    # -------------------------------

    lda_model2 = LdaModel(corpus=bow_corpus2, id2word=dictionary2, chunksize=1700,
                          random_state=42, iterations=50, num_topics=K, passes=20)

    # Plot topics

    for t in range(K):
        plt.figure()
        plt.imshow(WordCloud().fit_words(dict(lda_model2.show_topic(t, 200))))
        plt.axis('off')
        plt.title('Topic # ' + str(t))

    topics2 = ['Generic', 'Foreign Politics', 'Police', 'Trump Election', 'Trump-Clinton',
               'Sports', 'Clinton email', 'Business', 'Protest', 'Business', 'Foreign language',
               'Science/Health', 'Work', 'Middle East', 'Russia']


    return










#--------------------------------------
# Measure topic coherence of both
# (Allows us to see which does 'better')
#---------------------------------------

from gensim.models import CoherenceModel
cv_coherence1=CoherenceModel(model=lda_model,corpus=bow_corpus,
                                      texts=norm_corpus_trigrams,
                                      dictionary=dictionary,coherence='c_v')

avg_coherence=cv_coherence1.get_coherence()

cv_coherence2=CoherenceModel(model=lda_model2,corpus=bow_corpus2,
                             texts=norm_corpus_trigrams2,
                             dictionary=dictionary2,coherence='c_v')
avg_coherence2=cv_coherence2.get_coherence()

# Let's use porter text...

#-----------------------------------
# Topic distribution by fake/real
#-----------------------------------

fake_news=np.array(clean_doc2)[data['label']==1]
real_news=np.array(clean_doc2)[data['label']==0]

fake_news_comb=[word for doc in fake_news for word in doc]
real_news_comb=[word for doc in real_news for word in doc]

bow_corpus_fake=dictionary2.doc2bow(trigram_model2[bigram_model2[fake_news_comb]])
bow_corpus_real=dictionary2.doc2bow(trigram_model2[bigram_model2[real_news_comb]])

topics_fake=lda_model2[bow_corpus_fake]
topics_real=lda_model2[bow_corpus_real]

topics2=np.array(topics2)

topics_fake=pd.DataFrame(topics_fake).sort_values(by=1,ascending=False)
topics_fake['name']=topics2[np.array(topics_fake[0])]

topics_real=pd.DataFrame(topics_real).sort_values(by=1,ascending=False)
topics_real['name']=topics2[np.array(topics_real[0])]

plt.figure(figsize=(15,5))
plt.bar(topics_fake['name'],topics_fake[1])

plt.figure(figsize=(15,5))
plt.bar(topics_real['name'],topics_real[1])

#----------------------------------
# Classifiers
#----------------------------------

# Transform BOW-corpus into usable data

from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer()

X=vectorizer.fit_transform([" ".join(text) for text in norm_corpus_trigrams2])
X2=X.toarray()

X2=pd.DataFrame(X2,columns=vectorizer.get_feature_names())

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

logit=LogisticRegression()
dt=DecisionTreeClassifier()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X2,data['label'])

dt.fit(x_train,y_train)

dt.score(x_test,y_test)

imp=pd.DataFrame({'feature':X2.columns,'importance':dt.feature_importances_}).sort_values(by='importance',ascending=False)


#---------------------------------
# Vader Lexicon Stuff
#---------------------------------

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

fake_news_joined=" ".join(fake_news_comb)
real_news_joined=" ".join(real_news_comb)

analyzer=SentimentIntensityAnalyzer()

analyzer.polarity_scores(fake_news_joined)



if __name__ == "__main__":
    main()