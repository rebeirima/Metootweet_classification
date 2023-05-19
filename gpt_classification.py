# import the required packages and libraries
%matplotlib inline
import re
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import string
import nltk
import emoji
import contractions

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english')) - {'all'}

df = pd.read_csv("final_tweets.csv", encoding = "utf-8")

df_toxic = df.drop(['TweetId', 'text', 'Country'], axis=1)
counts = []
categories = list(df_toxic.columns.values)
for i in categories:
    counts.append((i, df_toxic[i].sum()))
df_stats = pd.DataFrame(counts, columns=['category', 'number_of_comments'])
df_stats
print('Number of missing comments in comment text:')
df.dropna(subset=['text'], inplace=True)
df = df[df['text'] != 'account suspended']
df['text'].isnull().sum()

df['text'][7]

#want to also get rid of stop words
stopwords = ["for", "on", "an", "a", "of", "and", "in", "the", "to", "from"]

# Gets the part of speech tag of word for lemmatization
# This function is based on code from:
#   https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_text(tweet):
    # Changes emojis to words
    tweet = emoji.demojize(tweet,  delimiters=(' ', ' '))
    # Removes contractions
    tweet = contractions.fix(tweet)
    # Removes 'RT' from tweet
    tweet = re.sub(r'RT[\s]+', '', tweet)
    # Removes capitalization
    tweet = tweet.lower()
    # Removes urls & user mentions from tweet
    tweet = re.sub(r"http\S+|www\S+|https\S+|\@\w+", ' ', tweet, flags=re.MULTILINE)
    # Removes punctuation
    tweet = re.sub(r'\p{P}+', '', tweet)
    # Removes stopwords
    #tokens = [w for w in word_tokenize(tweet) if not w in stop_words]
    # Perfoms lemmatization on tokens
    #lemmatizer = WordNetLemmatizer()
    #lemma_words = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in tokens]
    return tweet


def clean_text(text):
    text = text.lower()
    text = re.sub("'", "", text) # to avoid removing contractions in english
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub("@[A-Za-z0-9_]+","", text)
    text = re.sub("#[A-Za-z0-9_]+","", text) #other than metoo
    text = re.sub(r'http\S+', '', text)
    text = re.sub('[()!?“”‘’\'"]', ' ', text)
    text = re.sub('[()!?]', ' ', text)
    text = re.sub('\[.*?\]',' ', text)
    text = re.sub("[^a-z0-9]"," ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub("'", "", text) # to avoid removing contractions in english
    text = text.split()
    text = [w for w in text if not w in stopwords]
    text = " ".join(word for word in text)
    return text

df['text'] = df['text'].map(lambda com : preprocess_text(com))

print(df['text'][3])
df.head()

categories = ['Directed_Hate', 'Generalized_Hate', 'Sarcasm', 'Allegation', 'Justification', 'Support', 'Refutation', 'Oppose']
df = df.drop(['Country', 'TweetId'], axis=1)
df["Hate"] = df['Directed_Hate']+ df["Generalized_Hate"]
df = df.drop(['Directed_Hate', 'Generalized_Hate'], axis=1)

# move the 'text' column to the end
df = df[['Sarcasm', 'Allegation', 'Justification', 'Refutation', 'Support', 'Oppose', 'Hate', 'text']]
df.head()


df_1 = df[['text', 'Oppose']].copy()
df_1 = df_1.rename(columns={'text': 'prompt', 'Oppose': 'completion'})
#df_1['completion'] = df_1['completion'].replace({0.0: "0", 1.0: "1"})
#df_1.head()

df_1.to_csv("oppose.cvs")

df_2 = df[['text', 'Support']].copy()
df_2 = df_2.rename(columns={'text': 'prompt', 'Support': 'completion'})
df_2['completion'] = df_2['completion'].replace({0.0: "0", 1.0: "1"})
df_2.head()
df_2.to_csv("support.csv")

df_2 = df[['text', 'Refutation']].copy()
df_2 = df_2.rename(columns={'text': 'prompt', 'Refutation': 'completion'})
df_2['completion'] = df_2['completion'].replace({0.0: "0", 1.0: "1"})
df_2.head()
df_2.to_csv("Refutation_tweets.csv")

df_2 = df[['text', 'Justification']].copy()
df_2 = df_2.rename(columns={'text': 'prompt', 'Justification': 'completion'})
df_2['completion'] = df_2['completion'].replace({0.0: "0", 1.0: "1"})
df_2.head()
df_2.to_csv("justification_tweets.csv")

df_2 = df[['text', 'Allegation']].copy()
df_2 = df_2.rename(columns={'text': 'prompt', 'Allegation': 'completion'})
df_2['completion'] = df_2['completion'].replace({0.0: "0", 1.0: "1"})
df_2.head()
df_2.to_csv("Allegation_tweets.csv")

df_2 = df[['text', 'Hate']].copy()
df_2 = df_2.rename(columns={'text': 'prompt', 'Hate': 'completion'})
df_2['completion'] = df_2['completion'].replace({0.0: "0", 1.0: "1"})
df_2.head()
df_2.to_csv("Hate_tweets.csv")

df_2 = df[['text', 'Sarcasm']].copy()
df_2 = df_2.rename(columns={'text': 'prompt', 'Sarcasm': 'completion'})
df_2['completion'] = df_2['completion'].replace({0.0: "0", 1.0: "1"})
df_2.head()
df_2.to_csv("Sarcasm_tweets.csv")