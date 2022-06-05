#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import numpy as np
import pandas as pd
import seaborn as sns
import time
import string
import math

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import mean_squared_error as mse
from sklearn import linear_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize as wt
from nltk.stem import WordNetLemmatizer


# In[2]:


get_ipython().system('pip install transformers')


# In[3]:


get_ipython().system('pip install xgboost')


# In[5]:


get_ipython().system('pip install missingno')


# In[6]:


from transformers import BertTokenizer, BertConfig, TFBertModel
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.metrics import RootMeanSquaredError
import xgboost as xgb
import tensorflow as tf

stemmer = PorterStemmer()
from IPython.core.display import display
from scipy import stats
from scipy.spatial import distance
import matplotlib.pyplot as plt
import missingno as msno


# In[10]:


import string

import spacy


hrule = lambda x: "=" * x
stop_words = stopwords.words('english')


# In[8]:


get_ipython().system('pip install spacy')


# In[12]:


train = pd.read_csv('train111.csv')
test = pd.read_csv('test111.csv')
sample = pd.read_csv('sample_submission.csv')


# In[13]:


train.head()


# In[14]:


test.head()


# In[15]:


print(hrule(50))
print("Dataset shape:")
print(train.shape)
print(hrule(50))
train.info()
print(hrule(50))


# In[18]:


msno.bar(train,color='brown')
plt.show()


# In[19]:


pd.DataFrame(train.isnull().sum() / train.shape[0]).T


# In[20]:


subset = train[train['license'].isna() & (train['url_legal'].isna() == False) |
               (train['license'].isna() == False) & train['url_legal'].isna()]
print("Numero di occorrenze in cui url_legal è mancante e license è dato o viceversa: " + str(len(subset)))


# In[21]:


valoriMancanti = train.isnull().sum() / len(train)
valoriMancanti = valoriMancanti[valoriMancanti > 0]
print("Righe in cui mancano url_legal e license: " + str(int(valoriMancanti[0] * 100)) + "%")


# In[22]:


train.describe()


# In[23]:


test


# In[24]:


msno.bar(test,color='orange')
plt.show()


# In[25]:


subset = test[test['license'].isna() & (test['url_legal'].isna() == False) |
              (test['license'].isna() == False) & test['url_legal'].isna()]
print("Numero di occorrenze in cui url_legal è mancante e license è dato o viceversa: " + str(len(subset)))


# In[26]:


valoriMancanti = test.isnull().sum() / len(test)
valoriMancanti = valoriMancanti[valoriMancanti > 0]
print("Righe in cui mancano url_legal e license: " + str(int(valoriMancanti[0] * 100)) + "%")


# In[27]:


train.query('target == 0')


# In[28]:


we = stats.probplot(train.target, plot=plt)
plt.show()


# In[29]:


sns.histplot(train.target, kde=True, stat="density").set_title('Target distribution')
plt.show()


# In[30]:


display(train.sort_values(by=['target']).head())


# In[31]:


display(train.sort_values(by=['target'], ascending=False).head())


# In[32]:


count = train['excerpt'].str.split().str.len()
print( count)
print( max(count))


# In[33]:


train['excerpt_len'] = train['excerpt'].apply(
    lambda x: len(x)
)
train['excerpt_word_count'] = train['excerpt'].apply(
    lambda x: len(x.split(' '))
)


# In[34]:


sns.kdeplot(train['excerpt_len']).set_title("Excerpt Len distribution")
plt.show()


# In[35]:


sns.kdeplot(train['excerpt_word_count']).set_title("Excerpt word Count distribution")
plt.show()


# In[36]:


sns.jointplot(
    data=train[train.standard_error != 0],
    x="target",
    y="excerpt_len",
    kind="hex",
    height=8)
plt.suptitle("target vs excerpt_len", font="Serif", size=20)
plt.subplots_adjust(top=0.95)
plt.show()


# In[37]:


def uniqueWordCount(text):
    text = text.lower()
    words = text.split()
    words = [word.strip('.,!;()[]') for word in words]
    words = [word.replace("'s", '') for word in words]

    #finding unique
    unique = []
    for word in words:
        if word not in unique:
            unique.append(word)

    return len(unique)


uniqueWordCount("prova ciao Prova")
train['unique_word_count'] = train['excerpt'].apply(uniqueWordCount)
test['unique_word_count'] = test['excerpt'].apply(uniqueWordCount)


# In[38]:


sns.kdeplot(train['unique_word_count']).set_title("Unique word count distribution")
plt.show()


# In[39]:


sns.jointplot(
    data=train[train.standard_error != 0],
    x="target",
    y="unique_word_count",
    kind="hex",
    height=8)
plt.suptitle("target vs unique words", font="Serif", size=20)
plt.subplots_adjust(top=0.95)
plt.show()


# In[40]:


stats.probplot(train.standard_error[train.standard_error != 0], plot=plt)
plt.show()


# In[41]:


sns.histplot(train.standard_error).set_title('standard_error distribution')
plt.show()


# In[42]:


display(train.sort_values(by=['standard_error']).head())


# In[43]:


display(train.sort_values(by=['standard_error'], ascending=False).head())


# In[44]:


sns.jointplot(
    data=train[train.standard_error != 0],
    x="target",
    y="standard_error",
    kind="hex",
    height=8)
plt.suptitle("target vs standard_error", font="Serif", size=20)
plt.subplots_adjust(top=0.95)
plt.show()


# # Data cleaning

# In[45]:


def rimuoviStopWords(testo):
    tokenized_text = wt(testo)
    sms_processed = []
    for word in tokenized_text:
        if word not in set(stopwords.words('english')):
            sms_processed.append(word)

    clean_text = " ".join(sms_processed)

    return clean_text


print("Testo prima della funzione rimuoviStopWords:\n" + train['excerpt'][1])
print(hrule(20))
print("Testo dopo la funzione:\n" + rimuoviStopWords(train['excerpt'][1]))


# In[46]:


def rimuoviPunteggiatura(testo):
    return testo.translate(str.maketrans('', '', string.punctuation))


print("Testo prima della funzione rimuoviPunteggiatura:\n" + train['excerpt'][1])
print(hrule(20))
print("Testo dopo la funzione:\n" + rimuoviPunteggiatura(train['excerpt'][1]))


# In[47]:


def rimuoviLink(testo):
    clean_text = re.sub('https?://\S+|www\.\S+', '', testo)
    return clean_text


test_string = "Il link http://www.google.com/ andrebbe rimosso"
print(test_string)
print(hrule(20))
print(rimuoviLink(test_string))


# In[48]:


def rimuoviNumeri(testo):
    clean_text = re.sub(r'\d+', '', testo)
    return clean_text


test_string = "Il numero 34 va rimosso"
print(test_string)
print(hrule(20))
print(rimuoviNumeri(test_string))


# In[49]:


def clean(testo):
    testo = testo.lower()  #Lets make it lowercase
    testo = rimuoviStopWords(testo)
    testo = rimuoviPunteggiatura(testo)
    testo = rimuoviNumeri(testo)
    testo = rimuoviLink(testo)
    return testo


train['excerpt_clean'] = train['excerpt'].apply(clean)
test['excerpt_clean'] = test['excerpt'].apply(clean)

train.head()


# # Stemming

# In[50]:


stemmer = SnowballStemmer(language='english')

tokens = train['excerpt'][1].split()
clean_text = ' '

for token in tokens:
    print(token + ' --> ' + stemmer.stem(token))


# In[51]:


def stemWord(text):
    stemmer = SnowballStemmer(language='english')
    tokens = text.split()
    clean_text = ' '
    for token in tokens:
        clean_text = clean_text + " " + stemmer.stem(token)
    return clean_text


print("Testo prima della funzione stemWord: " + train['excerpt'][1])
print("Testo dopo la funzione: " + stemWord(train['excerpt'][1]))


# In[52]:


train['excerpt_clean'] = train['excerpt_clean'].apply(stemWord)
test['excerpt_clean'] = test['excerpt_clean'].apply(stemWord)


# #Bag of words e TF-IDF

# In[55]:


cv = CountVectorizer(stop_words='english')
tv = TfidfVectorizer(stop_words='english')

a = "Data mining is a beautiful subject, I like it!"
b = "Data mining is the best subject"

cv_score = cv.fit_transform([a, b])
tv_score = tv.fit_transform([a, b])

def matrix_to_list(matrix):
    matrix = matrix.toarray()
    return matrix.tolist()

cv_score_list = matrix_to_list(cv_score)
tv_score_list = matrix_to_list(tv_score)

print("tfidf_a  tfidf_b  count_a count_b   word")
print("-"*41)
for i in range(6):
    print("  {:.3f}    {:.3f}        {:}       {:}   {:}".format(tv_score_list[0][i],
                                               tv_score_list[1][i],
                                               cv_score_list[0][i],
                                               cv_score_list[1][i],
                                               cv.get_feature_names()[i]))


# # Linear Regression Unigram

# In[57]:


corpus = ['This is a useful document for testing']
vectorizer = CountVectorizer(ngram_range=(1, 1))
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())


# In[58]:


rmse = lambda y_true, y_pred: np.sqrt(mse(y_true, y_pred))
rmse_loss = lambda Estimator, X, y: rmse(y, Estimator.predict(X))


# In[59]:


x = train['excerpt_clean']
y = train['target']

print(len(x), len(y))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))


# In[60]:


model = make_pipeline(
    CountVectorizer(ngram_range=(1, 1)),
    LinearRegression(),
)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(f' RMSE: {rmse(y_test, y_pred):.4f}')


# In[61]:


fig, ax = plt.subplots(1, 1, figsize=(20, 6))
sns.histplot(y_test, color="green", ax=ax, label='Testset', kde=True, stat="density", linewidth=0)
sns.histplot(y_pred, color="orange", ax=ax, label='Prediction', kde=True, stat="density", linewidth=0)
ax.legend()
plt.show()


# In[62]:


model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 1)),
    LinearRegression(),
)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(f' RMSE: {rmse(y_test, y_pred):.4f}')


# In[63]:


fig, ax = plt.subplots(1, 1, figsize=(20, 6))
sns.histplot(y_test, color="green", ax=ax, label='Testset', kde=True, stat="density", linewidth=0)
sns.histplot(y_pred, color="orange", ax=ax, label='Prediction', kde=True, stat="density", linewidth=0)
ax.legend()
plt.show()


# Linear Regression Bigram
# 

# In[64]:


corpus = ['This is a useful document for testing']
vectorizer = TfidfVectorizer(ngram_range=(2, 2))
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())


# In[65]:


model = make_pipeline(
    TfidfVectorizer(ngram_range=(2, 2)),
    LinearRegression(),
)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(f' RMSE: {rmse(y_test, y_pred):.4f}')


# In[66]:


fig, ax = plt.subplots(1, 1, figsize=(20, 6))
sns.histplot(y_test, color="green", ax=ax, label='Testset', kde=True, stat="density", linewidth=0)
sns.histplot(y_pred, color="orange", ax=ax, label='Prediction', kde=True, stat="density", linewidth=0)
ax.legend()
plt.show()

Ridge Regression
# In[67]:


model = make_pipeline(
    TfidfVectorizer(binary=True, ngram_range=(1,1)),
    Ridge(fit_intercept=True, normalize=False),
)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(f' RMSE: {rmse(y_test, y_pred):.4f}')


# In[68]:


fig, ax = plt.subplots(1, 1, figsize=(20, 6))
sns.histplot(y_test, color="green", ax=ax, label='Testset', kde=True, stat="density", linewidth=0)
sns.histplot(y_pred, color="orange", ax=ax, label='Prediction', kde=True, stat="density", linewidth=0)
ax.legend()
plt.show()


# Extreme Gradient Boosting

# In[69]:


model = make_pipeline(
    CountVectorizer(ngram_range=(1, 1)),
    xgb.XGBRegressor(),
)

model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(f' RMSE: {rmse(y_test, y_pred):.4f}')


# In[70]:


fig, ax = plt.subplots(1, 1, figsize=(20, 6))
sns.histplot(y_test, color="green", ax=ax, label='Testset', kde=True, stat="density", linewidth=0)
sns.histplot(y_pred, color="orange", ax=ax, label='Prediction', kde=True, stat="density", linewidth=0)
ax.legend()
plt.show()


# In[71]:


def training(model, X_train, y_train, X_test, y_test, model_name, ngram_range):
    t1 = time.time()

    model = make_pipeline(
        TfidfVectorizer(binary=True, ngram_range=ngram_range),
        model,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    RMSE = rmse(y_test, y_pred)

    t2 = time.time()
    training_time = t2 - t1

    print("--- Model:", model_name, "---")
    print("RMSE: ", RMSE)
    print("Training time:", training_time)
    print("\n")


# In[73]:


lr = LinearRegression()
ridge = Ridge(fit_intercept=True, normalize=False)
xgbr = xgb.XGBRegressor()

models = [lr, ridge, xgbr]

modelnames = ["Linear Regression", "Ridge Regression", "Extreme Gradient Boosting"]

X = train["excerpt_clean"]
y = train['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_gram_dict = {"Unigram": (1, 1), "Unigrams + Bigrams": (1, 2), "Bigrams alone": (2, 2),
               "Unigrams + Bigrams + Trigrams": (1, 3), "Trigrams alone": (3, 3)}

for n_gram in n_gram_dict.keys():
    print("\033[1m " + n_gram + " \n \033[0m")
    for i in range(0, len(models)):
        training(model=models[i], X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                 model_name=modelnames[i], ngram_range=n_gram_dict[n_gram])
        print("*" * 40)


# In[ ]:




