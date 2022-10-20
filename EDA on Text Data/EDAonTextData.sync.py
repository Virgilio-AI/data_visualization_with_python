# %% [markdown]
# # EDA on Text Data

# %% [markdown]
# # Women's E-Commerce Clothing Reviews 

# %% [markdown]
# Dataset: https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews

# %% [markdown]
# Clothing ID: Integer Categorical variable that refers to the specific piece being reviewed.
#
# Age: Positive Integer variable of the reviewers age.
#
# Title: String variable for the title of the review.
#
# Review Text: String variable for the review body.
#
# Rating: Positive Ordinal Integer variable for the product score granted by the customer from 1 Worst, to 5 Best.
#
# Recommended IND: Binary variable stating where the customer recommends the product where 1 is recommended, 0 is not 
# recommended.
#
# Positive Feedback Count: Positive Integer documenting the number of other customers who found this review positive.
#
# Division Name: Categorical name of the product high level division.
#
# Department Name: Categorical name of the product department name.
#
# Class Name: Categorical name of the product class name.


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

# %%
import plotly as py
import cufflinks as cf

# %%
from plotly.offline import iplot

# %%
py.offline.init_notebook_mode(connected=True)
cf.go_offline()

# %%

# %% [markdown]
# ## Data Import 

# %%
df = pd.read_csv('Womens Clothing E-Commerce Reviews.csv', index_col=0)
df.head()

# %%
df.drop(labels=['Title', 'Clothing ID'], axis = 1, inplace=True)

# %%
df.head()

# %%
df.isnull().sum()

# %%
df.dropna(subset=['Review Text', 'Division Name'], inplace = True)

# %%
df.isnull().sum()

# %%
' '.join(df['Review Text'].tolist())

# %% [markdown]
# ## Text Cleaning 

# %%
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how does",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
" u ": " you ",
" ur ": " your ",
" n ": " and "}


# %%
def cont_to_exp(x):
    if type(x) is str:
        x = x.replace('\\', '')
        for key in contractions:
            value = contractions[key]
            x = x.replace(key, value)
        return x
    else:
        return x


# %%
x = "i don't know what date is today, I am 5'8\"" 

# %%
print(cont_to_exp(x))

# %%
# %%time
df['Review Text'] = df['Review Text'].apply(lambda x: cont_to_exp(x))

# %%
df.head()

# %%
print(' '.join(df['Review Text'].tolist())[:1000])

# %% [markdown]
# ## Feature Engineering 

# %%
from textblob import TextBlob

# %%
df.head()

# %%
df['polarity'] = df['Review Text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# %%
df['review_len'] = df['Review Text'].apply(lambda x: len(x))

# %%
df['word_count'] = df['Review Text'].apply(lambda x: len(x.split()))


# %%
def get_avg_word_len(x):
    words = x.split()
    word_len = 0
    for word in words:
        word_len = word_len + len(word)
        
    return word_len/len(words)


# %%
df['avg_word_len'] = df['Review Text'].apply(lambda x: get_avg_word_len(x))

# %%
df.head()

# %%

# %% [markdown]
# ## Distribution of Sentiment Polarity 

# %%
df.head()

# %%
df['polarity'].iplot(kind = 'hist', colors = 'red', bins = 50,
                    xTitle = 'Polarity', yTitle = 'Count', title  = 'Sentiment Polarity Distribution')

# %%

# %% [markdown]
# ## Distribution of Reviews Rating and Reviewers Age

# %%
df['Rating'].iplot(kind = 'hist', xTitle = 'Rating', yTitle = 'Count',
                  title = 'Review Rating Distribution')

# %%
df['Age'].iplot(kind = 'hist', bins = 40, xTitle = 'Age', yTitle = 'Count',
               title = 'Reviewers Age Dist', colors = 'orange', linecolor = 'gray')

# %%

# %%

# %% [markdown]
# ## Distribution of Review Text Length and Word Length

# %%
df['review_len'].iplot(kind = 'hist', xTitle = 'Review Len', yTitle = 'Count', title = 'Review Text Len Dist')

# %%
df['word_count'].iplot(kind = 'hist', xTitle = 'Word Count', yTitle = 'Count', title = 'Word Count Distribution')

# %%
df['avg_word_len'].iplot(kind = 'hist', xTitle = 'Avg Word Len', yTitle = 'Count', title = 'Review Text Avg Word Len Dist')

# %%

# %%
df['word_count'].iplot(kind = 'hist', xTitle = 'Word Count', yTitle = 'Count', 
                       title = 'Word Count Distribution')

# %%

# %%

# %% [markdown]
# ## Distribution of Department, Division, and Class 

# %%
df.head(1)

# %%
df['Department Name'].value_counts()

# %%
df.groupby('Department Name').count()

# %%
df['Department Name'].value_counts().iplot(kind = 'bar', yTitle = 'Count', xTitle = 'Department',
                                          title = "Bar Chart of Department's Name")

# %%
df['Division Name'].value_counts().iplot(kind = 'bar', yTitle = 'Count', xTitle = 'Division',
                                          title = "Bar Chart of Division's Name")


# %%
df['Class Name'].value_counts().iplot(kind = 'bar', yTitle = 'Count', xTitle = 'Class',
                                          title = "Bar Chart of Class's Name")


# %%

# %% [markdown]
# ## Distribution of Unigram, Bigram and Trigram 

# %%
from sklearn.feature_extraction.text import CountVectorizer

# %%
x = 'this is a test example'

# unigram = this, is, a, test, example
# bigram = this is, is a, a test, test example
# trigram = this is a, is a test, a test example

# %% [markdown]
# ### Unigram 

# %%
x = ['this is the list list this this this']

# %%
vec = CountVectorizer().fit(x)
bow = vec.transform(x)
sum_words = bow.sum(axis = 0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
words_freq[:2]


# %%
def get_top_n_words(x, n):
    vec = CountVectorizer().fit(x)
    bow = vec.transform(x)
    sum_words = bow.sum(axis = 0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
    return words_freq[:n]


# %%
get_top_n_words(x, 3)

# %%
words = get_top_n_words(df['Review Text'], 20)

# %%
words

# %%
df1 = pd.DataFrame(words, columns = ['Unigram', 'Frequency'])
df1 = df1.set_index('Unigram')
df1.iplot(kind = 'bar', xTitle = 'Unigram', yTitle = 'Count', title = ' Top 20 unigram words')


# %%

# %% [markdown]
# ### Bigram 

# %%
def get_top_n_words(x, n):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(x)
    bow = vec.transform(x)
    sum_words = bow.sum(axis = 0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
    return words_freq[:n]


# %%
get_top_n_words(x, 3)

# %%
words = get_top_n_words(df['Review Text'], 20)

# %%
words

# %%
df1 = pd.DataFrame(words, columns = ['Bigram', 'Frequency'])
df1 = df1.set_index('Bigram')
df1.iplot(kind = 'bar', xTitle = 'Bigram', yTitle = 'Count', title = ' Top 20 Bigram words')


# %%

# %%

# %% [markdown]
# ### Trigram 

# %%
def get_top_n_words(x, n):
    vec = CountVectorizer(ngram_range=(3, 3)).fit(x)
    bow = vec.transform(x)
    sum_words = bow.sum(axis = 0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
    return words_freq[:n]


# %%
get_top_n_words(x, 3)

# %%
words = get_top_n_words(df['Review Text'], 20)

# %%
words

# %%
df1 = pd.DataFrame(words, columns = ['Trigram', 'Frequency'])
df1 = df1.set_index('Trigram')
df1.iplot(kind = 'bar', xTitle = 'Trigram', yTitle = 'Count', title = ' Top 20 Trigram words')


# %%

# %% [markdown]
# ## Distribution of Unigram, Bigram and Trigram without STOP WORDS

# %% [markdown]
# ### Unigram 

# %%
def get_top_n_words(x, n):
    vec = CountVectorizer(ngram_range=(1, 1), stop_words='english').fit(x)
    bow = vec.transform(x)
    sum_words = bow.sum(axis = 0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
    return words_freq[:n]


# %%
get_top_n_words(x, 3)

# %%
words = get_top_n_words(df['Review Text'], 20)

# %%
words

# %%
df1 = pd.DataFrame(words, columns = ['Unigram', 'Frequency'])
df1 = df1.set_index('Unigram')
df1.iplot(kind = 'bar', xTitle = 'Unigram', yTitle = 'Count', title = ' Top 20 Unigram words')


# %%

# %%

# %% [markdown]
# ### Bigram 

# %%
def get_top_n_words(x, n):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(x)
    bow = vec.transform(x)
    sum_words = bow.sum(axis = 0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
    return words_freq[:n]


# %%
get_top_n_words(x, 3)

# %%
words = get_top_n_words(df['Review Text'], 20)

# %%
words

# %%
df1 = pd.DataFrame(words, columns = ['Bigram', 'Frequency'])
df1 = df1.set_index('Bigram')
df1.iplot(kind = 'bar', xTitle = 'Bigram', yTitle = 'Count', title = ' Top 20 Bigram words')


# %%

# %%

# %% [markdown]
# ### Trigram 

# %%
def get_top_n_words(x, n):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(x)
    bow = vec.transform(x)
    sum_words = bow.sum(axis = 0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
    return words_freq[:n]


# %%
x

# %%
# get_top_n_words(x, 3)

# %%
words = get_top_n_words(df['Review Text'], 20)

# %%
words

# %%
df1 = pd.DataFrame(words, columns = ['Trigram', 'Frequency'])
df1 = df1.set_index('Trigram')
df1.iplot(kind = 'bar', xTitle = 'Trigram', yTitle = 'Count', title = ' Top 20 Trigram words')

# %%

# %%

# %% [markdown]
# ## Distribution of Top 20 Parts-of-Speech POS tags 

# %%
# !pip install nltk

# %%
import nltk

# %%
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# %%
print(str(df['Review Text']))

# %%
blob = TextBlob(str(df['Review Text']))

# %%
nltk.download('tagsets')

# %%
print(nltk.help.upenn_tagset())

# %%
pos_df = pd.DataFrame(blob.tags, columns = ['words', 'pos'])
pos_df = pos_df['pos'].value_counts()
pos_df

# %%
pos_df.iplot(kind = 'bar')

# %%

# %%

# %%

# %% [markdown]
# ----

# %% [markdown]
# ## Bivariate Analysis 

# %%
df.head(2)

# %%
# create a dataframe with only the numeric columns
df_num = df.select_dtypes(include = np.number)


# %%
g = sns.pairplot(data = df_num, diag_kind = 'kde', corner = True, plot_kws = dict(marker="+",linewidth=1), diag_kws = dict(fill = False))
g.map_lower(sns.kdeplot, levels = 4, color = '.2')

# %%
sns.catplot(x = 'Division Name', y = 'polarity', data = df)

# %%
sns.catplot(x = 'Division Name', y = 'polarity', data = df, kind = 'box')

# %%
sns.catplot(x = 'Department Name', y = 'polarity', data = df)

# %%
sns.catplot(x = 'Department Name', y = 'polarity', data = df, kind = 'box')

# %%
sns.catplot(x = 'Division Name', y = 'review_len', data = df, kind = 'box')

# %%
sns.catplot(x = 'Department Name', y = 'review_len', data = df, kind = 'box')

# %% [markdown]
# ## Distribution of Sentiment Polarity of Reviews Based on the Recommendation 

# %%
import plotly.express as px
import plotly.graph_objects as go

# %%
x1 = df[df['Recommended IND']==1]['polarity']
x0 = df[df['Recommended IND']==0]['polarity']

# %%
type(x1)

# %%
trace0 = go.Histogram(x = x0, name = 'Not Recommended', opacity = 0.7)
trace1 = go.Histogram(x = x1, name = 'Recommended', opacity = 0.7)

# %%
data = [trace0, trace1]
layout = go.Layout(barmode = 'overlay', title = 'Distribution of Sentiment Polarity of Reviews Based on the Recommendation')
fig = go.Figure(data = data, layout = layout)

iplot(fig)

# %% [markdown]
# ## Distribution of Ratings Based on the Recommendation 

# %%

# %%
x1 = df[df['Recommended IND']==1]['Rating']
x0 = df[df['Recommended IND']==0]['Rating']

# %%
type(x1)

# %%
trace0 = go.Histogram(x = x0, name = 'Not Recommended', opacity = 0.7)
trace1 = go.Histogram(x = x1, name = 'Recommended', opacity = 0.7)

# %%
data = [trace0, trace1]
layout = go.Layout(barmode = 'overlay', title = 'Distribution of Reviews Rating Based on the Recommendation')
fig = go.Figure(data = data, layout = layout)

iplot(fig)

# %%
sns.jointplot(x = 'polarity', y = 'review_len', data = df, kind = 'kde')

# %%
sns.jointplot(x = 'polarity', y = 'Age', data = df, kind = 'kde')
