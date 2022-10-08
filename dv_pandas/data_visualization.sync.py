# %% [markdown]
##pandas

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# %%
from numpy.random import randn,randint, uniform, sample

# %%
data_dir = '../resources/DataVisualization/'
# %%

df = pd.DataFrame(randn(1000), index=pd.date_range('2019-06-07',periods=1000), columns=['value'])
# %%
df


# %%
df.head()
# %%
ts = pd.Series(randn(1000),index = pd.date_range('2019-06-07',periods=1000))
# %%
ts
# %%
ts.head()
# %%
ts.plot()
# %%
df['value'] = df['value'].cumsum()
# %%
df.head()
# %%
ts = ts.cumsum()
# %%
ts.plot()
# %%
df.plot()
# %%
df.plot(figsize=(12,5))
# %%
iris = sns.load_dataset('iris')
# %%
iris.head()
# %%
iris.plot(figsize=(12,5))
# %%
iris.plot(figsize=(12,5), subplots=True)
# %%
iris.plot(figsize=(12,5),logy=True)
# %% [markdown]
## More on Line Plot

# %%
x = iris.drop(labels=['sepal_width','petal_width'],axis=1)
# %%
x.head()
# %%
y = iris[['sepal_width','petal_width']]
# %%
y.head()
# %%
ax = x.plot()
y.plot(figsize=(16,10),secondary_y = True, ax=ax)


# %% [markdown]
## Bar Plot
# %%
iris.head(15)
# %%
df = iris.drop(['species'],axis=1)
# %%
df.iloc[0]
# %%
df.iloc[0].plot(kind='bar')
# %%
# get the average of each column
# show the exact value on the bar
df.mean().plot(kind='bar')

# %%
titanic = sns.load_dataset('titanic')
# %%

titanic.head()
# %%

titanic['pclass'].plot(kind='hist')

# %% [markdown]
## stacked plot
# %%

df = pd.DataFrame(randn(10,4),columns=['a','b','c','d'])
# %%
df.plot.bar(stacked=True)
# %%
df.plot.barh(stacked=True)
# %% [markdown]
## Histogram
# %%
iris.plot.hist()
# %%
# get all the unique values for all columns
iris['petal_width'].unique()


# %%
iris.plot(kind='hist',bins=50)
# %%
iris.plot(kind='hist',bins=50,stacked=True)
# %%
iris.plot(kind='hist',bins=50,stacked=True,orientation='horizontal')

# %%
iris['petal_length'].plot(kind='hist',bins=50)

# %%

df = iris.drop(['species'],axis=1)
# %%
df.diff().hist(color='k',alpha=0.5,bins=50)


# %% [markdown]
## Box Plot
# %%
df = iris.drop(['species'],axis=1)
df.head()
# %%
df.plot(kind='box')
# %%
df.plot(kind='box',figsize=(12,5))
# %%
df.info()
# %%
df.describe()
# %%
df['sepal_length'].mean()
# %% [markdown]
## Area and scatter plot
# %%
df.head()
# %%
df.plot(kind='area')
# %%
df.plot(kind='area',stacked=False,alpha=1)
# %%
df.plot.scatter(x='sepal_width',y='petal_width')
# %%
df.head()
# %%
df.plot.scatter(x='sepal_width',y='petal_width',c = 'sepal_length',label='width')
# %%
ax = df.plot.scatter(x='sepal_width',y = 'petal_width',label='width')
df.plot.scatter(x = 'sepal_length',y = 'petal_length',ax=ax,color='r')

# %% [markdown]
## Hex and Pie Plot

# %%
df.plot.hexbin(x='sepal_length',y = 'petal_length',gridsize=10,C = 'sepal_width')
# %%
df.iloc[0]
# %%
df.iloc[0].plot.pie()
# %%
d = df.head().T
d
# %%
d.plot.pie(subplots=True,figsize = (30,30),autopct='%.2f')
plt.show()
# %%
series = pd.Series([0.2] * 4, index=['a','b','c','d'],name='Pie Plot')
series
# %%
# create a pie plot that leaves empty space if the value does not adds up to 1
series.plot.pie(figsize=(6,6),autopct='%.2f',startangle=90,explode=[0,0,0.3,0],shadow=True)
plt.show()

# %% [markdown]
## scatter matrix and subplots
# %%
from pandas.plotting import scatter_matrix
import scipy
# %%
scatter_matrix(df,figsize=(12,12),diagonal='kde',color='r')
plt.show()
# %%
df.plot(subplots=True)
plt.show()
# %%

df.plot(subplots=True,sharex = False,layout=(1,4),figsize=(12,5))
plt.show()

df.plot(subplots=True)
plt.show()

