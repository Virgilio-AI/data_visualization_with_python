
# %%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline
# %%
import seaborn as sns
import plotly.offline as iplot
import plotly as py
# %%
import cufflinks as cf
# %%
py.__version__
# %%
df = pd.DataFrame()
# %%
py.offline.init_notebook_mode(connected = True)
# %%
cf.go_offline()
# %% [markdown]
## Line Plot
# %%
df = pd.DataFrame(np.random.randn(100,3), columns = ['A','B','C'])
# %%
df.head()
# %%
df = df.cumsum()
# %%
df.shape
# %%
df.iplot()
# %%
df.plot()
# %% [markdown]#
## scatter plot
# %%
df.head()
# %%
df.iplot(x = 'A', y = 'B', mode = 'markers', size = 10, title = 'Scatter Plot', xTitle = 'X', yTitle = 'Y')
# %% [markdown]
## Bar Plot
# %%
titanic = sns.load_dataset('titanic')
# %%
titanic.head()
# %%
titanic.iplot(kind = 'bar', x = 'sex', y = 'survived')
# %%
help(titanic.iplot)
# %%
titanic['sex'].value_counts()
# %%
df.iplot(kind = 'bar',barmode = 'stack', bargap = 0.5)
# %% [markdown]
## Box Plot and Area Plot
# %%
df.iplot(kind = 'box')
# %%
df.iplot(kind = 'area', fill = True)
# %% [markdown]
## 3d plot
# %%
df.iplot(kind = 'surface',colorscale = 'rdylbu')
# %%
df.head()
# %%
cf.datagen.sinwave(10,0.1).iplot(kind = 'surface')
# %% [markdown]
## spread plot and hist plot
# %%
df.iplot(kind = 'spread')
# %%
df.iplot(kind = 'hist', bins = 25, barmode = 'overlay', bargap = 0.5)
# %% [markdown]
## Bubble Plot and Heatmap
# %%
heatmapData = cf.datagen.heatmap(20,20)
# %%
df.corr()
# %%
# create a function that graphs a heatmap of the correlation matrix of a dataframe
# using seaborn and matplotlib
# where red means high correlation and blue means low correlation
# show the correlation values on the heatmap

def correlation_heatmap(df,title = "default"):
	_ , ax = plt.subplots(figsize =(14, 12))
	colormap = sns.diverging_palette(220, 10, as_cmap = True)

	_ = sns.heatmap(
			df.corr(),
			cmap = colormap,
			square=True,
			cbar_kws={'shrink':.9},
			ax=ax,
			annot=True,
			linewidths=0.1,vmax=1.0, linecolor='white',
			annot_kws={'fontsize':12 }
			)
	plt.title(title, y=1.05, size=15)

# %%
correlation_heatmap(df)
# %%
correlation_heatmap(heatmapData)
