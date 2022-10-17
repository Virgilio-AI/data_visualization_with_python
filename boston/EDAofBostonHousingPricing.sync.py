# %%

# %% [markdown]
# # EDA of Boston Housing Price Prediction Dataset

# %% [markdown]
# ### What is EDA 

# %% [markdown]
# Exploratory Data Analysis is a technique which is used to understand the data

# %% [markdown]
# - maximize insight into a data set
# - uncover underlying structure
# - extract important variables
# - detect outliers and anomalies
# - test underlying assumptions
# - determine optimal factor settings

# %% [markdown]
# Dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/housing/

# %% [markdown]
# # Data Preparation
#

# %%
# !pip install statsmodels

# %%
# from sklearn.ensemble import RandomForestRegressor # random forest regressor
# from sklearn.model_selection import cross_val_score # the cross validation module

# usefull
import pickle # for saving the model
import os # for saving the model


# Setup feedback system
# from learntools.core import binder # the binder
# binder.bind(globals())
# from learntools.feature_engineering_new.ex2 import * # for importing all in feature engineering



# import matplotlib.pyplot as plt # for importing matplotlib
# import numpy as np # for importing numpy
# import pandas as pd # for importing pandas
# import seaborn as sns # seaborn is used to plot
# from numpy.random import randint
# from matplotlib.patches import Polygon
# from matplotlib.animation import FuncAnimation

# import plotly.offline as py # for importing plotly
# import cufflinks as cf
# py.init_notebook_mode(connected=True) # do not delete for plotly to work correctly
# cf.go_offline() # do not delete for plotly to work correctly

# %%
#
import warnings
warnings.filterwarnings('ignore')

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

# %%
import pickle as pkl

# %%
boston = pkl.load(open('boston_housing.pkl', 'rb'))

# %%
type(boston)

# %%
boston.keys()

# %%
print(boston.DESCR)

# %%
boston['feature_names']

# %%
data = boston.data

# %%
data.shape

# %%
data = pd.DataFrame(data = data, columns=boston.feature_names)
data

# %%
data['Price'] = boston.target
data.head()

# %%

# %% [markdown]
# # Understand Your Data and Plot Style Setting 

# %%
data.describe()

# %%
data_desc = data.describe()
type(data_desc.loc['mean'])
# %%
print(data_desc.loc['mean'])
# %%
data.info()

# %%
data.isnull().sum()

# %%
data.duplicated().sum()

# %%
data_desc.loc['mean'].plot.bar()

# %% [markdown]
# # Plot Styling 

# %%
print(plt.style.available)

# %%
len(plt.style.available)

# %%
for style in plt.style.available:
    plt.style.use(style)
    data_desc.loc['mean'].plot.bar()
    plt.title(style)
    plt.savefig('plots/' + style + ".png")

# %%
plt.style.use('ggplot')
data_desc.loc['mean'].plot.bar()

# %%

# %% [markdown]
# # Pair Plot 

# %%
import seaborn as sns

# %%
data.head()

# %%
# seaborn pairplot
sns.pairplot(data, diag_kind = 'kde', hue = 'Price')

# %%
tmp_data = data.iloc[:,:7]
tmp_data['Price'] = data['Price']
tmp_data.head()

# %%
sns.pairplot(tmp_data, diag_kind = 'kde', hue = 'Price')

# %%
tmp_data2 = data.iloc[:,7:]
tmp_data2.head()

# %%
sns.pairplot(tmp_data2, diag_kind = 'kde', hue = 'Price')



# %%

# %% [markdown]
# # Distribution Plot

# %%
rows = 4
cols = 4

fig, ax = plt.subplots(nrows=rows, ncols = cols, figsize = (20,20))

col = data.columns
index = 0
limit = 14
index = 0

for i in range(rows):
	for j in range(cols):

		if index >= limit:
			continue

		sns.distplot(data[col[index]], ax = ax[i][j])
		ax[i][j].tick_params('x', labelrotation=45)

		# change font size of y axis label 
		ax[i][j].yaxis.label.set_size(20)
		# change font size of x axis label
		ax[i][j].xaxis.label.set_size(20)

		index = index + 1

plt.tight_layout()

# %%

# %% [markdown]
# # Scatter Plot
# ## Plotting `Price` with remaining columns 

# %%
rows = 4
cols = 4

fig, ax = plt.subplots(rows, cols, figsize = (20,20))

col = data.columns
index = 0
limit = 14

for i in range(rows):
	for j in range(cols):

		if index >= limit:
			continue
		sns.scatterplot(x = 'Price', y = col[index], data = data, ax = ax[i][j])
		ax[i][j].tick_params('x', labelrotation=45)
		# change font size of y axis label 
		ax[i][j].yaxis.label.set_size(20)
		# change font size of x axis label
		ax[i][j].xaxis.label.set_size(20)
		index = index + 1


plt.tight_layout()
plt.show()

# %%

# %% [markdown]
# # Heatmap 

# %%
corrmat = data.corr()
corrmat

# %%
corrmat.shape

# %%
import matplotlib
matplotlib.__version__

# %%
fig, ax = plt.subplots(figsize = (7, 5))
sns.heatmap(corrmat, annot = True, annot_kws = {'size': 7}, cmap = 'coolwarm')

bottom, top = ax.get_ylim()
ax.set_ylim(bottom+0.5, top-.5)
plt.show()

# %%

# %%

# %% [markdown]
# # Correlated Feature Selection

# %%
corrmat.index


# %%
def getCorrelatedFeature(corrdata, threshold):
	feature = []
	value = []

	for i, index in enumerate(corrdata.index):
		if abs(corrdata[index]) > threshold:
			feature.append(index)
			value.append(corrdata[index])

	df = pd.DataFrame(data = value, index=feature, columns=['corr value'])

	return df


# %%
threshold = 0.5
corr_df = getCorrelatedFeature(corrmat['Price'], threshold)

# %%
corr_df

# %%

# %% [markdown]
# # Heatmap and Pair Plot of Correlated Data 

# %%
correlated_data = data[corr_df.index]
correlated_data.head()

# %%
sns.pairplot(correlated_data, diag_kind = 'kde')
plt.tight_layout()

# %%
fig, ax = plt.subplots(figsize = (4, 4))
sns.heatmap(correlated_data.corr(), annot = True, annot_kws = {'size': 12})

bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
plt.show()

# %%

# %% [markdown]
# # Box and Rel Plot 

# %% [markdown]
#
#     :Attribute Information (in order):
#         - CRIM     per capita crime rate by town
#         - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
#         - INDUS    proportion of non-retail business acres per town
#         - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#         - NOX      nitric oxides concentration (parts per 10 million)
#         - RM       average number of rooms per dwelling
#         - AGE      proportion of owner-occupied units built prior to 1940
#         - DIS      weighted distances to five Boston employment centres
#         - RAD      index of accessibility to radial highways
#         - TAX      full-value property-tax rate per $10,000
#         - PTRATIO  pupil-teacher ratio by town
#         - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#         - LSTAT    % lower status of the population
#         - MEDV     Median value of owner-occupied homes in $1000's

# %%
sns.boxplot(y = 'Price', x = 'CHAS', data = data)

# %%
sns.relplot(x = 'RM', y = 'Price', data = data, hue = 'CHAS')

# %%
sns.relplot(x = 'RM', y = 'Price', data = data, style = 'CHAS')

# %%
sns.relplot(x = 'RM', y = 'Price', data = data, size = 'CHAS')

# %%
sns.relplot(x = 'RM', y = 'Price', data = data, col = 'CHAS')

# %% [markdown]
# # Joint Plot 

# %% [markdown]
# When dealing with a set of data, often the first thing you’ll want to do is get a sense for how the variables are distributed

# %%
sns.jointplot(x = data['RM'], y = data['Price'])

# %%
sns.jointplot(x = data['RM'], y = data['Price'], kind = 'hex')

# %%
sns.jointplot(x = data['RM'], y = data['Price'], kind = 'kde')

# %%
g = sns.jointplot(x = data['RM'], y = data['Price'], kind = 'kde', color = 'm')
g.plot_joint(plt.scatter, c = 'r', s = 40, linewidth = 1, marker = '+')
g.ax_joint.collections[0].set_alpha(0.3)

# %%
fig, ax = plt.subplots(figsize = (6, 6))
cmap = sns.cubehelix_palette(as_cmap = True, dark = 0, light = 1, reverse = True)
sns.kdeplot(x = data['RM'], y = data['Price'], cmap = cmap, n_levels = 60, fill = True)

# %%

# %% [markdown]
# # Linear Regression and Relationship

# %% [markdown]
# - regplot()
# - lmplot()

# %%
data.head()

# %%
sns.regplot(x = 'RM', y = 'Price', data = data, robust=True)

# %%
sns.lmplot(x = 'RM', y = 'Price', data = data)

# %%
sns.lmplot(x = 'RM', y = 'Price', data = data, hue = 'CHAS')

# %%
sns.lmplot(x = 'RM', y = 'Price', data = data, col = 'CHAS')

# %%
sns.lmplot(x = 'RM', y = 'Price', data = data, col = 'CHAS', robust=True)

# %%
sns.lmplot(x = 'RM', y = 'Price', data = data, col = 'CHAS', order = 2)

# %%
sns.lmplot(x = 'CHAS', y = 'Price', data = data, x_estimator=np.mean)
