# %% [markdown]
## Time series plot
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
# %%
home_dir = '../resources/DataVisualization/Time Series Plot/'
# %%
# open a csv file
df = pd.read_csv(home_dir + 'daily-temperature.csv')
# %%
plt.rcParams['figure.figsize'] = [8, 4]
# %%
plt.rcParams['figure.dpi'] = 140
# %%
df.head()
# %%
df.info()
# %%
df = pd.read_csv(home_dir + 'daily-temperature.csv', parse_dates = True)
df.info()
df['Date'] = pd.to_datetime(df['Date'])
# %%
df.info()
df.set_index('Date', inplace = True)
# %%
df.head()

# %% [markdown]
## line and scatter plot
# %%
df.plot(style = '.',color = 'red')
plt.ylabel('Temperature celcius')
plt.title('Line Daily temperature')
# %% [markdown]
## subplots
# %%
groups = df.groupby(pd.Grouper(freq = 'A'))
# %%
keys = groups.groups.keys()
# %%
groups.get_group('1981-12-31')
# %%
for key in keys:
	print(key)
# %%
years = pd.DataFrame()
for key in keys:
	years[key] = groups.get_group(key)['Temp'].values
# %%
years.head()
# %%
years.plot(subplots = True, figsize = (10,30))
plt.show()
plt.tight_layout()
# %% [markdown]
## heat map
# %%
plt.matshow(years.T, aspect = 'auto')
# %% [markdown]
## Histogram and KDE plot
# %%
df.head()
# %%
df.hist(bins = 30,grid = False,color = 'red')
plt.xlabel('Temperature celcius')
plt.show()
# %%
df.plot(kind = 'kde', color = 'red')
plt.xlabel('Temperature')
plt.show()
