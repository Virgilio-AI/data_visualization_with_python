# %% [markdown]
# # seaborn

# %% [markdown]
# # scatter plot

# %%
import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# %matplotlib inline

# %%
tips = sns.load_dataset('tips')
tips.head()

# %% [markdown]
# # hue, style, size

# %%
sns.relplot(x = 'total_bill', y = 'tip', data = tips, hue = 'smoker')

# %%
sns.relplot(x = 'total_bill', y = 'tip', data = tips, hue = 'smoker',style = 'time')

# %%
sns.relplot(x = 'total_bill', y = 'tip', data = tips, size = 'size')

# %%
tips['size'].value_counts()

# %%
sns.relplot(x = 'total_bill', y = 'tip', data = tips, style = 'smoker')

# %%
tips['smoker'].value_counts()

# %%
sns.relplot(x = 'total_bill', y = 'tip', data = tips, size = 'size', sizes = (15,200))

# %%
tips['time'].value_counts()

# %% [markdown]
# # Line Plot

# %%
from numpy.random import randn
from seaborn import relplot

# %%
df = pd.DataFrame(dict(time = np.arange(500),value = randn(500).cumsum()))

# %%
relplot(x = 'time', y = 'value', kind = 'line', data = df)

# %%
fmri = sns.load_dataset('fmri')
fmri.head()

# %%
fig, ax = plt.subplots()
sns.relplot(x = 'timepoint', y = 'signal', kind = 'line', data = fmri)

# %%
relplot(x = 'timepoint', y = 'signal', kind = 'line', data = fmri, label = 'signal', height = 4, aspect = 2)
plt.legend(loc = 'right')
plt.show()

# %%
fmri[fmri['timepoint'] == 0].mean()

# %%
relplot(x = 'timepoint', y = 'signal',estimator = None, kind = 'line', data = fmri, label = 'signal', height = 4, aspect = 2)
plt.legend(loc = 'right')

# %%
fmri

# %%
relplot(x = 'timepoint', y = 'signal', kind = 'line', data = fmri, label = 'signal', height = 4, aspect = 2,hue = 'event')
plt.legend(loc = 'right')

# %%
relplot(x = 'timepoint', y = 'signal', kind = 'line', data = fmri, label = 'signal', height = 4, aspect = 2,style = 'region',hue = 'event')
plt.legend(loc = 'right')

# %%
relplot(x = 'timepoint', y = 'signal', kind = 'line', data = fmri, label = 'signal', height = 4, aspect = 2,hue = 'event',marker = True,dashes = False)
plt.legend(loc = 'right')

# %%
dots = sns.load_dataset('dots').query("align == 'dots'")

# %%
dots.head()

# %%
relplot(x = 'time', y = 'firing_rate', data = dots, kind = 'line',hue = 'coherence',style = 'choice' )

# %%
palette = sns.color_palette( n_colors = 6)
relplot(x = 'time', y = 'firing_rate', data = dots, kind = 'line',hue = 'coherence',style = 'choice' , palette = palette)

# %%
relplot(x = 'time', y = 'firing_rate', data = dots, kind = 'line',hue = 'coherence',style = 'choice' , size = 'choice')

# %% [markdown]
# # subplots

# %%
tips.head()

# %%
relplot(x = 'total_bill', y = 'tip', hue = 'smoker', col = 'smoker',data = tips)

# %%
relplot(x = 'total_bill', y = 'tip', hue = 'smoker', col = 'size',data = tips)

# %%
relplot(x = 'total_bill', y = 'tip', hue = 'smoker', col = 'time',data = tips,row = 'size')

# %%
relplot(x = 'total_bill', y = 'tip', hue = 'smoker', col = 'size',data = tips, col_wrap = 3)
# %% [markdown]
## using sns.lineplot()
# %%
fmri.head()
# %%
sns.lineplot(x = 'timepoint', y = 'signal', style = 'event', hue = 'region', data = fmri, markers = True, errorbar=('ci', 68), err_style = 'bars')
# %%
sns.scatterplot(x = 'total_bill', y = 'tip', data = tips, hue = 'smoker', size = 'size',style = 'time')
# %% [markdown]
## categorical data plotting
# %%
tips.head()
# %%
sns.catplot(x = 'day', y = 'total_bill', data = tips,palette = 'Dark2', hue = 'day')
# %%
sns.catplot(y = 'day', x = 'total_bill', data = tips,palette = 'Dark2', hue = 'day')
# %%
sns.catplot(x = 'day', y = 'total_bill', data = tips,palette = 'Dark2', hue = 'day', jitter = False)
# %%
sns.catplot(x = 'day', y = 'total_bill', data = tips, kind = 'swarm', palette = 'Dark2', hue = 'size')
# %%
sns.catplot(x = 'smoker', y = 'tip', data = tips,order = ['No','Yes'],hue = 'smoker')
# %% [markdown]
## boxplot
# %%
sns.catplot(x = 'day', y = 'total_bill', kind = 'box', data = tips, hue = 'sex')
# %%
sns.catplot(x = 'day', y = 'total_bill', kind = 'box', data = tips, hue = 'sex',dodge = False)
# %% [markdown]
## boxen Plot
# %%
diamonds = sns.load_dataset('diamonds')
# %%
diamonds.head()
# %%
sns.catplot(x = 'color', y = 'price', kind = 'boxen', data = diamonds)
# %%
sns.catplot(x = 'color', y = 'price', kind = 'boxen', data = diamonds,hue = 'cut')


# %%
sns.catplot(x = 'color', y = 'price', kind = 'boxen', data = diamonds,dodge = False, hue = 'cut')

# %% [markdown]
## violin plot
# %%
sns.catplot(x = 'total_bill', y = 'day', data = tips, kind = 'violin',split = True, hue = 'sex')


# %%
sns.catplot(x = 'total_bill', y = 'day', data = tips, kind = 'violin', hue = 'sex')


# %%
g = sns.catplot(x = 'day', y = 'total_bill', data = tips, kind = 'swarm',hue = 'day', palette = 'Dark2')
sns.violinplot(x = 'day', y = 'total_bill', data = tips, ax = g.ax,split = True,hue = 'sex',palette = 'Dark2',height = 3)

# %% [markdown]
## bar plot
# %%
titanic = sns.load_dataset('titanic')
titanic.head()
# %%
sns.catplot(x = 'sex', y = 'survived', kind = 'bar', data = titanic, hue = 'class')
# %%
sns.catplot(x = 'sex', y = 'survived', kind = 'bar', data = titanic, hue = 'class', palette = 'ch:0.35')
# %% [markdown]
## Point plot
# %%
sns.catplot(x = 'sex', y = 'survived', hue = 'class', kind = 'point', data = titanic)
# %% [markdown]
## Joint plot
# %%
tips.head()
# %%
x = tips['total_bill']
y = tips['tip']
# %%
sns.jointplot(x = x,y = y)


# %%
sns.jointplot(x = x,y = y, kind = 'hex')

# %%
sns.jointplot(x = x,y = y, kind = 'kde', fill = True, thresh = 0.6)

# %% [markdown]
## sns.pairplot()
# %%
sns.pairplot(tips)
# %%
g = sns.PairGrid(tips)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels = 10)

# %%
g = sns.PairGrid(tips)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.scatterplot)


# %% [markdown]
## regression plot
# %%
tips.head()
# %%
sns.regplot(x = 'total_bill', y = 'tip', data = tips)
# %%
sns.regplot(x = 'total_bill', y = 'tip', data = tips, x_jitter = 0.05)

# %%
sns.lmplot(x = 'size', y = 'tip', data = tips)


# %% [markdown]
## control figure aesthetics
# styling
# axes style
# color palettes
# etc
# %%
def sinplot(flip = 1):
	x = np.linspace(0,14,100)
	for i in range(1,7):
		plt.plot(x,np.sin(x + i * 0.5) * (7 - i) * flip)

# %%
sinplot()
# %%
sinplot(-1)
# %%
sns.set_style('ticks', {'axes.grid':True, 'xticks.direction':'in'})
sinplot()
# %%
sns.set_style('ticks', {'axes.grid':True, 'xticks.direction':'in'})
sinplot()
sns.despine(left = True, bottom = False)


