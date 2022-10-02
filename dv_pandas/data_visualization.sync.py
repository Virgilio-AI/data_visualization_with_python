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

