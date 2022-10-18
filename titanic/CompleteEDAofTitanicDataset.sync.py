# %% [markdown]
# # Complete EDA of Titanic Dataset
#  

# %% [markdown]
# The RMS Titanic was a British passenger liner that sank in the North Atlantic Ocean in the early morning hours of 15 April 1912, after it collided with an iceberg during its maiden voyage from Southampton to New York City. There were an estimated 2,224 passengers and crew aboard the ship, and more than 1,500 died.
#
# Dataset: https://www.kaggle.com/c/titanic/data

# %% [markdown]
# - Survival - Survival (0 = No; 1 = Yes). Not included in test.csv file.
# - Pclass - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# - Name - Name
# - Sex - Sex
# - Age - Age
# - Sibsp - Number of Siblings/Spouses Aboard
# - Parch - Number of Parents/Children Aboard
# - Ticket - Ticket Number
# - Fare - Passenger Fare
# - Cabin - Cabin
# - Embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# %% [markdown]
# ## Load Dataset


# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline

# %%
plt.rcParams['figure.figsize'] = [6, 3]
plt.rcParams['figure.dpi'] = 80

# %%
titanic = sns.load_dataset('titanic')

# %%
titanic.head()

# %%
cols = titanic.columns
cols

# %%
titanic.describe()

# %%
titanic.info()

# %% [markdown]
#

# %% [markdown]
# # Heatmap

# %%
plt.style.use('ggplot')

# %%
titanic.isnull().sum()

# %%
sns.heatmap(titanic.isnull(), cmap = 'viridis', cbar = True)

# %%
corrmat = titanic.corr(numeric_only = True)
corrmat

# %%
sns.heatmap(corrmat)

# %% [markdown]
# # Univariate Analysis 

# %%
print(list(cols))

# %%

# create countplot for each categorical variable in the dataset using subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 10))


sns.countplot(x = 'survived', data = titanic, ax = axes[0][0])
sns.countplot(x = 'pclass', data = titanic, ax = axes[0][1])
sns.countplot(x = 'sex', data = titanic, ax = axes[0][2])
sns.countplot(x = 'sibsp', data = titanic, ax = axes[1][0])
sns.countplot(x = 'parch', data = titanic, ax = axes[1][1])
sns.countplot(x = 'embarked', data = titanic, ax = axes[1][2])
sns.countplot(x = 'alone', data = titanic, ax = axes[2][0])

plt.tight_layout()
plt.show()

# %%

sns.displot(x = 'fare', data = titanic, kde = True)
sns.displot(x = 'age',data = titanic, kde = True)

plt.tight_layout()
plt.show()

# %% [markdown]
# # Survived

# %%
titanic['survived'].value_counts()

# %%
sns.countplot(x = 'survived', data = titanic)
plt.title('Titanic Survival Plot')
plt.show()

# %%
titanic['survived'].plot.hist()

# %%
titanic['survived'].value_counts().plot.pie()

# %%
titanic['survived'].value_counts().plot.pie(autopct = '%1.2f%%')

# %%
explode = [0, 0.1]
titanic['survived'].value_counts().plot.pie(explode = explode, autopct = '%1.2f%%')

# %% [markdown]
# # PClass

# %%
titanic['pclass'].value_counts()

# %%
titanic.groupby(['pclass', 'survived'])['survived'].count()

# %%
sns.countplot(x = 'pclass', data = titanic)

# %%
sns.countplot(x = 'pclass', data = titanic, hue = 'survived')

# %%
titanic['pclass'].value_counts().plot.pie(autopct = "%1.1f%%")

# %%
sns.catplot(x = 'pclass', y = 'survived', kind = 'bar', data = titanic)

# %%
sns.catplot(x = 'pclass', y = 'survived', kind = 'point', data = titanic)

# %%
sns.catplot(x = 'pclass', y = 'survived', kind = 'violin', data= titanic)

# %%

# %% [markdown]
# # Sex 

# %% [markdown]
# - Survival - Survival (0 = No; 1 = Yes). Not included in test.csv file.
# - Pclass - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# - Name - Name
# - Sex - Sex
# - Age - Age
# - Sibsp - Number of Siblings/Spouses Aboard
# - Parch - Number of Parents/Children Aboard
# - Ticket - Ticket Number
# - Fare - Passenger Fare
# - Cabin - Cabin
# - Embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# %%
titanic['sex'].value_counts()

# %%
titanic.groupby(['sex', 'survived'])['survived'].count()

# %%
sns.countplot(x = 'sex', data = titanic)

# %%
sns.countplot(x ='sex', data = titanic, hue = 'survived')

# %%
titanic['sex'].value_counts().plot.pie(autopct = '%1.1f%%')

# %%
sns.catplot(x = 'sex', y = 'survived', kind = 'bar', data = titanic)

# %%
sns.catplot(x = 'sex', y = 'survived', kind = 'bar', data = titanic, hue = 'pclass')

# %%
sns.catplot(x = 'sex', y = 'survived', kind = 'bar', data = titanic, col = 'pclass')

# %%
sns.catplot(x = 'sex', y = 'survived', kind = 'bar', data = titanic, row = 'pclass')

# %%
sns.catplot(x = 'pclass', y = 'survived', kind = 'bar', data = titanic, col = 'sex')

# %%
sns.catplot(x = 'sex', y = 'survived', kind = 'point', data = titanic)

# %%
sns.catplot(x = 'sex', y = 'survived', kind = 'point', data = titanic, hue = 'pclass')

# %%
sns.catplot(x = 'pclass', y = 'survived', kind = 'point', data = titanic, hue = 'sex')

# %%
sns.catplot(x = 'sex', y = 'survived', kind = 'violin', data = titanic)

# %%
sns.catplot(x = 'sex', y = 'survived', kind = 'violin', data = titanic, hue = 'pclass')

# %%
sns.catplot(x = 'sex', y = 'survived', kind = 'violin', data = titanic, col = 'pclass')

# %% [markdown]
# # Age

# %% [markdown]
# - Survival - Survival (0 = No; 1 = Yes). Not included in test.csv file.
# - Pclass - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# - Name - Name
# - Sex - Sex
# - Age - Age
# - Sibsp - Number of Siblings/Spouses Aboard
# - Parch - Number of Parents/Children Aboard
# - Ticket - Ticket Number
# - Fare - Passenger Fare
# - Cabin - Cabin
# - Embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# %%
titanic['age'].hist(bins = 30, density = True, color = 'orange', figsize = (10, 5))
plt.xlabel('Age')
plt.show()

# %%
sns.displot(data = titanic,x = 'age', kde = True)

# %%
sns.displot(data = titanic, x = 'age', kde = True)

# %%
sns.kdeplot(titanic['age'], fill = True)

# %%
sns.catplot(x = 'sex', y = 'age', data = titanic, kind = 'box')

# %%
sns.catplot(x = 'sex', y = 'age', data = titanic, kind = 'box', hue = 'pclass')

# %%
sns.catplot(x = 'sex', y = 'age', data = titanic, kind = 'box', col = 'pclass')

# %%
sns.catplot(x = 'pclass', y = 'age', data = titanic, kind = 'violin')

# %%
sns.catplot(x = 'pclass', y = 'age', data = titanic, kind = 'violin', hue = 'sex')

# %%
sns.catplot(x = 'pclass', y = 'age', data = titanic, kind = 'violin', hue = 'sex', split = True)

# %%
sns.catplot(x = 'pclass', y = 'age', data = titanic, kind = 'violin', col = 'sex')

# %%
sns.catplot(x = 'pclass', y = 'age', kind = 'swarm', data = titanic, s = 10, height = 6)

# %%
sns.catplot(x = 'pclass', y = 'age', kind = 'swarm', data = titanic, col = 'sex',s = 10, height = 6)

# %%
sns.catplot(x = 'survived', y = 'age', data = titanic, kind = 'swarm', col = 'sex',s = 10, height = 6)

# %%

# %%
sns.catplot(x = 'survived', y = 'age', data = titanic, kind = 'swarm', row = 'sex', col = 'pclass',s = 10, height = 6)

# %% [markdown]
# # Fare

# %% [markdown]
# - Survival - Survival (0 = No; 1 = Yes). Not included in test.csv file.
# - Pclass - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# - Name - Name
# - Sex - Sex
# - Age - Age
# - Sibsp - Number of Siblings/Spouses Aboard
# - Parch - Number of Parents/Children Aboard
# - Ticket - Ticket Number
# - Fare - Passenger Fare
# - Cabin - Cabin
# - Embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# %%
titanic['fare'].hist(bins = 40, color = 'orange')

# %%
sns.displot(data = titanic, x = 'fare', kde = True)
plt.xlabel('Fare')
plt.show()

# %%
sns.kdeplot(titanic['fare'], fill = True)

# %%
sns.catplot(x = 'sex', y = 'fare', data = titanic, kind = 'box')

# %%
sns.catplot(x = 'sex', y = 'fare', data = titanic, kind = 'box', hue = 'pclass')

# %%
sns.catplot(x = 'sex', y = 'fare', data = titanic, kind = 'box', col = 'pclass')

# %%
sns.catplot(x = 'sex', y = 'fare', data = titanic, kind = 'boxen', col = 'pclass')

# %%
sns.catplot(x = 'pclass', y = 'fare', data = titanic, kind = 'swarm', col = 'sex',s = 12, height = 6)

# %%
sns.catplot(x = 'survived', y = 'fare', data = titanic, kind = 'swarm', col = 'sex',s = 12, height = 6)

# %%
sns.catplot(x = 'survived', y = 'fare', data = titanic, kind = 'swarm', col = 'pclass',s = 12, height = 6,  alpha = 0.5)


# %%
sns.jointplot(x = 'age', y = 'fare', data = titanic)

# %%
sns.jointplot(x = 'age', y = 'fare', data = titanic, kind = 'kde')

# %%
sns.relplot(x = 'age', y = 'fare', data = titanic, row = 'sex', col = 'pclass')

# %% [markdown]
# # SibSp

# %% [markdown]
# - Survival - Survival (0 = No; 1 = Yes). Not included in test.csv file.
# - Pclass - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# - Name - Name
# - Sex - Sex
# - Age - Age
# - Sibsp - Number of Siblings/Spouses Aboard
# - Parch - Number of Parents/Children Aboard
# - Ticket - Ticket Number
# - Fare - Passenger Fare
# - Cabin - Cabin
# - Embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# %%
titanic['sibsp'].value_counts()

# %%
sns.countplot(x = 'sibsp', data = titanic)

# %%
sns.countplot(x = 'sibsp', data = titanic, hue = 'survived')

# %%
sns.catplot(x = 'sibsp', y = 'survived', kind = 'bar', data = titanic)

# %%
sns.catplot(x = 'sibsp', y = 'survived', kind = 'bar', data = titanic, hue = 'sex')

# %%
sns.catplot(x = 'sibsp', y = 'survived', kind = 'bar', data = titanic, col = 'sex')

# %%
sns.catplot(x = 'sibsp', y = 'survived', kind = 'bar', data = titanic, col = 'pclass')

# %%
sns.catplot(x = 'sibsp', y = 'survived', kind = 'point', data = titanic)

# %%
sns.catplot(x = 'sibsp', y = 'survived', kind = 'point', data = titanic, hue = 'sex')

# %%
sns.catplot(x = 'sibsp', y = 'survived', kind = 'point', data = titanic, col = 'pclass')

# %%
sns.catplot(x = 'sibsp', y = 'fare', data = titanic, kind = 'swarm', col = 'sex',s = 10, height = 6)

# %%
sns.catplot(x = 'sibsp', y = 'fare', data = titanic, kind = 'swarm', col = 'pclass',s = 10, height = 6)

# %%
sns.catplot(x = 'sibsp', y = 'fare', data = titanic, kind = 'swarm', col = 'pclass', row = 'sex',s = 10, height = 6)

# %% [markdown]
# # Parch 

# %% [markdown]
# - Survival - Survival (0 = No; 1 = Yes). Not included in test.csv file.
# - Pclass - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# - Name - Name
# - Sex - Sex
# - Age - Age
# - Sibsp - Number of Siblings/Spouses Aboard
# - Parch - Number of Parents/Children Aboard
# - Ticket - Ticket Number
# - Fare - Passenger Fare
# - Cabin - Cabin
# - Embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# %%
titanic['parch'].value_counts()

# %%
sns.countplot(x = 'parch', data = titanic)

# %%
sns.countplot(x = 'parch', data = titanic, hue = 'sex')

# %%
sns.catplot(x = 'parch', y = 'survived', data = titanic, kind = 'bar')

# %%
sns.catplot(x = 'parch', y = 'survived', data = titanic, kind = 'bar', hue = 'sex')

# %%
sns.catplot(x = 'parch', y = 'fare', data = titanic, kind = 'swarm',s = 10, height = 6,aspect = 25)

# %%
sns.catplot(x = 'parch', y = 'fare', data = titanic, kind = 'swarm', col = 'sex',s = 10, height = 6)

# %%
sns.catplot(x = 'parch', y = 'fare', data = titanic, kind = 'swarm', col = 'pclass',s = 10, height = 7,aspect = 2)

# %%
sns.catplot(x = 'parch', y = 'fare', data = titanic, kind = 'swarm', col = 'pclass', row = 'sex',s = 10, height = 6)

# %% [markdown]
# # Embarked 

# %% [markdown]
# - Survival - Survival (0 = No; 1 = Yes). Not included in test.csv file.
# - Pclass - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
# - Name - Name
# - Sex - Sex
# - Age - Age
# - Sibsp - Number of Siblings/Spouses Aboard
# - Parch - Number of Parents/Children Aboard
# - Ticket - Ticket Number
# - Fare - Passenger Fare
# - Cabin - Cabin
# - Embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

# %%
titanic['embarked'].value_counts()

# %%
sns.countplot(x = 'embarked', data = titanic)

# %%
sns.countplot(x = 'embarked', data = titanic, hue = 'survived')

# %%
sns.catplot(x = 'embarked', y = 'survived', kind = 'bar', data = titanic)

# %%
sns.catplot(x = 'embarked', y = 'survived', kind = 'bar', data = titanic, hue = 'sex')

# %%
sns.catplot(x = 'embarked', y = 'survived', kind = 'bar', data = titanic, col = 'sex')

# %% [markdown]
# # Who

# %%
cols

# %%
titanic['who'].value_counts()

# %%
sns.countplot(x = 'who', data = titanic)

# %%
sns.countplot(x = 'who', data = titanic, hue = 'survived')

# %%
sns.catplot(x = 'who', y = 'survived', kind = 'bar', data = titanic)

# %%
sns.catplot(x = 'who', y = 'survived', kind = 'bar', data = titanic, hue = 'pclass')

# %%
sns.catplot(x = 'who', y = 'survived', kind = 'bar', data = titanic, col = 'parch')

# %%
sns.catplot(x = 'who', y = 'survived', kind = 'bar', data = titanic, col = 'parch' )
