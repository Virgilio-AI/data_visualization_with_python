# %% [markdown]
# # EDA on FIFA World Cup Matches
#

# %% [markdown]
# Dataset: https://www.kaggle.com/abecklas/fifa-world-cup


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
world_cups = pd.read_csv('WorldCups.csv')
players = pd.read_csv('WorldCupPlayers.csv')
matches = pd.read_csv('WorldCupMatches.csv')

# %%
world_cups.head()

# %%
players.head()

# %%
matches.head()

# %%
matches.tail()

# %% [markdown]
# # Data Cleaning 

# %%
matches.dropna(subset=['Year'], inplace=True)

# %%
matches.tail()

# %%
matches['Home Team Name'].value_counts()

# %%
names = matches[matches['Home Team Name'].str.contains('rn">')]['Home Team Name'].value_counts()
names

# %%
names.index

# %%
wrong = list(names.index)
wrong

# %%
correct = [name.split('>')[1] for name in wrong]
correct

# %%
old = ['Germany FR', 'Maracan� - Est�dio Jornalista M�rio Filho', 'Estadio do Maracana']
new = ['Germany', 'Maracanã Stadium', 'Maracanã Stadium']

# %%
wrong = wrong + old
correct = correct + new

# %%
wrong, correct

# %%

# %%
for index, wr in enumerate(wrong):
    world_cups = world_cups.replace(wrong[index], correct[index])
    
for index, wr in enumerate(wrong):
    matches = matches.replace(wrong[index], correct[index])
    
for index, wr in enumerate(wrong):
    players = players.replace(wrong[index], correct[index])
    

# %%

# %%
names = matches[matches['Home Team Name'].str.contains('rn">')]['Home Team Name'].value_counts()
names

# %%

# %%

# %% [markdown]
# #  Most Number of World Cup Winning Title

# %%
winner = world_cups['Winner'].value_counts()
winner

# %%
runnerup = world_cups['Runners-Up'].value_counts()
runnerup

# %%
third = world_cups['Third'].value_counts()
third

# %%
teams = pd.concat([winner, runnerup, third], axis = 1)
teams.fillna(0, inplace = True)
teams = teams.astype(int)
teams

# %%
teams.iplot(kind = 'bar', xTitle = 'Teams', yTitle = 'Count', title = 'FIFA World Cup Winning Count')

# %% [markdown]
# # Number of Goal Per Country 

# %%
matches.head(2)

# %%
home = matches[['Home Team Name', 'Home Team Goals']].dropna()
away = matches[['Away Team Name', 'Away Team Goals']].dropna()

# %%
home.columns = ['Countries', 'Goals']
away.columns = home.columns

# %%
goals = home.append(away, ignore_index=True)

# %%
goals = goals.groupby('Countries').sum()
goals

# %%
goals = goals.sort_values(by = 'Goals', ascending=False)
goals

# %%
goals[:20].iplot(kind = 'bar', xTitle = 'Country Names', yTitle = 'Goals', title = 'Countries Hits Most Number of Goals')

# %% [markdown]
# # Attendance, Number of Teams, Goals, and Matches per Cup

# %%
world_cups['Attendance'] = world_cups['Attendance'].str.replace(".", "")

# %%
world_cups.head(1)

# %%
fig, ax = plt.subplots(figsize = (10, 5))
sns.despine(right = True)
g = sns.barplot(x = 'Year', y = 'Attendance', data = world_cups)
g.set_xticklabels(g.get_xticklabels(), rotation = 80)
g.set_title('Attendance per Year')

#======================
fig, ax = plt.subplots(figsize = (10, 5))
sns.despine(right = True)
g = sns.barplot(x = 'Year', y = 'QualifiedTeams', data = world_cups)
g.set_xticklabels(g.get_xticklabels(), rotation = 80)
g.set_title('Qualified Teams per Year')

#======================
fig, ax = plt.subplots(figsize = (10, 5))
sns.despine(right = True)
g = sns.barplot(x = 'Year', y = 'GoalsScored', data = world_cups)
g.set_xticklabels(g.get_xticklabels(), rotation = 80)
g.set_title('Goals Scored by Teams per Year')

#======================
fig, ax = plt.subplots(figsize = (10, 5))
sns.despine(right = True)
g = sns.barplot(x = 'Year', y = 'MatchesPlayed', data = world_cups)
g.set_xticklabels(g.get_xticklabels(), rotation = 80)
g.set_title('Matches Played by Teams per Year')



plt.show()

# %%

# %% [markdown]
# # Goals Per Team Per Word Cup 

# %%
matches.head(1)

# %%
home = matches.groupby(['Year', 'Home Team Name'])['Home Team Goals'].sum()
home

# %%
away = matches.groupby(['Year', 'Away Team Name'])['Away Team Goals'].sum()
away

# %%
goals = pd.concat([home, away], axis = 1)
goals.fillna(0, inplace = True)
goals['Goals'] = goals['Home Team Goals'] + goals['Away Team Goals']
goals = goals.drop(labels = ['Home Team Goals', 'Away Team Goals'], axis = 1)
goals

# %%
goals = goals.reset_index()

# %%
goals.columns = ['Year', 'Country', 'Goals']
goals = goals.sort_values(by = ['Year', 'Goals'], ascending=[True, False])
goals

# %%
top5 = goals.groupby('Year').head(5)
top5.head(10)

# %%
import plotly.graph_objects as go

# %%
x, y = goals['Year'].values, goals['Goals'].values

# %%

# %%
data = []

for team in top5['Country'].drop_duplicates().values:
    year = top5[top5['Country'] == team]['Year']
    goal = top5[top5['Country'] == team]['Goals']
    
    data.append(go.Bar(x = year, y = goal, name = team))
    
layout = go.Layout(barmode = 'stack', title = 'Top 5 Teams with Most Goals', showlegend = False)

fig = go.Figure(data = data, layout = layout)
fig.show()

# %%

# %%

# %% [markdown]
# # Matches with Highest Number of Attendance

# %%
matches.head(1)

# %%
matches['Datetime'] = pd.to_datetime(matches['Datetime'])

# %%
matches['Datetime'] = matches['Datetime'].apply(lambda x: x.strftime('%d %b, %Y'))

# %%
matches.head(1)

# %%

# %%
top10 = matches.sort_values(by = 'Attendance', ascending = False)[:10]
top10['vs'] = top10['Home Team Name'] + " vs " + top10['Away Team Name']

plt.figure(figsize = (12, 9))

ax = sns.barplot(y = top10['vs'], x = top10['Attendance'])
sns.despine(right = True)

plt.ylabel('Match Teams')
plt.xlabel('Attendance')
plt.title('Matches with the highest number of attendance')

for i, s in enumerate("Stadium: " + top10['Stadium'] + ", Date: " + top10['Datetime']):
    ax.text(2000, i, s, fontsize = 12, color = 'white')

plt.show()

# %%

# %% [markdown]
# # Stadiums with Highest Average Attendance

# %%
matches['Year'] = matches['Year'].astype(int)

std = matches.groupby(['Stadium', 'City'])['Attendance'].mean().reset_index().sort_values(by = 'Attendance', ascending = False)

top10 = std[:10]


plt.figure(figsize= (12, 9))
ax = sns.barplot(y = top10['Stadium'], x = top10['Attendance'])
sns.despine(right = True)

plt.ylabel('Stadium Names')
plt.xlabel('Attendance')
plt.title('Stadium with the highest number of attendance')

for i, s in enumerate("City: " + top10['City']):
    ax.text(2000, i, s, fontsize = 12, color = 'white')

plt.show()

# %%
matches['City'].value_counts()[:20].iplot(kind = 'bar')


# %% [markdown]
# # Match outcomes by home and away teams

# %%
def get_labels(matches):
    if matches['Home Team Goals'] > matches['Away Team Goals']:
        return 'Home Team Win'
    if matches['Home Team Goals'] < matches['Away Team Goals']:
        return 'Away Team Win'
    return 'DRAW'


# %%
matches['outcomes'] = matches.apply(lambda x: get_labels(x), axis = 1)

# %%
matches.head()

# %%
mt = matches['outcomes'].value_counts()
mt

# %%
plt.figure(figsize = (6, 6))

mt.plot.pie(autopct = "%1.0f%%", colors = sns.color_palette('winter_r'), shadow = True)

c = plt.Circle((0, 0), 0.4, color = 'white')
plt.gca().add_artist(c)
plt.title('Match Outcomes by Home and Away Teams')
plt.show()
