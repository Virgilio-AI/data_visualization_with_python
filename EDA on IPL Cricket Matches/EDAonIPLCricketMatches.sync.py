# %% [markdown]
# # EDA on IPL Cricket Matches

# %% [markdown]
# Dataset: https://www.kaggle.com/nowke9/ipldata

# %% [markdown]
# What is IPL: Indian Premier League (IPL) is a Twenty20 cricket format league in India. It is usually played in April and May every year. As of 2019, the title sponsor of the game is Vivo. The league was founded by Board of Control for Cricket India (BCCI) in 2008.

# %% [markdown]
# ![image.png](attachment:image.png)

# %% [markdown]
# Source: https://en.wikipedia.org/wiki/Cricket

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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
df = pd.read_csv('matches.csv', index_col='id', parse_dates=['date'])
df.head()

# %%

# %% [markdown]
# # Win and Lost Team Analysis 

# %%
df.head(1)

# %%
df['result'].value_counts()

# %%
df[df['result'] == 'tie']

# %%
df[df['result'] == 'no result']

# %%

# %%
df['winner'].isnull().sum()

# %%
winslost = df[['team1', 'team2', 'winner']]
winslost.head()

# %%
winslost['looser'] = winslost.apply(lambda x: (x['team2'] if x['team1'] == x['winner'] else x['team1']), axis = 1)

# %%
winslost.head()

# %%
wins = winslost['winner'].value_counts()
loosers = winslost['looser'].value_counts()

# %%
wins.iplot(kind = 'bar', xTitle = 'Team', yTitle = 'Count', title = 'Winning Count')

# %%
loosers.iplot(kind = 'bar', xTitle = 'Team', yTitle = 'Count', title = 'Loosers Count')

# %%

# %%

# %%

# %% [markdown]
# # MoM and Citywise Analysis 

# %%
df.head(1)

# %%
mom = df['player_of_match'].value_counts()
mom

# %%
mom[:20].iplot(kind = 'bar', xTitle = 'Player', yTitle = 'Count', title = 'Top 20 Most MoM')

# %% [markdown]
# ## Matches Hosted in Each City 

# %%
city = df['city'].value_counts()

# %%
city[:20].iplot(kind = 'bar')

# %% [markdown]
# ## Matches Hosted at Venue 

# %%
stadium = df['venue'].value_counts()
stadium[:20].iplot(kind = 'bar')


# %%

# %% [markdown]
# ## MI vs CSK Head to Head 

# %%
def get_micsk(team1, team2):
    teams = ['Chennai Super Kings', 'Mumbai Indians']
    if team1 in teams and team2 in teams:
        return True
    else:
        return False


# %%
index = []
for row in df.iterrows():
    flag = get_micsk(row[1]['team1'], row[1]['team2'])
    index.append(flag)

# %%
sum(index)

# %%
micsk = df[index]
micsk.head()

# %%
micsk['toss_decision'].value_counts().iplot(kind = 'bar')

# %%
micsk['toss_winner'].value_counts().iplot(kind = 'bar')

# %%
micsk['winner'].value_counts().iplot(kind = 'bar')

# %%
micsk['player_of_match'].value_counts().iplot(kind = 'bar')

# %%
micsk.head()

# %%
temp = micsk[['winner', 'win_by_runs', 'win_by_wickets']]
temp = temp.set_index('winner')
temp.max()

# %%
temp.plot.bar(figsize = (15, 5), rot = 80)

# %%

# %%

# %% [markdown]
# ## Season wise Match Summary 

# %%
sns.catplot(x = 'season', y = 'win_by_runs', data = df, kind = 'swarm', height=4, aspect=3)

# %%
sns.catplot(x = 'season', y = 'win_by_wickets', data = df, kind = 'swarm', height=4, aspect=3)

# %%
season = df.groupby('season')[['win_by_runs']].max()

# %%
season

# %%
season.iplot(kind = 'bar')

# %% [markdown]
# # Ball by Ball Analysis 

# %%
df = pd.read_csv('deliveries.csv', index_col='match_id')
df.head()

# %%
df['batsman'].value_counts()[:19].iplot(kind = 'bar')

# %%
df['bowler'].value_counts()[:19].iplot(kind = 'bar')

# %%
df['non_striker'].value_counts()[:19].iplot(kind = 'bar')

# %%
df.columns

# %%
runs = df.groupby('batting_team').sum()[['batsman_runs', 'total_runs']]

# %%
runs

# %%
runs.iplot(kind = 'bar')

# %%
batsman = df.groupby('batsman')['batsman_runs'].sum()
batsman = batsman.sort_values(ascending = False)

# %%
batsman[:19].iplot(kind = 'bar')

# %%

# %%
df['player_dismissed'].value_counts()[:19].iplot(kind = 'bar')

# %%
df['dismissal_kind'].value_counts().iplot(kind = 'bar')

# %%
df['dismissal_kind'].value_counts()
