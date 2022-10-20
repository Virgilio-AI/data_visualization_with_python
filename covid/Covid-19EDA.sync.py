# %% [markdown]
# # Covid-19 Exploratory Data Analysis 

# %% [markdown]
# ## Covid-19 Dataset Understanding 

# %% [markdown]
# Preprocessed Dataset Link: https://github.com/laxmimerit/Covid-19-Preprocessed-Dataset

# %%
import kaleido

# %%
# imports
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

import folium


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline

import math
import random
from datetime import timedelta

import warnings
warnings.filterwarnings('ignore')

#color pallette
cnf = '#393e46'
dth = '#ff2e63'
rec = '#21bf73'
act = '#fe9801'

# %% [markdown]
# # Dataset Preparation 

# %%
import plotly as py
py.offline.init_notebook_mode(connected = True)

# %%
df = pd.read_csv('Covid-19-Preprocessed-Dataset/preprocessed/covid_19_data_cleaned.csv', parse_dates=['Date'])

country_daywise = pd.read_csv('Covid-19-Preprocessed-Dataset/preprocessed/country_daywise.csv', parse_dates=['Date'])
countywise = pd.read_csv('Covid-19-Preprocessed-Dataset/preprocessed/countrywise.csv')
daywise = pd.read_csv('Covid-19-Preprocessed-Dataset/preprocessed/daywise.csv', parse_dates=['Date'])

# %%
df['Province/State'] = df['Province/State'].fillna("")
df.head()

# %%
confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()
recovered = df.groupby('Date').sum()['Recovered'].reset_index()
deaths = df.groupby('Date').sum()['Deaths'].reset_index()
deaths.head()

# %%

# %%
df.isnull().sum()

# %%
df.info()

# %%
df.query('Country == "US"')

# %% [markdown]
# # Worldwide Total Confirmed, Recovered, and Deaths 

# %%
confirmed.tail()

# %%
recovered.tail()

# %%
deaths.tail()

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x = confirmed['Date'], y = confirmed['Confirmed'], mode = 'lines+markers', name = 'Confirmed', line = dict(color = "Orange", width = 2)))
fig.add_trace(go.Scatter(x = recovered['Date'], y = recovered['Recovered'], mode = 'lines+markers', name = 'Recovered', line = dict(color = "Green", width = 2)))
fig.add_trace(go.Scatter(x = deaths['Date'], y = deaths['Deaths'], mode = 'lines+markers', name = 'Deaths', line = dict(color = "Red", width = 2)))
fig.update_layout(title = 'Worldwide Covid-19 Cases', xaxis_tickfont_size = 14, yaxis = dict(title = 'Number of Cases'))

fig.show()

# %% [markdown]
# # Cases Density Animation on World Map

# %%
df.info()

# %%
df['Date'] = df['Date'].astype(str)

# %%
df.info()

# %%
df.head()

# %%
fig = px.density_mapbox(df, lat = 'Lat', lon = 'Long', hover_name = 'Country', hover_data = ['Confirmed', 'Recovered', 'Deaths'], animation_frame='Date', color_continuous_scale='Portland', radius = 7, zoom = 0, height= 700)
fig.update_layout(title = 'Worldwide Covid-19 Cases with Time Laps')
fig.update_layout(mapbox_style = 'open-street-map', mapbox_center_lon = 0)


fig.show()

# %%

# %% [markdown]
# # Total Cases on Ships 

# %%
df['Date'] = pd.to_datetime(df['Date'])
df.info()

# %%
# Ships
# =====================

ship_rows = df['Province/State'].str.contains('Grand Princess') | df['Province/State'].str.contains('Diamond Princess') | df['Country'].str.contains('Grand Princess') | df['Country'].str.contains('Diamond Princess') | df['Country'].str.contains('MS Zaandam')
ship = df[ship_rows]

df = df[~ship_rows]

# %%
ship_latest = ship[ship['Date'] == max(ship['Date'])]
ship_latest

# %%
ship_latest.style.background_gradient(cmap = 'Pastel1_r')

# %%

# %%

# %% [markdown]
# # Cases Over the Time with Area Plot

# %%
temp = df.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()
temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop = True)

tm = temp.melt(id_vars = 'Date', value_vars = ['Active', 'Deaths', 'Recovered'])
fig = px.treemap(tm, path = ['variable'], values = 'value', height = 250, width = 800, color_discrete_sequence=[act, rec, dth])

fig.data[0].textinfo = 'label+text+value'
fig.show()

# %%
temp = df.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()
temp = temp.melt(id_vars = 'Date', value_vars = ['Recovered', 'Deaths', 'Active'], var_name = 'Case', value_name = 'Count')

fig = px.area(temp, x = 'Date', y = 'Count', color= 'Case', height = 400, title = 'Cases over time', color_discrete_sequence=[rec, dth,  act])
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()

# %%

# %% [markdown]
# # Folium Maps 

# %%
# Worldwide Cases on Folium Maps

# %%
temp = df[df['Date']==max(df['Date'])]

m = folium.Map(location=[0, 0], tiles='cartodbpositron', min_zoom = 1, max_zoom=4, zoom_start=1)

for i in range(0, len(temp)):
    folium.Circle(location=[temp.iloc[i]['Lat'], temp.iloc[i]['Long']], color = 'crimson', fill = 'crimson',
                 tooltip = '<li><bold> Country: ' + str(temp.iloc[i]['Country'])+
                            '<li><bold> Province: ' + str(temp.iloc[i]['Province/State'])+
                             '<li><bold> Confirmed: ' + str(temp.iloc[i]['Confirmed'])+
                             '<li><bold> Deaths: ' + str(temp.iloc[i]['Deaths']),
                 radius = int(temp.iloc[i]['Confirmed'])**0.5).add_to(m)
    

m

# %%

# %% [markdown]
# # Confirmed Cases with Choropleth Map

# %%
country_daywise.head()

# %%
fig = px.choropleth(country_daywise, locations= 'Country', locationmode='country names', color = np.log(country_daywise['Confirmed']),
                   hover_name = 'Country', animation_frame=country_daywise['Date'].dt.strftime('%Y-%m-%d'),
                   title='Cases over time', color_continuous_scale=px.colors.sequential.Inferno)

fig.update(layout_coloraxis_showscale = True)
fig.show()

# %%
fig = px.choropleth(country_daywise, locations= 'Country', locationmode='country names', color = country_daywise['Confirmed'],
                   hover_name = 'Country', animation_frame=country_daywise['Date'].dt.strftime('%Y-%m-%d'),
                   title='Cases over time', color_continuous_scale=px.colors.sequential.Inferno)

fig.update(layout_coloraxis_showscale = True)
fig.show()

# %% [markdown]
# # Confirmed and Death Cases with Static Colormap 

# %%
fig_c = px.choropleth(countywise, locations='Country', locationmode='country names',
                     color = np.log(countywise['Confirmed']), hover_name = 'Country',
                     hover_data = ['Confirmed'])

temp = countywise[countywise['Deaths']>0]
fig_d = px.choropleth(temp, locations='Country', locationmode='country names',
                     color = np.log(temp['Deaths']), hover_name = 'Country',
                     hover_data = ['Deaths'])

fig = make_subplots(rows = 1, cols = 2, subplot_titles= ['Confirmed', 'Deaths'],
                   specs = [[{'type': 'choropleth'}, {'type': 'choropleth'}]])

fig.add_trace(fig_c['data'][0], row = 1, col = 1)
fig.add_trace(fig_d['data'][0], row = 1, col = 2)

fig.update(layout_coloraxis_showscale = False)

fig.show()

# %% [markdown]
# # Deaths and Recoveries per 100 Cases 

# %%
daywise.head()

# %%
fig_c = px.bar(daywise, x = 'Date', y = 'Confirmed', color_discrete_sequence=[act])
fig_d = px.bar(daywise, x = 'Date', y = 'Deaths', color_discrete_sequence=[dth])

fig = make_subplots(rows = 1, cols = 2, shared_xaxes=False, horizontal_spacing=0.1,
                   subplot_titles=('Confirmed Cases', 'Deaths Cases'))

fig.add_trace(fig_c['data'][0], row = 1, col = 1)
fig.add_trace(fig_d['data'][0], row = 1, col = 2)

fig.update_layout(height = 400)

fig.show()

# %%
daywise.columns

# %%
fig1 = px.line(daywise, x = 'Date', y = 'Deaths / 100 Cases', color_discrete_sequence=[dth])
fig2 = px.line(daywise, x = 'Date', y = 'Recovered / 100 Cases', color_discrete_sequence=[rec])
fig3 = px.line(daywise, x = 'Date', y = 'Deaths / 100 Recovered', color_discrete_sequence=['aqua'])

fig = make_subplots(rows = 1, cols = 3, shared_xaxes=False,
                   subplot_titles=('Deaths / 100 Cases', 'Recovered / 100 Cases', 'Deaths / 100 Recovered'))

fig.add_trace(fig1['data'][0], row = 1, col = 1)
fig.add_trace(fig2['data'][0], row = 1, col = 2)
fig.add_trace(fig3['data'][0], row = 1, col = 3)

fig.update_layout(height = 400)
fig.show()

# %%

# %% [markdown]
# # New Cases and No. of Countries 

# %%
fig_c = px.bar(daywise, x = 'Date', y = 'Confirmed', color_discrete_sequence=[act])
fig_d = px.bar(daywise, x = 'Date', y = 'No. of Countries', color_discrete_sequence=[dth])

fig = make_subplots(rows = 1, cols = 2, shared_xaxes=False, horizontal_spacing=0.1,
                   subplot_titles=('No. of New Cases per Day', 'No. of Countries'))

fig.add_trace(fig_c['data'][0], row = 1, col = 1)
fig.add_trace(fig_d['data'][0], row = 1, col = 2)

fig.show()

# %% [markdown]
# # Top 15 Countries Case Analysis

# %%
countywise.columns

# %%
countywise.head()

# %%
top  = 15

fig_c = px.bar(countywise.sort_values('Confirmed').tail(top), x = 'Confirmed', y = 'Country',
              text = 'Confirmed', orientation='h', color_discrete_sequence=[act])
fig_d = px.bar(countywise.sort_values('Deaths').tail(top), x = 'Deaths', y = 'Country',
              text = 'Deaths', orientation='h', color_discrete_sequence=[dth])


fig_a = px.bar(countywise.sort_values('Active').tail(top), x = 'Active', y = 'Country',
              text = 'Active', orientation='h', color_discrete_sequence=['#434343'])
fig_r = px.bar(countywise.sort_values('Recovered').tail(top), x = 'Recovered', y = 'Country',
              text = 'Recovered', orientation='h', color_discrete_sequence=[rec])


fig_dc = px.bar(countywise.sort_values('Deaths / 100 Cases').tail(top), x = 'Deaths / 100 Cases', y = 'Country',
              text = 'Deaths / 100 Cases', orientation='h', color_discrete_sequence=['#f84351'])
fig_rc = px.bar(countywise.sort_values('Recovered / 100 Cases').tail(top), x = 'Recovered / 100 Cases', y = 'Country',
              text = 'Recovered / 100 Cases', orientation='h', color_discrete_sequence=['#a45398'])


fig_nc = px.bar(countywise.sort_values('New Cases').tail(top), x = 'New Cases', y = 'Country',
              text = 'New Cases', orientation='h', color_discrete_sequence=['#f04341'])
temp = countywise[countywise['Population']>1000000]
fig_p = px.bar(temp.sort_values('Cases / Million People').tail(top), x = 'Cases / Million People', y = 'Country',
              text = 'Cases / Million People', orientation='h', color_discrete_sequence=['#b40398'])



fig_wc = px.bar(countywise.sort_values('1 week change').tail(top), x = '1 week change', y = 'Country',
              text = '1 week change', orientation='h', color_discrete_sequence=['#c04041'])
temp = countywise[countywise['Confirmed']>100]
fig_wi = px.bar(temp.sort_values('1 week % increase').tail(top), x = '1 week % increase', y = 'Country',
              text = '1 week % increase', orientation='h', color_discrete_sequence=['#b00398'])


fig = make_subplots(rows = 5, cols = 2, shared_xaxes=False, horizontal_spacing=0.2, 
                    vertical_spacing=.05,
                   subplot_titles=('Confirmed Cases', 'Deaths Reported', 'Recovered Cases', 'Active Cases',
                                  'Deaths / 100 Cases', 'Recovered / 100 Cases',
                                  'New Cases', 'Cases / Million People',
                                  '1 week change', '1 week % increase'))

fig.add_trace(fig_c['data'][0], row = 1, col = 1)
fig.add_trace(fig_d['data'][0], row = 1, col = 2)

fig.add_trace(fig_r['data'][0], row = 2, col = 1)
fig.add_trace(fig_a['data'][0], row = 2, col = 2)

fig.add_trace(fig_dc['data'][0], row = 3, col = 1)
fig.add_trace(fig_rc['data'][0], row = 3, col = 2)

fig.add_trace(fig_nc['data'][0], row = 4, col = 1)
fig.add_trace(fig_p['data'][0], row = 4, col = 2)

fig.add_trace(fig_wc['data'][0], row = 5, col = 1)
fig.add_trace(fig_wi['data'][0], row = 5, col = 2)

fig.update_layout(height = 3000)
fig.show()

# %% [markdown]
# # Save Static Plots 

# %% [markdown]
# # Scatter Plot for Deaths vs Confirmed Cases 

# %%
top = 15
fig = px.scatter(countywise.sort_values('Deaths', ascending = False).head(top), 
                x = 'Confirmed', y = 'Deaths', color = 'Country', size = 'Confirmed', height = 600,
                text = 'Country', log_x = True, log_y = True, title='Deaths vs Confirmed Cases (Cases are on log10 scale)')

fig.update_traces(textposition = 'top center')
fig.update_layout(showlegend = False)
fig.update_layout(xaxis_rangeslider_visible = True)
fig.show()

# %%
countywise.sort_values('Deaths', ascending = False).head(15)

# %%

# %% [markdown]
# # Confirmed, Deaths, New Cases vs Country and Date

# %% [markdown]
# ## Bar Plot

# %%
country_daywise.head()

# %%
# get all individual values of a column and the count of each value
country_daywise['Country'].value_counts()

# %%
country_daywise.iloc[996:1000]

# %%

from typing import *
import heapq as hp
from collections import deque
from collections import defaultdict
import sys



# %%
dic = defaultdict(int)
step = 997
arr = []
arr_deaths = []
for i in range(996,len(country_daywise),step):
	print(str(country_daywise.loc[i,"Country"]) + " " + str(country_daywise.loc[i,"Confirmed"]))
	arr.append([country_daywise.loc[i,"Country"],country_daywise.loc[i,"Confirmed"]])
	arr_deaths.append([country_daywise.loc[i,"Country"],country_daywise.loc[i,"Deaths"]])


arr.sort(key = lambda x: x[1], reverse = True)
confirmed = arr[:10]
set_confirmed = set()
for i in range(len(confirmed)):
	set_confirmed.add(confirmed[i][0])


arr_deaths.sort(key = lambda x: x[1], reverse = True)
deaths = arr_deaths[:10]
set_deaths = set()
for i in range(len(deaths)):
	set_deaths.add(deaths[i][0])

# %%
set_confirmed.add("Mexico")

# %%
country_daywise_deaths = country_daywise[country_daywise['Country'].isin(set_deaths)]
country_daywise_confirmed = country_daywise[country_daywise['Country'].isin(set_confirmed)]

# %%
fig = px.bar(country_daywise_confirmed, x = 'Date', y = 'Confirmed', color = 'Country', height = 600,
            title='Confirmed')
fig.show()

# %%
fig = px.bar(country_daywise_deaths, x = 'Date', y = 'Deaths', color = 'Country', height = 600,
            title='Deaths')
fig.show()

# %%
# fig = px.bar(country_daywise, x = 'Date', y = 'Recovered', color = 'Country', height = 600,
#             title='Recovered')
# fig.show()
# 
# # %%
# fig = px.bar(country_daywise, x = 'Date', y = 'New Cases', color = 'Country', height = 600,
#             title='New Cases')
# fig.show()

# %%

# %%

# %%

# %% [markdown]
# ## Line Plot 

# %%
fig = px.line(country_daywise, x = 'Date', y = 'Confirmed', color = 'Country', height = 600,
             title='Confirmed', color_discrete_sequence = px.colors.cyclical.mygbm)
fig.show()

fig = px.line(country_daywise, x = 'Date', y = 'Deaths', color = 'Country', height = 600,
             title='Deaths', color_discrete_sequence = px.colors.cyclical.mygbm)
fig.show()

fig = px.line(country_daywise, x = 'Date', y = 'Recovered', color = 'Country', height = 600,
             title='Recovered', color_discrete_sequence = px.colors.cyclical.mygbm)
fig.show()

# %%

# %% [markdown]
# # Growth Rate after 100 Cases 

# %%
df.head()

# %%
gt_100 = country_daywise[country_daywise['Confirmed']>100]['Country'].unique()
temp = df[df['Country'].isin(gt_100)]

temp = temp.groupby(['Country', 'Date'])['Confirmed'].sum().reset_index()
temp = temp[temp['Confirmed']>100]


min_date = temp.groupby('Country')['Date'].min().reset_index()
min_date.columns = ['Country', 'Min Date']


from_100th_case = pd.merge(temp, min_date, on = 'Country')
from_100th_case['N days'] = (from_100th_case['Date'] - from_100th_case['Min Date']).dt.days

fig = px.line(from_100th_case, x = 'N days', y = 'Confirmed', color = 'Country', title = 'N days from 100 case', height = 600)
fig.show()

# %%

# %% [markdown]
# # Growth Rate after 1000 Cases 

# %%
gt_1000 = country_daywise[country_daywise['Confirmed']>1000]['Country'].unique()
temp = df[df['Country'].isin(gt_1000)]

temp = temp.groupby(['Country', 'Date'])['Confirmed'].sum().reset_index()
temp = temp[temp['Confirmed']>1000]


min_date = temp.groupby('Country')['Date'].min().reset_index()
min_date.columns = ['Country', 'Min Date']


from_1000th_case = pd.merge(temp, min_date, on = 'Country')
from_1000th_case['N days'] = (from_1000th_case['Date'] - from_1000th_case['Min Date']).dt.days

fig = px.line(from_1000th_case, x = 'N days', y = 'Confirmed', color = 'Country', title = 'N days from 1000 case', height = 600)
fig.show()

# %% [markdown]
# # Growth Rate after 10,000 Cases

# %%
gt_10000 = country_daywise[country_daywise['Confirmed']>10000]['Country'].unique()
temp = df[df['Country'].isin(gt_10000)]

temp = temp.groupby(['Country', 'Date'])['Confirmed'].sum().reset_index()
temp = temp[temp['Confirmed']>10000]


min_date = temp.groupby('Country')['Date'].min().reset_index()
min_date.columns = ['Country', 'Min Date']


from_10000th_case = pd.merge(temp, min_date, on = 'Country')
from_10000th_case['N days'] = (from_10000th_case['Date'] - from_10000th_case['Min Date']).dt.days

fig = px.line(from_10000th_case, x = 'N days', y = 'Confirmed', color = 'Country', title = 'N days from 10000 case', height = 600)
fig.show()

# %% [markdown]
# # Growth Rate After 100k Cases 

# %%
gt_100000 = country_daywise[country_daywise['Confirmed']>100000]['Country'].unique()
temp = df[df['Country'].isin(gt_100000)]

temp = temp.groupby(['Country', 'Date'])['Confirmed'].sum().reset_index()
temp = temp[temp['Confirmed']>100000]


min_date = temp.groupby('Country')['Date'].min().reset_index()
min_date.columns = ['Country', 'Min Date']


from_100000th_case = pd.merge(temp, min_date, on = 'Country')
from_100000th_case['N days'] = (from_100000th_case['Date'] - from_100000th_case['Min Date']).dt.days

fig = px.line(from_100000th_case, x = 'N days', y = 'Confirmed', color = 'Country', title = 'N days from 100000 case', height = 600)
fig.show()

# %%

# %% [markdown]
# # Tree Map Analysis

# %% [markdown]
# ## Confirmed Cases

# %%
full_latest = df[df['Date'] == max(df['Date'])]

fig = px.treemap(full_latest.sort_values(by = 'Confirmed', ascending = False).reset_index(drop = True),
                path = ['Country', 'Province/State'], values = 'Confirmed', height = 700,
                title = 'Number of Confirmed Cases',
                color_discrete_sequence = px.colors.qualitative.Dark2)

fig.data[0].textinfo = 'label+text+value'
fig.show()

# %% [markdown]
# ## Deaths Cases

# %%
full_latest = df[df['Date'] == max(df['Date'])]

fig = px.treemap(full_latest.sort_values(by = 'Deaths', ascending = False).reset_index(drop = True),
                path = ['Country', 'Province/State'], values = 'Deaths', height = 700,
                title = 'Number of Deaths Cases',
                color_discrete_sequence = px.colors.qualitative.Dark2)

fig.data[0].textinfo = 'label+text+value'
fig.show()

# %% [markdown]
# # First and Last Case Report Time

# %%
first_date = df[df['Confirmed']>0]
first_date = first_date.groupby('Country')['Date'].agg(['min']).reset_index()


last_date = df.groupby(['Country', 'Date'])['Confirmed', 'Deaths', 'Recovered']
last_date = last_date.sum().diff().reset_index()


mask = (last_date['Country'] != last_date['Country'].shift(1))

last_date.loc[mask, 'Confirmed'] = np.nan
last_date.loc[mask, 'Deaths'] = np.nan
last_date.loc[mask, 'Recovered'] = np.nan

last_date = last_date[last_date['Confirmed']>0]
last_date = last_date.groupby('Country')['Date'].agg(['max']).reset_index()


first_last = pd.concat([first_date, last_date['max']], axis = 1)
first_last['max'] = first_last['max'] + timedelta(days = 1)

first_last['Days'] = first_last['max'] - first_last['min']
first_last['Task'] = first_last['Country']

first_last.columns = ['Country', 'Start', 'Finish', 'Days', 'Task']

first_last = first_last.sort_values('Days')

colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(first_last))]

fig = ff.create_gantt(first_last, index_col = 'Country', colors = colors, show_colorbar = False,
                     bar_width=0.2, showgrid_x = True, showgrid_y=True, height = 2500)

fig.show()

# %% [markdown]
# # Confirmed Cases Country and Day wise

# %%
country_daywise.head()

# %%
# install tqdm for progress bar
# !pip install tqdm

# %%
# import tqdm for progress bar
from tqdm import tqdm

# %%
temp = country_daywise.groupby(['Country', 'Date'])['Confirmed'].sum().reset_index()
temp = temp[temp['Country'].isin(gt_10000)]

countries = temp['Country'].unique()

ncols = 3
nrows = math.ceil(len(countries)/ncols)

fig = make_subplots(rows=nrows, cols = ncols, shared_xaxes= False, subplot_titles=countries)

counter = 0
maxi = 10
for ind, country in tqdm(enumerate(countries)):
	if maxi == counter:
		break
	counter +=1
	row = int((ind/ncols)+1)
	col = int((ind%ncols)+1)
	fig.add_trace(go.Bar(x = temp['Date'], y = temp.loc[temp['Country']==country, 'Confirmed'], name = country), row = row, col = col)

fig.update_layout(height=4000, title_text = 'Confirmed Cases in Each Country')
fig.update_layout(showlegend = False)
fig.show()

# %% [markdown]
# # Covid-19 vs Other Similar Epidemics

# %%
# Wikipedia Source

epidemics = pd.DataFrame({
    'epidemic' : ['COVID-19', 'SARS', 'EBOLA', 'MERS', 'H1N1'],
    'start_year' : [2019, 2002, 2013, 2012, 2009],
    'end_year' : [2020, 2004, 2016, 2020, 2010],
    'confirmed' : [full_latest['Confirmed'].sum(), 8422, 28646, 2519, 6724149],
    'deaths' : [full_latest['Deaths'].sum(), 813, 11323, 866, 19654]
})

epidemics['mortality'] = round((epidemics['deaths']/epidemics['confirmed'])*100, 2)

epidemics.head()

# %%
temp = epidemics.melt(id_vars='epidemic', value_vars=['confirmed', 'deaths', 'mortality'],
                     var_name='Case', value_name='Value')

fig = px.bar(temp, x = 'epidemic', y = 'Value', color = 'epidemic', text = 'Value', facet_col = 'Case',
            color_discrete_sequence= px.colors.qualitative.Bold)

fig.update_traces(textposition='outside')
fig.update_layout(uniformtext_minsize = 8, uniformtext_mode = 'hide')
fig.update_yaxes(showticklabels = False)
fig.layout.yaxis2.update(matches = None)
fig.layout.yaxis3.update(matches = None)
fig.show()
