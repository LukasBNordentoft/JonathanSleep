# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 19:44:37 2024
@authors: 
    Lukas B. Nordentoft
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import scipy

st.set_page_config(
     page_title='Jonathans Søvn',
     layout="wide",
     initial_sidebar_state="expanded",
     page_icon='👨‍👶'
)

# Read sheets data
sheet_id = '1TkpiDscz__S_MD9TO5PzVmGlVzlGiLRRP-zP225hDt8'
data = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv")

def set_plot_options():
    import matplotlib.pyplot as plt
    import matplotlib
    color_bg      = "0.99"          #Choose background color
    color_gridaxe = "0.85"          #Choose grid and spine color
    rc = {"axes.edgecolor":color_gridaxe} 
    plt.style.use(('ggplot', rc))           #Set style with extra spines
    plt.rcParams['figure.dpi'] = 300        #Set resolution
    plt.rcParams['figure.figsize'] = [10, 5]
    matplotlib.rc('font', size=15)
    matplotlib.rc('axes', titlesize=20)
    matplotlib.rcParams['font.family'] = ['DejaVu Sans']     #Change font to Computer Modern Sans Serif
    plt.rcParams['axes.unicode_minus'] = False          #Re-enable minus signs on axes))
    plt.rcParams['axes.facecolor']= color_bg             #Set plot background color
    plt.rcParams.update({"axes.grid" : True, "grid.color": color_gridaxe}) #Set grid color
    plt.rcParams['axes.grid'] = True
    # plt.fontname = "Computer Modern Serif"

set_plot_options()

#%% Format data
# Set date as index
data['dato'] = pd.to_datetime(data['dato'], format='%d/%m/%Y')
data.set_index('dato', inplace=True)
data.index = data.index.date

time_columns = ['Vågnede', '1. Lur Start', '1. Lur Slut', '2. Lur Start', '2. Lur Slut', 'Sov']

# Looop through and convert each element to datetime
for column in time_columns:
    for row in data.index:
        data[column][row] = datetime.combine(row, datetime.strptime(data[column][row], '%H:%M').time())

max_rows = len(data.index)

#%% Pull Streamlit ineractivity and limit data
st.sidebar.header('Juster data')
st.sidebar.write('Indtast eller ret i data:')
st.sidebar.link_button('Gå til Google Sheet', 'https://docs.google.com/spreadsheets/d/1TkpiDscz__S_MD9TO5PzVmGlVzlGiLRRP-zP225hDt8/edit?gid=0#gid=0')

st.sidebar.write('')
chosen_days = st.sidebar.slider('Vælg antal dage til analyse.',
                                max_value = max_rows, value = max_rows)

data = data.iloc[-chosen_days:]
#%% Sleep durations

durations = pd.DataFrame(index = data.index)

durations['1. Lur'] = data['1. Lur Slut'] - data['1. Lur Start']
durations['2. Lur'] = data['2. Lur Slut'] - data['2. Lur Start']

sleep_durations = pd.Series(index=durations.index)

# Loop through each row and calculate time asleep at night
for i in range(len(sleep_durations.index)):
    if i == 0:
        sleep_durations[i] = timedelta(hours=10)
    else:
        sleep_durations[i] = data['Vågnede'].iloc[i] - data['Sov'].iloc[i-1]

durations['Nat']        = sleep_durations
durations['Søvn i alt'] = durations['1. Lur'] + durations['2. Lur'] + durations['Nat']


durations_stats = durations.copy()
durations_stats['1. Lur'] = pd.to_timedelta(durations_stats['1. Lur'].astype(str))
durations_stats['2. Lur'] = pd.to_timedelta(durations_stats['2. Lur'].astype(str))
durations_stats['Nat'] = pd.to_timedelta(durations_stats['Nat'].astype(str))
durations_stats['Søvn i alt'] = pd.to_timedelta(durations_stats['Søvn i alt'].astype(str))

durations_stats_display = durations_stats.mean()
durations_stats_display.name = "Søvn [hr]"

#%% Plot One

# Convert times to hours since midnight
def time_to_hours(dt):
    return dt.hour + dt.minute / 60.0

data_hours = data[time_columns].applymap(lambda x: time_to_hours(x))
data_hours['Midnat'] = 24
# data_hours = data_hours.iloc[0:chosen_days,:]
data_hours_diff = data_hours.copy()

for i in range(len(data_hours.columns)):
    if i == 0:
        pass
    else:
        col     = data_hours.columns[i]
        pre_col = data_hours.columns[i-1]
        
        data_hours_diff[col] = data_hours[col] - data_hours[pre_col]

# data_hours_diff['Sum'] = data_hours_diff.sum(axis = 1)
data_hours_diff.columns = ['Nat morgen', 'Vågen 1',
                         '1. Lur', 'Vågen 2',
                         '2. Lur', 'Vågen 3', 
                         'Nat aften']
colors = ['cornflowerblue', '0.99', 
          'Gold', '0.99',
          'Orange', '0.99',
          'cornflowerblue']

data_hours_diff = data_hours_diff[::-1]

ax = data_hours_diff.plot.barh(stacked = True,
                               color = colors, figsize = (20, 8*(chosen_days/max_rows)))

ax.axvline(x=data_hours['1. Lur Start'].median(), color='Gold', linestyle='--', linewidth=2)
ax.axvline(x=data_hours['1. Lur Slut'].median(), color='Gold', linestyle='--', linewidth=2)
ax.axvline(x=data_hours['2. Lur Start'].median(), color='Orange', linestyle='--', linewidth=2)
ax.axvline(x=data_hours['2. Lur Slut'].median(), color='Orange', linestyle='--', linewidth=2)
ax.axvline(x=data_hours['Vågnede'].median(), color='cornflowerblue', linestyle='--', linewidth=2)
ax.axvline(x=data_hours['Sov'].median(), color='cornflowerblue', linestyle='--', linewidth=2)


from matplotlib.lines import Line2D
lw1 = 10
lw2 = 4
legend_elements = [Line2D([0], [0], color='cornflowerblue', lw=lw1, label='Nat'),
                   Line2D([0], [0], color='cornflowerblue', linestyle = '--', 
                          lw=lw2, label='Vågner'),
                   Line2D([0], [0], color='Gold', lw=lw1, label='1. Lur'),
                   Line2D([0], [0], color='Gold', linestyle = '--', 
                          lw=lw2, label='1. Lur Start/Slut'),
                   Line2D([0], [0], color='Orange', lw=lw1, label='2. Lur'),
                   Line2D([0], [0], color='Orange', linestyle = '--', 
                          lw=lw2, label='2. Lur Start/Slut'),
                   Line2D([0], [0], color='0.99', lw=lw1, label='Vågen'),
                   Line2D([0], [0], color='cornflowerblue', linestyle = '--', 
                          lw=lw2, label='Sover'),
                  ]

fig = ax.figure
offset = 0.8
bbox_y = -offset / fig.get_figheight()
ax.legend(handles = legend_elements,
          loc='upper center', bbox_to_anchor=(0.5, bbox_y), ncol=4)

ax.set_xlabel('Tid på dagen [hr]')
ax.set_xlim([0, 24])
ax.set_xticks(np.arange(0, 25))
ax.xaxis.set_tick_params(labeltop=True)
# ax.set_title('Jonathans Døgnrytme', fontsize=35, pad=20)

total_sleep = data_hours_diff[['Nat morgen', '1. Lur', '2. Lur', 'Nat aften']].sum(axis=1)
total_strings = [f"         {total:.1f}" for total in total_sleep.iloc[:-1]] + [f" I alt: {total_sleep.iloc[-1]:.1f} timer"]
ax.bar_label(ax.containers[6], labels = total_strings)

#%% Plot three

import plotly.figure_factory as ff

dist1 = ff.create_distplot([data_hours_diff['1. Lur']], ['1. Lur længder'],
                           bin_size = 0.25)

#%% Format dataframes to contain strings for display

# Format data
data_str = data[time_columns].applymap(lambda x: x.strftime("%H:%M"))
data_str['Noter'] = data['Noter']

# Format durations
def timedelta_to_str(td):
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes = remainder // 60
    return f"{int(hours):02}:{int(minutes):02}"

durations_str = durations.applymap(timedelta_to_str)

durations_stats_display = durations_stats_display.apply(timedelta_to_str)

#%% Correlations

import seaborn as sns
import matplotlib.pyplot as plt

data_for_corr = pd.DataFrame(index = data_hours_diff.index)

data_for_corr[['1. Lur', '2. Lur']] = data_hours_diff[['1. Lur', '2. Lur']]
data_for_corr['Nat'] = data_hours_diff['Nat aften'] + data_hours_diff['Nat morgen']
data_corr = data_for_corr.corr()

fig4, ax4 = plt.subplots(nrows = 1, ncols = 2, figsize=(20, 5),
                         gridspec_kw={'width_ratios': [1, 2]})
sns.heatmap(data_corr, ax=ax4[0], cmap='Blues', annot = True)

#%% Streamlit

tab1, tab2 = st.tabs(['Data Visualiseret', 'Data i tal'])

with tab1:
    # Sidebar:
    st.sidebar.header('Overblik')
    st.sidebar.write('Jonathans gennemsnitlige søvn')
    st.sidebar.dataframe(durations_stats_display)
    
    st.header('Jonathans døgnrytme visualiseret')
    st.write('Jonathans nat, 1. lur og 2. lur hver dag, med median start/slut tider (stiplede linjer)')
    st.pyplot(fig)
    st.header('Korrelationer og fordelinger')
    st.write('Korrelationer mellem søvnperioder, samt fordelinger for søvnperioderne.')
    st.pyplot(fig4)

with tab2:
    st.header('Data i tal')
    col1, col2 = st.columns(2)
    
    with col1:
        st.write('Data:')
        st.dataframe(data_str)
    
    with col2:
        st.write('Durations:')
        st.dataframe(durations_str)
    st.divider()















