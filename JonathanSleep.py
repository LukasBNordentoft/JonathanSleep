# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 19:44:37 2024
@authors: 
    Lukas B. Nordentoft, lbn@mpe.au.dk
    Anders L. Andreasen, ala@mpe.au.dk
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(
     page_title='Jonathans Søvn',
     layout="wide",
     initial_sidebar_state="expanded",
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

st.sidebar.header('Juster plot')
chosen_days = st.sidebar.slider('Vælg antal dage til plot. Viser som udgangspunkt de seneste 7 dage. Dette påvirker også median udregning.',
                                max_value = len(data.index), value = 7)

max_rows = len(data.index)

data = data.iloc[0:chosen_days,:]
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
durations_stats_display.name = "Gennemsnitlige søvnlængder"

#%% Plot One

# Convert times to hours since midnight
def time_to_hours(dt):
    return dt.hour + dt.minute / 60.0

# st.sidebar.header('Juster plot')
# chosen_days = st.sidebar.slider('Vælg antal dage til plot. Viser som udgangspunkt de seneste 7 dage. Dette påvirker også median udregning.',
#                                 max_value = len(data.index), value = 7)

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
                          lw=lw2, label='Vågner (Median)'),
                   Line2D([0], [0], color='Gold', lw=lw1, label='1. Lur'),
                   Line2D([0], [0], color='Gold', linestyle = '--', 
                          lw=lw2, label='1. Lur Start/Slut (Median)'),
                   Line2D([0], [0], color='Orange', lw=lw1, label='2. Lur'),
                   Line2D([0], [0], color='Orange', linestyle = '--', 
                          lw=lw2, label='2. Lur Start/Slut (Median)'),
                   Line2D([0], [0], color='0.99', lw=lw1, label='Vågen'),
                   Line2D([0], [0], color='cornflowerblue', linestyle = '--', 
                          lw=lw2, label='Sover (Median)'),
                  ]
ax.legend(handles = legend_elements,
          loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)

ax.set_xlabel('Tid på dagen [hr]')
ax.set_xlim([0, 24])
ax.set_xticks(np.arange(0, 25))
ax.xaxis.set_tick_params(labeltop=True)
# ax.set_title('Jonathans Døgnrytme', fontsize=35, pad=20)
fig = ax.figure

#%% Plot two

# import plotly.express as px

# # fig2 = px.bar(data_hours_diff, barmode = 'stack')

# df = data_hours_diff.copy()
# df.index.name = 'Date'
# df.reset_index(inplace=True)

# df_melted = df.melt(id_vars = 'Date', var_name='Category', value_name='Value')

# category_order = ['Nat morgen', 'Vågen 1', '1. Lur', 'Vågen 2', '2. Lur', 'Vågen 3', 'Nat aften']

# custom_colors = {
#     'Nat morgen': 'cornflowerblue',
#     'Vågen 1': 'white',
#     '1. Lur': 'Gold',
#     'Vågen 2': 'white',
#     '2. Lur': 'Orange',
#     'Vågen 3': 'white',
#     'Nat aften': 'cornflowerblue'
# }

# fig2 = px.bar(df_melted, y = 'Date', x = 'Value', 
#               color = 'Category', color_discrete_map=custom_colors,
#               category_orders={'Category': category_order},
#               orientation='h')

# fig2.update_yaxes(tickmode='array', tickvals = df_melted['Date'].unique())
# fig2.update_xaxes(tickmode='array', tickvals = np.arange(0, 24))

# fig2.update_layout(
#     legend=dict(
#         orientation="h",
#         yanchor="top",
#         y=-0.3,  # Adjust the vertical position as needed
#         xanchor="center",
#         x=0.5,
#         title='Legend',
#         font=dict(size=10),  # Adjust font size if needed
#         traceorder="normal",  # Keep the legend in the original order
#         bgcolor='rgba(255,255,255,0)',  # Set background color of legend
#         itemsizing='constant'  # Keep items in a fixed size
#     ),
#     margin=dict(b=80),  # Adjust the bottom margin to fit the legend
#     width=2000,   # Set the desired width (in pixels)
#     height=400,  # Set the desired height (in pixels)
# )

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

#%% Streamlit

# Sidebar:
st.sidebar.header('Overblik')
st.sidebar.write('Overblik over Jonathans søvn i nøgletal')
st.sidebar.dataframe(durations_stats_display)

st.header('Jonathans døgnrytme')
st.write('Brug slider i sidebar til at justere antallet af dage der vises.')
st.pyplot(fig)

# st.plotly_chart(fig2)

# st.plotly_chart(fig_go)


st.header('Overblik over data i tal')
col1, col2 = st.columns(2)

with col1:
    st.write('Data:')
    st.dataframe(data_str)

with col2:
    st.write('Durations:')
    st.dataframe(durations_str)

#%%

# # Colors for each sleep period
# colors = ['royalblue', 'darkorange', 'green', 'purple']

# # Add the 24-hour timeline
# fig.add_trace(go.Scatter(x=list(range(25)), y=[i for i in range(len(data) + 1)], 
#                          mode='lines', line=dict(color='black'), showlegend=False))

# # Plot each day's data
# for i, row in data.iterrows():
#     # Convert times to hours since midnight
#     intervals = [
#         (time_to_hours(row['1. Lur Start']), time_to_hours(row['1. Lur Slut']), 'First Nap'),
#         (time_to_hours(row['2. Lur Start']), time_to_hours(row['2. Lur Slut']), 'Second Nap'),
#         (time_to_hours(row['Sov']), 24, 'Night Sleep Start'),
#     ]
    
#     # Plot each interval with a different color
#     for (start, end, label), color in zip(intervals, colors):
#         fig.add_trace(go.Scatter(
#             x=[start, end],
#             y=[i, i],
#             mode='lines',
#             line=dict(color=color, width=10),
#             name=f'{label} - {row["day"]}',
#             showlegend=True
#         ))

# # Customize layout
# fig.update_layout(
#     title="Child's Sleep Patterns Over Multiple Days",
#     xaxis=dict(
#         title='Hour of the Day',
#         tickmode='linear',
#         tick0=0,
#         dtick=1,
#         range=[0, 24]
#     ),
#     yaxis=dict(
#         title='Day',
#         tickvals=list(range(len(data))),
#         ticktext=[f'Day {i + 1}' for i in range(len(data))],
#         range=[-1, len(data)]
#     ),
#     showlegend=True,
#     height=400,
#     margin=dict(l=40, r=0, t=50, b=20)
# )














