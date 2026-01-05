# organising data

import pandas as pd

df_ft_02 = pd.read_csv('data/df_sum_0.2.csv')
df_ft_03 = pd.read_csv('data/df_sum_0.3.csv')
df_ft_05 = pd.read_csv('data/df_sum_0.5.csv')
df_ft_07 = pd.read_csv('data/df_sum_0.7.csv')

# read data
print(len(df_ft_07))
print(len(df_ft_05))
print(len(df_ft_03))
print(len(df_ft_02))

#%%
# clean data
# 4 features (leader_pv_id, leader_v, leader_r_dis, platoon_type), 1 target (tail_r_t)
# 80% training data, 20% test data
# combination 1, 4 features
def filter_data(df):
    df_filtered = df.loc[:, ['platoon_type', 'dis_leader_pv', 'leader_v', 'leader_r_dis', 'tail_arr_duration']]
    return df_filtered

# combination 2, 6 features
def filter_data2(df):
    df_filtered = df.loc[:, ['platoon_type', 'dis_leader_pv', 'leader_v', 'leader_r_dis', 'tail_arr_duration',
                             'tail_v', 'tail_r_dis']]
    return df_filtered

#%%
# clean, combine and shuffle
df_ft_02f = filter_data2(df_ft_02)
df_ft_03f = filter_data2(df_ft_03)
df_ft_05f = filter_data2(df_ft_05)
df_ft_07f = filter_data2(df_ft_07)

print(len(df_ft_02f))
print(len(df_ft_03f))
print(len(df_ft_05f))
print(len(df_ft_07f))

df_combined = pd.concat([df_ft_02f, df_ft_03f, df_ft_05f, df_ft_07f], ignore_index=True)
df_combined

#%%
# shift the format of platoon_type: from 'AHHH' to its length
df_combined_c = df_combined.copy()
df_combined_c['platoon_type'] = df_combined_c['platoon_type'].str.len()
df_combined_c
df_combined_c.to_csv('data/df_combined_c_4f_102124.csv', index=False)

#%% plot 2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Extracting columns for plotting
x = df_combined_c['dis_leader_pv']
y = df_combined_c['leader_v']
z = df_combined_c['leader_r_dis']
c = df_combined_c['tail_arr_duration']  # Color intensity
s = df_combined_c['platoon_type'] # Color intensity

# Create 3D scatter plot
fig = plt.figure(dpi=600)
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with color based on tail_arr_duration
sc = ax.scatter(x, y, s, s=0.2, c=c, cmap='coolwarm')

# Adding color bar
color_bar = plt.colorbar(sc, ax=ax, shrink=0.7, aspect=15, location='left')
color_bar.set_label('Tail Arrival Duration (s)')

# Labeling axes
ax.set_xlabel('dis_leader_pv (m)')
ax.set_ylabel('leader_v (m/s)')
ax.set_zlabel('leader_r_dis (m)')

# Show plot
plt.show()

#%% ploty
import plotly.express as px
# Create 3D scatter plot with Plotly
import plotly.io as pio
pio.renderers.default = 'browser'
fig = px.scatter_3d(df_combined_c,
                    x='dis_leader_pv',
                    y='leader_v',
                    z='leader_r_dis',
                    color='tail_arr_duration',  # Color based on duration
                    # size='platoon_type',        # Point size based on platoon_type
                    # color_continuous_scale='coolwarm',
                    labels={
                        'dis_leader_pv': 'dis_leader_pv (m)',
                        'leader_v': 'leader_v (m/s)',
                        'leader_r_dis': 'leader_r_dis (m)',
                        'tail_arr_duration': 'Tail Arrival Duration (s)'
                    })

# Show plot
fig.show()