import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import correlate
from scipy.signal import correlation_lags
from sklearn.decomposition import PCA

# normalization function
def normalize(data):
    # calculate dataset mean and standard deviation
    mean = data.mean()
    std = data.std()

    # get column names
    col = data.columns

    # normalise dataset with previously calculated values
    data_norm =(data[col] - mean)/std

    return data_norm

filename = 'data/all_data.xlsx'
filename_kaub = 'data/merged_all.xlsx'

df = pd.read_excel(filename, sheet_name='Sheet1')
df_kaub = pd.read_excel(filename_kaub, sheet_name='Sheet1')

df['weather_date'] = pd.to_datetime(df['weather_date'])
## remove wrong data
df_kaub = df_kaub[~(df_kaub['datum'] > '2020-12-31')]
df_kaub = df_kaub[~(df_kaub['datum'] < '2004-10-24')]
df.dropna(inplace=True)

# set date as index
df.set_index('weather_date', inplace=True)

df_kaub = df_kaub[['datum','kaub_level']]
# df_kaub.set_index('datum', inplace=True)

df = df_kaub.merge(df, how='inner', left_on='datum', right_on='weather_date')

df.set_index('datum', inplace=True)

# sort data from past to present
df = df.iloc[::-1]

df_D = df.diff()

df_D.dropna(inplace=True)

with pd.ExcelWriter('data/kaub_weather.xlsx') as writer:
     df.to_excel(writer, sheet_name='abs_data')
     df_D.to_excel(writer, sheet_name='delta')

breakpoint()

df_D = normalize(df_D)
# store column names
col = df_D.columns

# define label and features
y = df_D[col[0]]
x = df_D[col[1:]]

pca_comp = 10
# PCA analyis
pca_x = PCA(n_components=pca_comp)
p_x = pca_x.fit_transform(x)
print('Explained variation per principal component: {}'.format(pca_x.explained_variance_ratio_))

plt.figure(figsize=(15,20))
for i in range(pca_comp):
     plt.subplot(pca_comp, 1, i+1)
     plt.scatter(y,p_x[:,i], alpha=0.5)
     plt.title('PC_'+str(i+1))
     plt.grid()
plt.suptitle('PCA')
plt.savefig('data/pca_d_weather.png')
plt.close()

# calculate lag per feature
lag = np.zeros(len(col)-1)

for i in range(1,len(col)):
    x_i = df_D[col[i]]
    correlation = correlate(x_i, y, mode='full')
    lags = correlation_lags(x_i.size, y.size, mode='full')
    lag[i-1] = lags[np.argmax(correlation)]

# make dataframe to add column names
lag_df_D = pd.DataFrame(lag).transpose()
lag_df_D.columns=col[1:]

metrics = ['minT', 'maxT', 'meanT', 'precip', 'snow', 'soil_5', 'soil_15']

def group_metrics(data, metric):
     lag_metric = data.filter(like=metric)
     lag_metric.columns = lag_metric.columns.str.rstrip('_'+metric)
     return lag_metric

with pd.ExcelWriter('data/grouped_metrics_lag.xlsx') as writer:
     for i in range(len(metrics)):
          grouped_data = group_metrics(lag_df_D,metrics[i])
          grouped_data.to_excel(writer, sheet_name=metrics[i])

breakpoint()

# filter precipitation lag
lag_precip = lag_df.filter(like='precip')
# remove precip suffix
lag_precip.columns = lag_precip.columns.str.rstrip("_precip")
# save to excel
lag_precip.to_excel('data/precip_lag.xlsx')


# specify columns to plot
columns = list(range(0,len(df.columns)))
# set counter
i = 1
values = df.values
breakpoint()
# plot 2d histogram
# define figure object and size
plt.figure(figsize=(15,40))
# plot each column with a for loop
for variable in columns:
     plt.subplot(int(len(columns)/2)+1, 2, i)
     plt.hist2d(y,values[:, variable], bins=(100,100), cmap='Blues')
     plt.title(df.columns[variable], y=0.5, loc='right')
     plt.grid()
     i += 1
plt.savefig('data/hist2d_plots_d.png')
plt.close()

breakpoint()

i=1
# scatter plots of label vs features
# define figure object and size
plt.figure(figsize=(15,40))
# plot each column with a for loop
for variable in columns:
     plt.subplot(int(len(columns)/2)+1, 2, i)
     plt.scatter(y,values[:, variable],alpha=0.2)
     plt.title(df.columns[variable], y=0.5, loc='right')
     plt.grid()
     i += 1
plt.savefig('data/scatter_plots_d.png')
plt.close()

i=1
# line plots of label and features
# define figure object and size
plt.figure(figsize=(15,40))
# plot each column with a for loop
for variable in columns:
     plt.subplot(len(columns), 1, i)
     plt.plot(values[:, variable])
     plt.title(df.columns[variable], y=0.5, loc='right')
     i += 1
plt.savefig('data/line_plots_d.png')
plt.close()
breakpoint()
# plot histogram
df.hist(figsize=(9,18))
plt.savefig('data/histogram_d.png')
plt.close()

# violing plot of data
# calculate dataset mean and standard deviation
mean = df.mean()
std = df.std()
# normalise dataset with previously calculated values
df_std = (df - mean) / std
# create violin plot
df_std = df_std.melt(var_name='Column', value_name='Normalised')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalised', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)
plt.savefig('data/violin_d.png')
plt.close()

breakpoint()
