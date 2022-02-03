import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from tueplots import bundles


import matplotlib as mpl
import matplotlib.pyplot as plt

plt_settings = bundles.neurips2021(usetex=False)
plt_settings['figure.constrained_layout.use'] = False
plt.rcParams.update(plt_settings)

fig_width, fig_height = plt_settings['figure.figsize']

# data:
interesting_features = ['explicit', 'danceability','loudness','valence']
predictors = ['explicit', 'danceability', 'energy', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
coef_ = pd.read_csv('../dat/log_reg_coef.csv', index_col=0).to_numpy()
train = pd.read_csv('../dat/train_set.csv')

# start of script
fig = plt.figure(figsize=(fig_width, fig_height*1.2))
gs = fig.add_gridspec(2, hspace=0)
axs = gs.subplots(sharex=True, sharey=True)
ax1, ax2 = axs

year_series = np.unique((np.unique(train['year'])))

for i,feature in enumerate(predictors):
    if feature in interesting_features:
        ax1.plot(year_series, coef_[:,i], '.-', label=feature,)
    else:
        ax1.plot(year_series, coef_[:,i], '-', alpha=.3,color='grey')
    

train_ = train[predictors + ['year']]
train_[predictors] = preprocessing.StandardScaler().fit_transform(train_[predictors])

tracks_by_year_avg = train_.groupby('year').mean()


for i,feature in enumerate(predictors[:-1]):
    if feature in interesting_features:    
        ax2.plot(year_series,tracks_by_year_avg[feature], '.-',label=feature )
    else:
        ax2.plot(year_series,tracks_by_year_avg[feature], '-', alpha=.3, color='grey')
ax2.plot(year_series,tracks_by_year_avg[predictors[-1]], '-', alpha=.3, color='grey', label='other')


ax2.set_xlabel('Year')
ax2.set_ylabel('Avg. Feature Value')
ax1.set_ylabel('Coefficient Value')
for ax in axs:
    #ax.label_outer()
    ax.set_ylim(-.75, .5)
    ax.grid(alpha=.2)
ax2.legend(loc=(.4,0.85), framealpha=1)
#ax2.legend(loc= 'lower center')

ax1.tick_params(axis='x', which='both', bottom=False,labelbottom=False)

plt.savefig('../doc/fig/coefs_avg.pdf',bbox_inches='tight')
plt.show()