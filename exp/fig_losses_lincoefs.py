import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from tueplots import bundles

plt_settings = bundles.neurips2021(usetex=False)
plt.rcParams.update(plt_settings)

fig_width, fig_height = plt_settings['figure.figsize']

# data 1:
names = ['Random', 'Lin. Regression', 'Poly. Lin. Regression', 'Log. Regression', 'Poly. Log. Regression']
l1_losses = [10.129583746499913, 6.113696974889281, 5.956544964282252, 7.596235415401686, 7.7255190609943085]
l2_losses = [153.8909257698003, 53.87570765992985, 51.64688522673475, 95.0179798447513, 98.38477129864627]
accuracy = [0.031334553539836545, 0.04506318718521075, 0.04871070185171433, 0.07885620929308108, 0.07910520131223163]

# data 2:
interesting_features = ['explicit', 'danceability','loudness','valence']
predictors = ['explicit', 'danceability', 'energy', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
coef_ = pd.read_csv('../dat/lin_reg_coef.csv', index_col=0).to_numpy()

N = len(names)
ind = np.arange(N)
fig, axs = plt.subplots(1,2, figsize=(fig_width-.1, fig_height), gridspec_kw={'width_ratios': [2, 3]})
ax, ax2 = axs
ax1 = ax.twinx()
                       
width = 0.3       

# TODO change these colors
color0 = 'red'
color1 = 'green'

# Plotting
b0 = ax.bar(ind, l1_losses , width, label='MAE', color=color0)
#b0 = ax.bar(ind, l2_losses , width, label='$L_2$-Loss (years)', color=color0)
b1 = ax1.bar(ind + width, accuracy, width, label='Accuracy', color=color1)
ax.set_xticks(ind + width / 2, names, rotation=45, horizontalalignment='right')

ax.legend([b0,b1],[b.get_label() for b in [b0, b1]], loc='upper left')

ax.tick_params(axis='y', labelcolor=color0)
accs = np.array(list(range(0,9,1)))
ax1.set_yticks(accs/100, [f'{a}%' for a in accs])
ax1.tick_params(axis='y', labelcolor=color1)
ax.set_ylim(0,12.5)

interesting_features = ['explicit', 'danceability','loudness','valence']
for f, c in zip(predictors,coef_):
    ax2.bar(f, c, color='grey' if f not in interesting_features else None)
plt.setp(ax2.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.savefig('../doc/fig/losses_lincoefs.pdf',bbox_inches='tight')
plt.show()