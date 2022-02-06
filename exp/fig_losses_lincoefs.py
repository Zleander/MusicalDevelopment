import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from tueplots import bundles

plt_settings = bundles.neurips2021(usetex=False, family='serif')
plt.rcParams.update(plt_settings)
plt.style.use('seaborn-colorblind')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  

fig_width, fig_height = plt_settings['figure.figsize']

# data 1:
model_names = ['Random', 'Lin. Regression', 'Poly. Lin. Regression', 'Log. Regression', 'Poly. Log. Regression']
l1_losses = [10.129583746499913, 6.113696974889281, 5.956544964282252, 7.596235415401686, 7.7255190609943085]
l2_losses = [153.8909257698003, 53.87570765992985, 51.64688522673475, 95.0179798447513, 98.38477129864627]
accuracy = [0.031334553539836545, 0.04506318718521075, 0.04871070185171433, 0.07885620929308108, 0.07910520131223163]
accuracy[1:3] = [0,0] # setting accuracy of linear models to 0

# data 2:
interesting_features = ['explicit', 'danceability','loudness','valence']
predictors = ['explicit', 'danceability', 'energy', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']
pred_names = predictors.copy()
pred_names[-1] = "duration"
coef_ = pd.read_csv('../dat/lin_reg_coef.csv', index_col=0).to_numpy()

N = len(model_names)
ind = np.arange(N)
fig, axs = plt.subplots(1,2, figsize=(fig_width-.1, fig_height), gridspec_kw={'width_ratios': [2, 3]})
ax, ax2 = axs
ax1 = ax.twinx()
                       
width = 0.3       


# Plotting
b0 = ax.bar(ind, l1_losses , width, label='MAE', color=colors[2])
#b0 = ax.bar(ind, l2_losses , width, label='$L_2$-Loss (years)', color=color0)
b1 = ax1.bar(ind + width, accuracy, width, label='Accuracy', color=colors[1])
ax.set_xticks(ind + width / 2, model_names, rotation=45, horizontalalignment='right')

ax.legend([b0,b1],[b.get_label() for b in [b0, b1]], loc='upper left')

ax.tick_params(axis='y', labelcolor=colors[2])
accs = np.array(list(range(0,9,1)))
ax1.set_yticks(accs/100, [f'{a}%' for a in accs])
ax1.tick_params(axis='y', labelcolor=colors[1])
ax.set_ylim(0,12.5)

interesting_features = ['explicit', 'danceability','loudness','valence']
for f, c in zip(pred_names,coef_):
    ax2.bar(f, c,color='grey' if f not in interesting_features else None)
plt.setp(ax2.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.savefig('../doc/fig/losses_lincoefs.pdf',bbox_inches='tight')
plt.show()