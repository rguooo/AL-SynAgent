import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

df = pd.read_csv('train_allfeature_ed_8_Fe1.csv').dropna(how='all')
features = df.iloc[:, 4:-1]
data = pd.concat([features, df['Label']], axis=1)
corr_matrix = data.corr(method='pearson')
colors = ["#DDA3A6", "#E2B0B2", "#E5B9BB", "#EED2D3", "#FDF9F9", "#F6FAFC", "#C5E0ED", "#83BCD9", "#63ABCF", "#58A5CC"]
cmap = LinearSegmentedColormap.from_list("custom", colors, N=100)
plt.figure(figsize=(14, 12))
heatmap = sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap=cmap,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    annot_kws={
        'size': 16,
        'fontname': 'Arial'
    },
    cbar_kws={
        'shrink': 0.8,
        'label': 'Correlation'
    }
)

heatmap.set_xticklabels(
    heatmap.get_xticklabels(),
    rotation=45,
    ha='right',
    fontsize=26,
    fontname='Arial'
)
heatmap.set_yticklabels(
    heatmap.get_yticklabels(),
    rotation=0,
    fontsize=26,
    fontname='Arial'
)

plt.title(" ", fontsize=1, pad=25, fontname='Arial')
cbar = heatmap.collections[0].colorbar
cbar.ax.yaxis.label.set_fontproperties({'family': 'Arial', 'size': 26})
cbar.ax.tick_params(labelsize=18)
plt.tight_layout()
plt.savefig('pearson_feature_engineering.png', dpi=500)
plt.show()