import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

file_path = 'tsne_all.xlsx'

predict_df = pd.read_excel(file_path, sheet_name='predict')
train_df = pd.read_excel(file_path, sheet_name='train')
ver_df = pd.read_excel(file_path, sheet_name='ver')

features = predict_df.columns[4:]

predict_features = predict_df[features].values
predict_labels = predict_df['Label'].values

train_features = train_df[features].values
train_labels = train_df['Label'].values

ver_features = ver_df[features].values
ver_labels = ver_df['Label'].values

all_features = np.vstack([predict_features, train_features, ver_features])
all_labels = np.concatenate([predict_labels, train_labels, ver_labels])

tsne = TSNE(n_components=2, random_state=42, perplexity=70)
tsne_results = tsne.fit_transform(all_features)

plt.figure(figsize=(23, 8))

macaron_colors = ['#92A7C8' ,'#DFABAC']
cmap = plt.cm.colors.ListedColormap(macaron_colors)

scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1],
                      c=all_labels,
                      cmap=cmap,
                      s=50,
                      alpha=0.9,
                      edgecolors='none')

plt.grid(False)

plt.title('t-SNE visualization of features (colored by Label)', fontsize=16)
plt.xlabel('t-SNE 1', fontsize=12)
plt.ylabel('t-SNE 2', fontsize=12)

unique_labels = np.unique(all_labels)
handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Label {label}',
                      markerfacecolor=macaron_colors[i], markersize=8) 
           for i, label in enumerate(unique_labels)]
plt.legend(handles=handles, title="Label", loc='upper right')

plt.tight_layout()
plt.savefig('tsne_high_resolution.png', dpi=1000, bbox_inches='tight')
plt.show()