import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_path = r"combined_features_smiles_classified.xlsx"
df = pd.read_excel(file_path)

assert 'Label' in df.columns and 'SMILES_Class' in df.columns

counts_0 = df[df['Label'] == 0]['SMILES_Class'].value_counts()
counts_1 = df[df['Label'] == 1]['SMILES_Class'].value_counts()
freq_df = pd.DataFrame({'Label_0': counts_0, 'Label_1': counts_1}).fillna(0)
freq_df['Total'] = freq_df['Label_0'] + freq_df['Label_1']
freq_df = freq_df.sort_values(by='Total', ascending=False)
freq_df = freq_df.drop(columns='Total')

x = np.arange(len(freq_df))
width = 0.35

fig, axs = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={'width_ratios': [2, 1, 1]})

axs[0].bar(x - width/2, freq_df['Label_0'], width, label='Label = 0', color='#fdb863')
axs[0].bar(x + width/2, freq_df['Label_1'], width, label='Label = 1', color='#80b1d3')
axs[0].set_xticks(x)
axs[0].set_xticklabels(freq_df.index, rotation=45, ha='right')
axs[0].set_ylabel("Frequency")
axs[0].set_xlabel("SMILES_Classzz")
axs[0].legend()
axs[0].set_title("SMILES Class Frequency by Label")
axs[0].grid(axis='y', linestyle='--', alpha=0.5)

axs[1].pie(counts_0.values, labels=counts_0.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
axs[1].set_title("SMILES Class (Label = 0)")

axs[2].pie(counts_1.values, labels=counts_1.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel2.colors)
axs[2].set_title("SMILES Class (Label = 1)")

plt.tight_layout()
plt.show()
