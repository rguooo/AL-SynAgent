import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d

plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_excel(
    r"combined_features_smiles_classified.xlsx", 
    sheet_name='Sheet1'
)
df = df[df['Label'].notna()].copy()
df['Label'] = df['Label'].astype(int)

features = [
    'group_electronegativity',
    'Rot_min',
    'TPSA',
    'dipoleX',
    't_2D_min',
    'hardness',
    'E3biE1',
    'kappa1'
]

save_path = r"figures"
os.makedirs(save_path, exist_ok=True)

base_width, base_height = 7, 4

for feature in features:
    fig, ax = plt.subplots(figsize=(base_width, base_height))
    data0 = df[df['Label'] == 0][feature].dropna().values
    data1 = df[df['Label'] == 1][feature].dropna().values
    if len(data0) < 2 or len(data1) < 2:
        ax.text(0.5, 0.5, f'Skipped: {feature}', fontsize=22, ha='center')
        ax.set_title(f'{feature} Distribution (skipped)')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.tick_params(axis='both', direction='in')
    else:
        noise_std = 0.5
        data1_shifted = data1 + np.random.normal(0, noise_std, len(data1))
        x_min = min(data0.min(), data1_shifted.min()) - 0.2
        x_max = max(data0.max(), data1_shifted.max()) + 0.2
        x = np.linspace(x_min, x_max, 1000)
        hist0, bins0 = np.histogram(data0, bins=200, range=(x_min, x_max), density=False)
        hist1, bins1 = np.histogram(data1_shifted, bins=200, range=(x_min, x_max), density=False)
        centers = 0.5 * (bins0[:-1] + bins0[1:])
        sigma = 6
        smooth0 = gaussian_filter1d(hist0, sigma)
        smooth1 = gaussian_filter1d(hist1, sigma)
        smooth0_scaled = smooth0 / 1000
        smooth1_scaled = smooth1 / 1000
        ax.fill_between(centers, smooth0_scaled, color='#58A5CC', alpha=0.6)
        ax.fill_between(centers, smooth1_scaled, color='#DDA3A6', alpha=0.6)
        ax.set_ylabel('Frequency ($\\times 10^3$)', fontsize=20)
        ax.set_ylim(bottom=0)
        ax.margins(x=0)
        ax.margins(y=0)
        ax.set_xlabel(feature, fontsize=20)
        ax.tick_params(axis='both', direction='in')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties('Arial')
            label.set_size(20)
    plt.tight_layout()
    filename = f"{feature}_distribution_filled_shifted.png"
    fig.savefig(os.path.join(save_path, filename), dpi=600, bbox_inches='tight')
    plt.close(fig)

print("All filled distribution plots with shift have been generated.")