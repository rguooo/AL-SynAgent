import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file_path = r"ferroelectric_materials_processed_final.xlsx"

organic_df = pd.read_excel(file_path, sheet_name="organic")
nonorganic_df = pd.read_excel(file_path, sheet_name="nonorganic")

target_columns = [
    "Band gap (eV)", "Crystal size (nm)", "Curie temperature (K)",
    "Spontaneous polarization (μC/cm²)", "Saturated polarization (μC/cm²)",
    "Remnant polarization (μC/cm²)", "Coercive field (kV/cm)", "Synthesis temperature (K)",
    "Synthesis time (h)", "Yield (%)", "Dielectric constant",
    "Piezoelectric coefficient (pm/V)", "Piezoelectric coefficient (pC/N)",
    "Lifetime (ns)", "Absorption wavelength (nm)",
    "Breakdown field (kV/cm)",
    "Quantum yield (%)", "PCE (%)",
    "Voc (V)", "Jsc (mA/cm2)",
    "Rise time (ms)", "Decay time (ms)",
    "Switching Cycles"
]

def extract_numeric_value(value):
    if pd.isna(value):
        return float('nan')
    if isinstance(value, str):
        parts = value.split()
        for part in parts:
            try:
                return float(part)
            except ValueError:
                continue
        return float('nan')
    else:
        return float(value)

for column in target_columns:
    organic_df[column] = organic_df[column].apply(extract_numeric_value)
    nonorganic_df[column] = nonorganic_df[column].apply(extract_numeric_value)

def process_tc(row):
    tc_value = extract_numeric_value(row["Curie temperature (K)"])
    unit = row.get("Tc Unit", "").strip() if not pd.isna(row.get("Tc Unit")) else ""
    if pd.isna(tc_value):
        return float('nan')
    if unit == "°C":
        return tc_value + 273
    else:
        return tc_value

organic_df["Curie temperature (K)"] = organic_df.apply(process_tc, axis=1)
nonorganic_df["Curie temperature (K)"] = nonorganic_df.apply(process_tc, axis=1)

data = {"Material Type": [], "Metric": [], "Value": []}
for material_type, df in [("organic", organic_df), ("nonorganic", nonorganic_df)]:
    for metric in target_columns:
        values = df[metric].dropna().tolist()
        data["Material Type"].extend([material_type] * len(values))
        data["Metric"].extend([metric] * len(values))
        data["Value"].extend(values)

long_data = pd.DataFrame(data)

def remove_outliers(df):
    Q1 = df['Value'].quantile(0.25)
    Q3 = df['Value'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df['Value'] >= lower) & (df['Value'] <= upper)]

long_data = long_data.groupby('Metric', group_keys=False).apply(remove_outliers)

n_metrics = len(target_columns)
ncols = 4
nrows = (n_metrics + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 4), sharey=False)
axes = axes.flatten()

custom_palette = {"organic": "#8488B5", "nonorganic": "#8FBEDC"}

log_transform_metrics = ["spontaneous_polarization (μC/cm²)", "Dielectric Constant_value"]

for i, metric in enumerate(target_columns):
    subset = long_data[long_data['Metric'] == metric]
    
    if metric in log_transform_metrics:
        subset = subset.copy()
        subset['Value'] = subset['Value'].apply(lambda x: np.log1p(x) if x > 0 else float('nan'))

    if metric == "rise time (ms)":
        axes[i].set_ylim(-1, 5)
        axes[i].set_ylabel("rise time (ms)")
    elif metric == "decay time (ms)":
        axes[i].set_ylim(-0.1, 1.5)
        axes[i].set_ylabel("decay time (ms)")
    elif metric == "lifetime (ns)":
        axes[i].set_ylim(-100, 2000)
        axes[i].set_ylabel("lifetime (ns)")

    ax = axes[i]

    sns.boxplot(
        x="Metric",
        y="Value",
        hue="Material Type",
        data=subset,
        ax=ax,
        palette=custom_palette,
        width=0.3,
        flierprops=dict(marker='o', markersize=3, alpha=0.5),
        saturation=0.7,
        boxprops=dict(linewidth=2),
        medianprops=dict(linewidth=2),
        whiskerprops=dict(linewidth=2),
        capprops=dict(linewidth=2)
    )

    ax.legend_.remove()
    ax.tick_params(axis='x', labelsize=12)
    
    ax.margins(x=0.1)

    ax.set_ylabel(metric, fontsize=16)

    ax.set_xlabel("HOIPS              AIPs", fontsize=16)
    ax.set_xticklabels([])

    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.grid(False)

for j in range(i+1, len(axes)):
    fig.delaxes(axes[j])

handles, labels = axes[0].get_legend_handles_labels()

plt.rcParams.update({
    'font.sans-serif': ['Arial'],
    'font.size': 14,
    'axes.labelsize': 14,
    'ytick.labelsize': 14,
    'xtick.labelsize': 14,
    'legend.fontsize': 14
})

plt.subplots_adjust(wspace=0.2, hspace=0.6)

plt.tight_layout(rect=[0, 0, 0.9, 0.95])
output_file = 'xiang_all.png'
plt.savefig(output_file, dpi=1500, bbox_inches='tight', format='png')
plt.show()