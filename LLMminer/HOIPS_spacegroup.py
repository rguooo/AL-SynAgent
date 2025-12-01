import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"

file_path =  r"ferroelectric_materials_processed_final.xlsx"
df_organic = pd.read_excel(file_path, sheet_name='organic')
df_spacegroup = pd.read_excel(file_path, sheet_name='spacegroup')

def clean_data(data):
    return data.str.strip().str.lower().str.replace('_', '')

df_organic['Cleaned_Space_Group'] = clean_data(df_organic['Space Group'])

df_spacegroup['Cleaned_Space_Group'] = clean_data(df_spacegroup['Space Group'])

valid_spacegroups = df_spacegroup['Cleaned_Space_Group'].unique()
df_organic['Exists'] = df_organic['Cleaned_Space_Group'].isin(valid_spacegroups)

valid_counts = df_organic[df_organic['Exists']]['Cleaned_Space_Group'].value_counts()

top_n = 10
other_label = "Other"

top_counts = valid_counts.head(top_n)
other_count = valid_counts[top_n:].sum()

new_counts = pd.concat([top_counts, pd.Series(other_count, index=[other_label])])

colors = ['#F0F4F9','#E2E8ED','#D3DCE9','#C8D1E3','#BAC6DB',
          '#ACBDD4','#9FAFCE','#90A4C4','#839AC0','#6D81AF','#657BAB']

plt.figure(figsize=(8, 8))

wedgeprops = {'width': 0.4, 'edgecolor': 'white', 'linewidth': 1}

wedges, texts, autotexts = plt.pie(
    new_counts.values,
    labels=new_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops=wedgeprops,
    colors=colors
)

for text in texts:
    text.set_fontsize(10)
    text.set_fontname('Arial')

for autotext in autotexts:
    autotext.set_fontsize(16)
    autotext.set_fontname('Arial')
    autotext.set_color('black')

plt.gca().add_artist(plt.Circle((0, 0), 0.2, color='white'))
plt.text(0, 0, 'HOIPs\nspacegroup', ha='center', va='center', fontsize=24, fontname='Arial')

plt.axis('equal')

output_file = 'pie_chart_high_or.png'
plt.savefig(output_file, dpi=500, bbox_inches='tight', format='png')
print(f"Chart saved as {output_file}")