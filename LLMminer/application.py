import pandas as pd
import matplotlib.pyplot as plt

file_path = 'application.xlsx'
sheet_name = 'Statistics'
df = pd.read_excel(file_path, sheet_name=sheet_name)

data = df.head(15)

x_labels = data.iloc[:, 0]
y_values = data.iloc[:, 1]

plt.figure(figsize=(12, 6))
plt.bar(x_labels, y_values, color='#90A4C4')
plt.ylabel('Frequency', fontname='Arial', fontsize=22)
plt.xticks(rotation=45, ha='right', fontname='Arial')
plt.tick_params(axis='both', which='major', labelsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
output_file = 'bar_chart_high_res.png'
plt.savefig(output_file, dpi=500, bbox_inches='tight')
print(f"figure saved {output_file}")