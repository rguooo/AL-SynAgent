import pandas as pd
from collections import Counter

input_file_path = r"noGuiYi_combined_features_smiles_Group.xlsx"
output_file_path = r"smiles_class_frequency_by_label.xlsx"

df = pd.read_excel(input_file_path)

words_0 = []
words_1 = []

for _, row in df.iterrows():
    label = row['Label']
    class_str = row['SMILES_Class']
    
    if pd.isna(class_str):
        continue
        
    words = class_str.split('; ')
    
    if label == 0:
        words_0.extend(words)
    elif label == 1:
        words_1.extend(words)

counter_0 = Counter(words_0)
counter_1 = Counter(words_1)

all_words = set(counter_0.keys()).union(set(counter_1.keys()))

result = []
for word in all_words:
    result.append({
        'Word': word,
        'Frequency_Label_0': counter_0.get(word, 0),
        'Frequency_Label_1': counter_1.get(word, 0)
    })

result_df = pd.DataFrame(result)

result_df['Total'] = result_df['Frequency_Label_0'] + result_df['Frequency_Label_1']
result_df = result_df.sort_values(by='Total', ascending=False).drop(columns=['Total'])
result_df.to_excel(output_file_path, index=False)

print(f"done: {output_file_path}")