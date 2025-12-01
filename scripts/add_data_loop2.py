import pandas as pd

workbook_v = "data_xitu_LOOP2.xlsx"
sheet_v = "Sheet1"

df = pd.read_excel(workbook_v, sheet_name=sheet_v)

search_ids = [13, 15, 134, 135]

filtered_rows = df[df['ID'].isin(search_ids)]

df_new = pd.DataFrame(columns=df.columns)

for _, row in filtered_rows.iterrows():
    if row['Label'] == 0:
        idx = df_new[df_new['Label'] == 1].index.min()
        if pd.isna(idx):
            df_new = pd.concat([df_new, pd.DataFrame([row])])
        else:
            df_new = pd.concat([df_new.iloc[:idx], pd.DataFrame([row]), df_new.iloc[idx:]])
    else:
        first_idx = df_new[df_new['Label'] == 1].index.min()
        if pd.isna(first_idx):
            df_new = pd.concat([df_new, pd.DataFrame([row])])
        else:
            last_continuous_1_idx = first_idx
            while (last_continuous_1_idx + 1 in df_new.index and df_new.loc[last_continuous_1_idx + 1, 'Label'] == 1):
                last_continuous_1_idx += 1
            df_new = pd.concat([df_new.iloc[:last_continuous_1_idx + 1], pd.DataFrame([row]), df_new.iloc[last_continuous_1_idx + 1:]])

remaining_df = df[~df['ID'].isin(search_ids)]
df_final = pd.concat([df_new, remaining_df])

df_final.reset_index(drop=True, inplace=True)

output_file = "data_xitu_LOOP3.xlsx"
df_final.to_excel(output_file, index=False)
print(f"Updated file saved as {output_file}")