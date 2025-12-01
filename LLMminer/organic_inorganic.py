import pandas as pd

input_file = r"your-file-path"
df = pd.read_excel(input_file, sheet_name="Sheet1")

contains_O3_O9 = df["Chemical Formula"].str.contains("O3|O6|O7|O9|O12|O15|O17|I2|In2|S6|Cd3|Ti3|Cl3", case=False, na=False)
contains_C_Pb_I_Br = df["Chemical Formula"].str.contains("C|Pb|I|Br", case=False, na=False)

sheet1_df = df.copy()

sheet2_df = df[contains_O3_O9]

sheet3_df = df[~contains_O3_O9 & contains_C_Pb_I_Br]

with pd.ExcelWriter(input_file, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    sheet3_df.to_excel(writer, sheet_name="organic", index=False)
    sheet2_df.to_excel(writer, sheet_name="nonorganic", index=False)

print("Data successfully written to organic and nonorganic")