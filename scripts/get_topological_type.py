import pandas as pd
from rdkit import Chem

file_path = r"combined_features.xlsx"
df = pd.read_excel(file_path)

def classify_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid"
        is_aromatic = any(atom.GetIsAromatic() for atom in mol.GetAtoms())
        if is_aromatic:
            return "Aromatic"
        ring_info = mol.GetRingInfo()
        if ring_info.NumRings() > 0:
            return "Aliphatic ring"
        return "Aliphatic chain"
    except Exception as e:
        return "Invalid"

df["SMILES_Class"] = df["SMILES"].apply(classify_smiles)

output_path = r"noGuiYi_combined_features_smiles_classified.xlsx"
df.to_excel(output_path, index=False)

print(df[["SMILES", "SMILES_Class"]].head())