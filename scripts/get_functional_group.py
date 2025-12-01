import pandas as pd
from rdkit import Chem

file_path = r"combined_features.xlsx"
df = pd.read_excel(file_path)

functional_groups = {
    "Carboxylic acid": "[CX3](=O)[OX1H0-,OX2H1]",
    "Aldehyde": "[CX3H1](=O)[#6]",
    "Ketone": "[CX3](=O)[#6]",
    "Alcohol": "[#6][OX2H]",  
    "Amine": "[NX3;H2,H1;!$(NC=O)]",  
    "Nitrile": "[CX2]#N",
    "Halogen": "[F,Cl,Br,I]",
    "Aromatic": "a",
    "Ether": "[OD2]([#6])[#6]", 
    "Nitro": "[NX3](=O)[O-]",
    "Thiol": "[#16X2H]"
}

fg_patterns = {name: Chem.MolFromSmarts(smarts) for name, smarts in functional_groups.items()}

def classify_by_functional_group(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid"
        matches = []
        for name, pattern in fg_patterns.items():
            if mol.HasSubstructMatch(pattern):
                matches.append(name)
        if matches:
            return "; ".join(matches)
        else:
            return "Unclassified"
    except:
        return "Invalid"

df["SMILES_Class"] = df["SMILES"].apply(classify_by_functional_group)

output_path = r"noGuiYi_combined_features_smiles_Group.xlsx"
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    df.to_excel(writer, index=False)

print(df[["SMILES", "SMILES_Class"]].head())