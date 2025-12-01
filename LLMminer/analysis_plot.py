import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import LinearSegmentedColormap

file_path = r"ferroelectric_materials_processed_final.xlsx"
df = pd.read_excel(file_path, sheet_name=0)

df = df.dropna().reset_index(drop=True)
df.columns = ['Element1', 'Element2', 'Organic']

x_unique = df['Element1'].unique()
y_unique = df['Element2'].unique()
z_unique = df['Organic'].unique()

custom_x_order = ['Eu','Cd','Ce', 'Cr', 'Cu',  'Fe', 'Ge', 'Mn', 'Ni',  'Pr', 'Rb', 'Sb',  'Zn','Bi','Sn','Pb']
custom_y_order = ['I','Br', 'Cl', 'NO2', 'NO3','HCOO', 'N(CN)2']
custom_z_order = ['BA','(4-aminophenyl)', '(C4H3S)CH2NH3', '(CH3)2CHCH2NH3', '(CH3)2NH2', '(CH3)3NH', '2-FBA', '2meptH2', '3-PyA','A', 
                'IA','3AMPY', '4ACHO', '4FPEA',  'AMPI', 'AP', 'ATHP',  'BBA', 'BDA',  '3-hydroxypyrrolidinium',
                'EA','BPA',  'Br-MM', 'BrBA', 'C10H18N2', 'C10H21NH3', 'C12H15N', 'C3H10N', 'C3H8N', 'C3H9N',
                'FA','C4H10FN', 'C4H10N', 'C4H12IN', 'C4H8NH2', 'C4H9FN', 'C5H11NH3', 'C5H12N', 'C5H13NBr', 'C5H14N2', 
                'HA','C6H4(NH3)2', 'C6H5C2H4NH3', 'C6H5C3H6NH3', 'C6H5CH(CH3)NH3', 'C6H5N(CH3)3', 'C6H5NH3', 'C6H8N2', 'C6H9N2','DAP', 
                'GA','C7H13NH3', 'C7H14NO', 'C8H10BrN', 'C8H12ClN', 'C8H18N', 'C8H24N2', 'C9H13NH3', 'CH3(CH2)3NH3', 'CH3NH2NH2', 
                'PA','CH3OC3H9N', 'CHFClNH3',  'CYHEA', 'Cl-MM', 'ClC2H4NH3', 'ClC6H4CH(CH3)NH3', 'ClMBA', 'D-AlaH', 'BFDA',
                'CPA','DF-CBA', 'DFCBA', 'DFCHA', 'DFHHA', 'DFPD', 'DFPIP', 'DMA', 'DMAA', 'DMTE',  
                'MHY','FC2H4NH3', 'FMQ', 'FMTMA', 'FMeTP', 'FP', 'FPEA',  'H3NC6H4NH3',  'HAD', 'HDA',  
                'BZA','L-HisH', 'M3HQ', 'MA', 'MACH', 'MBA', 'MDABCO',  'MP', 'MPA', 'MPEA', 
                'NEA','MeBA', 'N-methylpyrrolidinium',  'NH3CH2CH2F', 'NH3NH2', 'NMEA',  'PFBA', 'C6H20N2','EATMP', 'EQ', 
                'PEA','R3HP', 'RM3HQ', 'TFBDA', 'TMA', 'TMAEA', 'TMBM', 'TMCM', 'TMFM', 'TMIM', 
                'AMP','TMPDA', 'TMS', '[Me3NCH2Cl','IAA', 'Me3NCH2Cl', 'Me3NCH2F', 'hydroxypyrrolidinium', 't-ACH','PMA',
                'DPA']

df['Element1'] = pd.Categorical(df['Element1'], categories=custom_x_order, ordered=True)
df['Element2'] = pd.Categorical(df['Element2'], categories=custom_y_order, ordered=True)
df['Organic'] = pd.Categorical(df['Organic'], categories=custom_z_order, ordered=True)

le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
df['X'] = le1.fit_transform(df['Element1'])
df['Y'] = le2.fit_transform(df['Element2'])
df['Z'] = le3.fit_transform(df['Organic'])

custom_colors = ["#CD7175", "#DEA2A5", "#EBC7C9", "#C2CDE0", "#8AA0C4", "#5371A3"]
cmap = LinearSegmentedColormap.from_list('custom', custom_colors, N=256)

fig = plt.figure(figsize=(11, 14))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(df['X'], df['Y'], df['Z'],c=df['Z'], cmap=cmap, s=80, edgecolor='k', alpha=0.9)

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

x_tick_indices = list(range(0, len(custom_x_order), 1))
y_tick_indices = list(range(0, len(custom_y_order), 1))
z_tick_indices = list(range(0, len(custom_z_order), 10))

ax.set_xticks(x_tick_indices)
ax.set_xticklabels([custom_x_order[i] for i in x_tick_indices], 
                   rotation=45, ha='right', fontsize=16)
ax.set_yticks(y_tick_indices)
ax.set_yticklabels([custom_y_order[i] for i in y_tick_indices], 
                   rotation=0, ha='left', fontsize=16)
ax.set_zticks(z_tick_indices)
ax.set_zticklabels([custom_z_order[i] for i in z_tick_indices], 
                   rotation=45,va='center',  fontsize=16)

ax.xaxis.set_tick_params(pad=-8)
ax.yaxis.set_tick_params(pad=-5)
ax.zaxis.set_tick_params(pad=7)

for label in ax.zaxis.get_ticklabels():
    label.set_position((label.get_position()[0], label.get_position()[1] + 0.5))

ax.view_init(elev=25, azim=-70)
ax.dist = 9.5

ax.set_xlabel('B-position', fontsize=16, labelpad=2)
ax.set_ylabel('X-position', fontsize=16, labelpad=5)
ax.set_zlabel('A-position', fontsize=16, labelpad=14)

plt.rcParams['font.family'] = 'Arial'

ax.view_init(elev=10, azim=-70)

ax.set_box_aspect([1, 1, 1.25])

ax.dist = 8

plt.tight_layout()

plt.savefig('3d_scatter_plot.png', dpi=600, bbox_inches='tight', facecolor='white')

plt.show()