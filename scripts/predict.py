import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, precision_score, accuracy_score, recall_score, f1_score
import os
import numpy as np
import xlrd
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from hyperopt import fmin, tpe, hp, rand, anneal, partial, Trials
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, confusion_matrix, auc, roc_auc_score
from sklearn.model_selection import cross_val_score
from imblearn.ensemble import EasyEnsembleClassifier, BalancedBaggingClassifier
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.axes._axes import _log as matplotlib_axes_logger
import warnings
import pandas as pd
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)

def data_load(filename, sheet_name):
    try:
        df = pd.read_excel(filename, sheet_name=sheet_name)
        print("Excel file read successfully.")
    except Exception as e:
        print(f"Error reading the Excel file: {e}")
        return None
    ID = df['ID'].astype(int).tolist()
    formula = df['SMILES'].tolist()
    prototype = df['B_feature'].tolist()
    prototype_2 = df['A_feature'].tolist()
    features = df.iloc[:, 4:-1].to_numpy()
    label = df['Label'].to_numpy()
    features_name = df.columns[4:-1].tolist()
    return ID, formula, prototype, prototype_2, features, label, features_name

workbook_v = "ver.xlsx"
sheet_v = str("Sheet1")

ID_v, formula_v, prototype_v, prototype_2_v, features_v, labels_v, features_name_v = data_load(workbook_v, sheet_v)
verX = features_v
verY = labels_v

model_files = [f'model/Results - 3/Feature_Engineering_Result_FF/best_model_{i}_FF.pickle' for i in range(10)]

probabilities_list = []
for model_file in model_files:
    with open(model_file, 'rb') as file:
        loaded_model = pickle.load(file)
        probabilities = loaded_model.predict_proba(verX)
        probabilities_list.append(probabilities)

average_probabilities = np.mean(probabilities_list, axis=0)
average_predictions = np.argmax(average_probabilities, axis=1)
df_result = pd.read_excel(workbook_v, sheet_name=sheet_v)
df_result['Predicted_Label'] = average_predictions
df_result['Positive_Probability'] = average_probabilities[:, 1]
output_file = "train23.xlsx"
df_result.to_excel(output_file, index=False)
print(f"Results saved to {output_file}")
print("Predictions:", average_predictions)
print("Probabilities:", average_probabilities)
fpr, tpr, _ = roc_curve(verY, average_probabilities[:, 1])
roc_auc = auc(fpr, tpr)
precision = precision_score(verY, average_predictions)
accuracy = accuracy_score(verY, average_predictions)
recall = recall_score(verY, average_predictions)
f1 = f1_score(verY, average_predictions)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic', fontsize=12)
plt.legend(loc="lower right", fontsize=12)
plt.text(0.05, 0.95, f'Precision: {precision:.2f}', fontsize=12, verticalalignment='top')
plt.text(0.05, 0.90, f'Accuracy: {accuracy:.2f}', fontsize=12, verticalalignment='top')
plt.text(0.05, 0.85, f'Recall: {recall:.2f}', fontsize=12, verticalalignment='top')
plt.text(0.05, 0.80, f'F1 Score: {f1:.2f}', fontsize=12, verticalalignment='top')
plt.show()
cm = confusion_matrix(verY, average_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 30})
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.title('Confusion Matrix', fontsize=16)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
report = classification_report(verY, average_predictions)
print("Classification Report:\n", report)
