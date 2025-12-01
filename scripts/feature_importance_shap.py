import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
import shap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
import matplotlib.collections as collections

workbook_v = "data_xitu_LOOP3s.xlsx"
sheet_v = str("Sheet1")

ID_v, formula_v, prototype_v, prototype_2_v, features_v, labels_v, features_name_v = data_load(workbook_v, sheet_v)
verX = features_v
verY = labels_v

scaler = MinMaxScaler()
verX_scaled = scaler.fit_transform(verX)

model_files = [f'Results - 3/Feature_Engineering_Result_FF/best_model_{i}_FF.pickle' for i in range(10)]

probabilities_list = []
loaded_models = []
for model_file in model_files:
    with open(model_file, 'rb') as file:
        loaded_model = pickle.load(file)
        loaded_models.append(loaded_model)
        probabilities = loaded_model.predict_proba(verX)
        probabilities_list.append(probabilities)

average_probabilities = np.mean(probabilities_list, axis=0)

average_predictions = np.argmax(average_probabilities, axis=1)

print("Predictions:", average_predictions)

print("Probabilities:", average_probabilities)

fpr, tpr, _ = roc_curve(verY, average_probabilities[:, 1])
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc}")

colors = [
        "#CD7175", "#DEA2A5", "#EBC7C9",
        "#C2CDE0", "#8AA0C4", "#5371A3"
]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

model_for_explain = loaded_models[0]
X_for_shap = verX
try:
    if hasattr(model_for_explain, 'named_steps') and 'classifier' in model_for_explain.named_steps:
        preproc = model_for_explain[:-1]
        X_for_shap = preproc.transform(verX)
        model_for_explain = model_for_explain.named_steps['classifier']
except Exception:
    pass
try:
    explainer = shap.TreeExplainer(model_for_explain)
    shap_values = explainer.shap_values(X_for_shap, check_additivity=False)
except Exception:
    bg = shap.sample(X_for_shap, min(200, X_for_shap.shape[0]), random_state=0)
    explainer = shap.KernelExplainer(model_for_explain.predict_proba, bg)
    X_explain = X_for_shap
    try:
        if X_for_shap.shape[0] > 2000:
            X_explain = shap.sample(X_for_shap, 2000, random_state=0)
    except Exception:
        X_explain = X_for_shap
    shap_values = explainer.shap_values(X_explain)
    X_for_shap = X_explain

_sv_for_plot = shap_values if isinstance(shap_values, np.ndarray) else (shap_values[1] if len(shap_values) > 1 else shap_values[0])
norm = Normalize(vmin=np.min(_sv_for_plot), vmax=np.max(_sv_for_plot))

plt.rcParams['font.family'] = 'Arial'
feature_names_to_use = features_name_v
try:
    if hasattr(X_for_shap, 'shape') and X_for_shap.shape[1] != len(features_name_v):
        feature_names_to_use = None
except Exception:
    feature_names_to_use = None
shap.summary_plot(_sv_for_plot, X_for_shap, feature_names=feature_names_to_use, cmap=cmap, show=False)

fig = plt.gcf()
fig.set_size_inches(7.8, 5)
for ax in fig.axes:
    for coll in ax.collections:
        if isinstance(coll, collections.PathCollection):
            coll.set_sizes([50])

plt.draw()
plt.tight_layout()
summary_plot_png = "SHAP-LOOP3.png"
plt.savefig(summary_plot_png, dpi=1000, bbox_inches="tight")
plt.close()

plt.show()
