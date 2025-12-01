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

norm = Normalize(vmin=np.min(shap_values), vmax=np.max(shap_values))

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['font.size'] = 22
plt.rcParams['legend.fontsize'] = 22
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['ytick.major.width'] = 2

feature_names_to_use = features_name_v
try:
    if hasattr(X_for_shap, 'shape') and X_for_shap.shape[1] != len(features_name_v):
        feature_names_to_use = None
except Exception:
    feature_names_to_use = None
shap.summary_plot(shap_values, X_for_shap, feature_names=feature_names_to_use, cmap=cmap, show=False)

fig = plt.gcf()

for ax in fig.axes:
    for collection in ax.collections:
        if isinstance(collection, collections.PathCollection):
            collection.set_sizes([30])

plt.draw()
plt.tight_layout()
summary_plot_png = "summary_plot_high_res.png"
plt.savefig(summary_plot_png, dpi=1000, bbox_inches="tight")
plt.close()

plt.show()

xgb_model = model_for_explain
try:
    avg_importances = xgb_model.feature_importances_
except Exception:
    try:
        avg_importances = None
        if hasattr(model_for_explain, 'estimators_'):
            base_estimators = model_for_explain.estimators_
        elif hasattr(loaded_models[0], 'named_steps') and 'classifier' in loaded_models[0].named_steps and hasattr(loaded_models[0].named_steps['classifier'], 'estimators_'):
            base_estimators = loaded_models[0].named_steps['classifier'].estimators_
        elif hasattr(loaded_models[0], 'estimators_'):
            base_estimators = loaded_models[0].estimators_
        else:
            base_estimators = None

        if base_estimators is not None and len(base_estimators) > 0:
            accum = None
            for est in base_estimators:
                try:
                    fi = est.named_steps['classifier'].feature_importances_
                except Exception:
                    fi = getattr(est, 'feature_importances_', None)
                if fi is None:
                    continue
                fi = np.asarray(fi)
                accum = fi if accum is None else (accum + fi)
            if accum is not None:
                avg_importances = accum / len(base_estimators)

        if avg_importances is None:
            avg_importances = np.zeros(X_for_shap.shape[1])
    except Exception:
        avg_importances = np.zeros(X_for_shap.shape[1])

indices = np.argsort(avg_importances)[::-1]

labels = features_name_v
try:
    if len(features_name_v) != len(avg_importances):
        labels = [f"f{i}" for i in range(len(avg_importances))]
except Exception:
    labels = [f"f{i}" for i in range(len(avg_importances))]

plt.figure(figsize=(12, 8))
bars = plt.bar(range(len(avg_importances)), avg_importances[indices], align='center', color="#5371A3")
plt.xticks(range(len(avg_importances)), [labels[i] for i in indices], rotation=90)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('XGBoost Feature Importances')
plt.tight_layout()

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.001, f"{yval:.3f}", ha='center', va='bottom', fontsize=10)

plt.savefig("feature_importance_ensemble_avg.png", dpi=600, bbox_inches="tight")
plt.show()

indices = np.argsort(avg_importances)[::-1]

epsilon = 1e-8
importance_shifted = avg_importances + epsilon

importance_normalized = (importance_shifted - np.min(importance_shifted)) / \
                        (np.max(importance_shifted) - np.min(importance_shifted))
scale_factor = 10
importance_scaled = importance_normalized * scale_factor

plt.figure(figsize=(12, 8))
bars = plt.bar(range(len(avg_importances)), importance_scaled[indices], align='center', color="#5371A3")
plt.xticks(range(len(avg_importances)), [labels[i] for i in indices], rotation=90)
plt.xlabel('Features')
plt.ylabel('Scaled Importance')
plt.title('XGBoost Feature Importances (Normalized and Scaled)')
plt.tight_layout()

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom')

plt.savefig("feature_importance_scaled.png", dpi=600, bbox_inches="tight")
plt.show()