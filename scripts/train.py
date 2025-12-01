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

def imputer(dataArr):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean', axis=0)
    imp.fit(dataArr)
    dataArr_full = imp.transform(dataArr)
    return dataArr_full

def normData (dataX, dataY):
    scaler = StandardScaler()
    X = dataX
    Y = dataY
    return X, Y

def splitDataHO (X, Y, testSize, random_state):
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=testSize, random_state=random_state)
    return trainX, testX, trainY, testY

def split_train_predict(X, Y, number_train_test):
    items_number = len(X)
    train_test_data_Arr = []; predict_data_Arr = []; train_test_label_Arr = []; predict_label_Arr = []
    for i in range(number_train_test):
        train_test_data = X[i]
        train_test_label = Y[i]
        train_test_data_Arr.append(train_test_data)
        train_test_label_Arr.append(train_test_label)
    for j in range(number_train_test, items_number):
        predict_data = X[j]
        predict_label = Y[j]
        predict_data_Arr.append(predict_data)
        predict_label_Arr.append(predict_label)
    return np.array(train_test_data_Arr), np.array(train_test_label_Arr), np.array(predict_data_Arr), np.array(predict_label_Arr)

def ModelEvaluationClassifier(testY, prediction):
    AUC = roc_auc_score(testY, prediction)
    Accuracy = accuracy_score(testY, prediction)
    Precision = precision_score(testY, prediction)
    Recall = recall_score(testY, prediction)
    return AUC, Accuracy, Precision, Recall

def plot_confusion_matrix(y_true, cm, title='Confusion Matrix', cmap=plt.cm.Blues):
    labels = list(set(y_true))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, norm=colors.Normalize(vmin=0.0, vmax=1.0), aspect='auto',
                origin='lower')
    plt.title(title, fontsize=28, fontweight='bold', fontname='Arial')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=24, width=2, length=6)
    cbar.set_label('Ratio of Each Class', rotation=90, fontsize=26, fontweight='bold', fontname='Arial')
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_fontweight('bold')
        l.set_family('Arial')
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90, fontsize=28, fontweight='bold', fontname='Arial')
    plt.yticks(xlocations, labels, fontsize=28, fontweight='bold', fontname='Arial')
    plt.ylabel('True label', fontsize=28, fontweight='bold', fontname='Arial')
    plt.xlabel('Predicted label', fontsize=28, fontweight='bold', fontname='Arial')

def ConfusionMatrix(y_true, y_pred, Loop_Step, screen_step):
    labels = list(set(y_true))
    tick_marks = np.array(range(len(labels))) + 0.5
    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 8), dpi=330)
    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=28, va='center', ha='center', fontweight='bold')
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    plot_confusion_matrix(y_true, cm_normalized, title='Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(result_path + 'Normalized_Confusion_Matrix_' + str(Loop_Step) + "_" + str(screen_step) + '.png', format='png')
    plt.close()

def hyper_parameters_plot(parameters, trials, Loop_Step, screen_step):
    matplotlib_axes_logger.setLevel('ERROR')
    f, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), dpi=330)
    cmap = plt.cm.jet
    for i, val in enumerate(parameters):
        xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
        ys = [-t['result']['loss'] for t in trials.trials]
        xs, ys = zip(*sorted(zip(xs, ys)))
        axes[int(i / 3), int(i % 3)].scatter(xs, ys, s=50, linewidth=0.01, alpha=0.5, norm=0.5,
                                             c=cmap(float(i) / len(parameters)))
        axes[int(i / 3), int(i % 3)].set_title(val, fontsize=20, fontweight="bold", fontname='Arial')
        axes[int(i / 3), int(i % 3)].tick_params(labelsize=18, direction='out', width=2, length=6)
        axes[int(i / 3), int(i % 3)].set_ylim((0.5, 1))
        labels = axes[int(i / 3), int(i % 3)].get_xticklabels() + axes[int(i / 3), int(i % 3)].get_yticklabels()
        [label.set_fontname('Arial') for label in labels]
        [label.set_fontweight('bold') for label in labels]
        axes[int(i / 3), int(i % 3)].spines['top'].set_linewidth(2.5)
        axes[int(i / 3), int(i % 3)].spines['bottom'].set_linewidth(2.5)
        axes[int(i / 3), int(i % 3)].spines['right'].set_linewidth(2.5)
        axes[int(i / 3), int(i % 3)].spines['left'].set_linewidth(2.5)
    plt.tight_layout()
    plt.savefig(result_path + 'hyper_parameters_trials_' + str(Loop_Step) + "_" + str(screen_step) + '.png', format='png')
    plt.close()
    return

def ROCcurve (mean_train_fpr, mean_test_fpr,mean_tpr_train, mean_tpr_test, std_tpr_train, std_tpr_test, std_auc_train, std_auc_test,
             mean_auc_train, mean_auc_test, Loop_Step, screen_step):
    plt.figure(figsize=(10, 10), dpi=330)
    lw = 3
    if Loop_Step == "average":
        tprs_train_upper = np.minimum(mean_tpr_train + std_tpr_train, 1)
        tprs_train_lower = np.maximum(mean_tpr_train - std_tpr_train, 0)
        tprs_test_upper = np.minimum(mean_tpr_test + std_tpr_test, 1)
        tprs_test_lower = np.maximum(mean_tpr_test - std_tpr_test, 0)
        plt.fill_between(mean_train_fpr, tprs_train_lower, tprs_train_upper, color='grey', alpha=.5)
        plt.fill_between(mean_test_fpr, tprs_test_lower, tprs_test_upper, color='grey', alpha=.5,
                         label=r'$\pm$ 1 std. dev.')
        plt.plot(mean_train_fpr, mean_tpr_train, color=(11 / 255, 52 / 255, 110 / 255),
             label=r'Mean Training ROC (AUC = %0.2f $\pm$ %0.3f)' % (mean_auc_train, std_auc_train),
             lw=lw, linestyle='-')
        plt.plot(mean_test_fpr, mean_tpr_test, color=(208 / 255, 16 / 255, 76 / 255),
                 label=r'Mean Test ROC (AUC = %0.2f $\pm$ %0.3f)' % (mean_auc_test, std_auc_test),
                 lw=lw, linestyle='-')
    else:
        plt.plot(mean_train_fpr, mean_tpr_train,
                 label='Training AUC (area = {0:0.2f})'
                       ''.format(mean_auc_train),
                 color=(11 / 255, 52 / 255, 110 / 255), linestyle='-', linewidth=lw)
        plt.plot(mean_test_fpr, mean_tpr_test,
                 label='Test AUC (area = {0:0.2f})'
                       ''.format(mean_auc_test),
                 color=(208 / 255, 16 / 255, 76 / 255), linestyle='-', linewidth=lw)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=26, fontweight="bold", fontname='Arial')
    plt.ylabel('True Positive Rate', fontsize=26, fontweight="bold", fontname='Arial')
    plt.title('ROC Curve of Classification', fontsize=26, fontweight="bold", fontname='Arial')
    plt.legend(loc="lower right")
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    bwith = 3
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    plt.setp(ltext, fontsize=24, fontweight='bold', fontname='Arial')
    plt.tick_params(direction='out', width=2, length=6)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Arial') for label in labels]
    [label.set_fontweight('bold') for label in labels]
    [label.set_fontsize(24) for label in labels]
    plt.tight_layout()
    plt.savefig(result_path + "ROC_Curve_of_Classification_" + str(Loop_Step) + "_" + str(screen_step) + ".png", format="png")
    plt.close()

def GBClassifier(trainX, testX, trainY, testY, max_evals, Loop_Step, screen_step):
    GBC_model_score_filename = open(result_path + "hyper_parameter_selection_" + str(Loop_Step) + "_" + str(screen_step) + ".dat", 'a')
    predicted_testY_file = open(result_path + "predicted_testY_" + str(Loop_Step) + "_" + str(screen_step) + ".dat", 'a')
    predicted_trainY_file = open(result_path + "predicted_trainY_" + str(Loop_Step) + "_" + str(screen_step) + ".dat", 'a')
    ROC_file = open(result_path + "tpr&fpr_" + str(Loop_Step) + "_" + str(screen_step) + ".dat", 'a')

    parameter_space_gbm = {"colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
                           "max_depth": hp.quniform("max_depth", 3, 10, 1),
                           "n_estimators": hp.quniform("n_estimators", 10, 200, 1),
                           "learning_rate": hp.uniform("learning_rate", 0.01, 0.5),
                           "subsample": hp.uniform("subsample", 0.5, 1),
                           "min_child_weight": hp.uniform("min_child_weight", 0.5, 10),
                           "gamma": hp.uniform("gamma", 0.01, 1)
                           }
    count = 0
    def function(argsDict):
        colsample_bytree = argsDict["colsample_bytree"]
        max_depth = argsDict["max_depth"]
        n_estimators = argsDict['n_estimators']
        learning_rate = argsDict["learning_rate"]
        subsample = argsDict["subsample"]
        min_child_weight = argsDict["min_child_weight"]
        gamma = argsDict["gamma"]

        clf = xgb.XGBClassifier(nthread=4,
                                colsample_bytree=colsample_bytree,
                                max_depth=int(max_depth),
                                n_estimators=int(n_estimators),
                                learning_rate=learning_rate,
                                subsample=subsample,
                                min_child_weight=min_child_weight,
                                objective="binary:logistic",
                                gamma=gamma,
                                random_state=int(42),
                                scale_pos_weight=3
                                )
        eec = EasyEnsembleClassifier(n_estimators=5, estimator=clf, random_state=42)
        eec.fit(trainX, trainY)
        prediction = eec.predict(testX)
        nonlocal count
        count = count + 1
        AUC, Accuracy, Precision, Recall = ModelEvaluationClassifier(testY, prediction)
        print("Screen_step: %s, Loop: %s, No.%s，AUC: %f, Accuracy: %f, Precision: %f, Recall: %f" %
              (str(screen_step), str(Loop_Step), str(count), AUC, Accuracy, Precision, Recall))
        print("Screen_step: %s, Loop: %s, No.%s，AUC: %f, Accuracy: %f, Precision: %f, Recall: %f" %
              (str(screen_step), str(Loop_Step), str(count), AUC, Accuracy, Precision, Recall), argsDict,
              file=GBC_model_score_filename)
        return -AUC

    trials = Trials()
    best = fmin(function, parameter_space_gbm, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    parameters = ['colsample_bytree', 'max_depth', 'n_estimators', 'learning_rate', 'subsample', 'min_child_weight']
    hyper_parameters_plot(parameters, trials, Loop_Step, screen_step)

    colsample_bytree = best["colsample_bytree"]
    max_depth = best["max_depth"]
    n_estimators = best['n_estimators']
    learning_rate = best["learning_rate"]
    subsample = best["subsample"]
    min_child_weight = best["min_child_weight"]
    gamma = best["gamma"]

    based_best_model = xgb.XGBClassifier(nthread=4,
                                   colsample_bytree=colsample_bytree,
                                   max_depth=int(max_depth),
                                   n_estimators=int(n_estimators), 
                                   learning_rate=learning_rate,
                                   subsample=subsample, 
                                   min_child_weight=min_child_weight, 
                                   objective="binary:logistic",
                                   gamma=gamma,
                                   random_state=int(42),
                                   scale_pos_weight=3,
                                   )

    best_model = EasyEnsembleClassifier(n_estimators=5, estimator=based_best_model, random_state=42)
    based_best_model.fit(trainX, trainY)
    best_model.fit(trainX, trainY)
    best_model_pred_trainY = best_model.predict(trainX)
    best_model_pred_testY = best_model.predict(testX)

    AUC_train, Accuracy_train, Precision_train, Recall_train = ModelEvaluationClassifier(trainY, best_model_pred_trainY)
    AUC_test, Accuracy_test, Precision_test, Recall_test = ModelEvaluationClassifier(testY, best_model_pred_testY)
    print('best model parameters: ', best, file=GBC_model_score_filename)
    print('The_best_model_train_score: AUC: %f, Accuracy: %f, Precision: %f, Recall: %f' % (
        AUC_train, Accuracy_train, Precision_train, Recall_train), file=GBC_model_score_filename)
    print('The_best_model_test_score: AUC: %f, Accuracy: %f, Precision: %f, Recall: %f' % (
        AUC_test, Accuracy_test, Precision_test, Recall_test), file=GBC_model_score_filename)

    # 保存模型
    with open(result_path + "best_model_" + str(Loop_Step) + "_" + str(screen_step) + '.pickle', 'wb') as f:
        pickle.dump(best_model, f)
    with open(result_path + "based_best_model_" + str(Loop_Step) + "_" + str(screen_step) + '.pickle', 'wb') as f:
        pickle.dump(based_best_model, f)

    for y in best_model_pred_testY:
        print(y, file=predicted_testY_file)
    for y in best_model_pred_trainY:
        print(y, file=predicted_trainY_file)

    pred_trainY_proba = best_model.predict_proba(trainX)
    pred_testY_proba = best_model.predict_proba(testX)
    pred_trainY_grd = pred_trainY_proba[:, 1]
    pred_testY_grd = pred_testY_proba[:, 1]
    fpr_train, tpr_train, _ = roc_curve(trainY, pred_trainY_grd)
    fpr_test, tpr_test, _ = roc_curve(testY, pred_testY_grd)
    print('fpr_train: ', [i for i in fpr_train], file=ROC_file)
    print('tpr_train: ', [i for i in tpr_train], file=ROC_file)
    print('fpr_test: ', [i for i in fpr_test], file=ROC_file)
    print('tpr_test: ', [i for i in tpr_test], file=ROC_file)

    GBC_model_score_filename.close()
    predicted_testY_file.close()
    predicted_trainY_file.close()
    ROC_file.close()
    return based_best_model, best_model, best_model_pred_trainY, best_model_pred_testY, pred_trainY_proba, pred_testY_proba, \
           AUC_train, Accuracy_train, Precision_train, Recall_train,\
           AUC_test, Accuracy_test, Precision_test, Recall_test, \
           fpr_train, tpr_train, fpr_test, tpr_test

def path_mkdir (path, result_subfile,screen_step):
    feature_engineering_result_file = result_file + str(result_subfile) + str(screen_step)
    if os.path.exists(result_file) == False:
        os.mkdir(path + result_file)
    if os.path.exists(feature_engineering_result_file) == False:
        os.mkdir(path + feature_engineering_result_file)
    result_path = path + feature_engineering_result_file + "/"
    return result_path

def train_test(trainX, testX, trainY, testY, screen_step):
    features_importanceArr = []; mean_features_importanceArr = []; std_features_importanceArr = []
    sum_pred_trainY_proba = np.zeros((len(trainY), 2)); sum_pred_testY_proba = np.zeros((len(testY), 2))
    tprs_train = []; aucs_train = []; tprs_test = []; aucs_test = []
    mean_train_fpr = np.linspace(0, 1, len(trainY))
    mean_test_fpr = np.linspace(0, 1, len(testY))
    for n in range(LoopStepMin, LoopStepMax):
        based_best_model, best_model, best_model_pred_trainY, best_model_pred_testY, pred_trainY_proba, pred_testY_proba, \
        AUC_train, Accuracy_train, Precision_train, Recall_train, \
        AUC_test, Accuracy_test, Precision_test, Recall_test, \
        fpr_train, tpr_train, fpr_test, tpr_test \
            = GBClassifier(trainX, testX, trainY, testY, HyperParameter_Step, n, screen_step)
        sum_pred_testY_proba += pred_testY_proba
        sum_pred_trainY_proba += pred_trainY_proba
        interp_tpr_train = np.interp(mean_train_fpr, fpr_train, tpr_train)
        interp_tpr_test = np.interp(mean_test_fpr, fpr_test, tpr_test)
        interp_tpr_train[0] = 0.0; interp_tpr_test[0] = 0.0
        aucs_train.append(AUC_train)
        aucs_test.append(AUC_test)
        tprs_train.append(interp_tpr_train)
        tprs_test.append(interp_tpr_test)
        features_importance = based_best_model.feature_importances_
        features_importanceArr.append(features_importance)

    for k in range(np.shape(features_importanceArr)[1]):
        sample = np.array(features_importanceArr)[:, k]
        features_importance_mean = np.mean(sample)
        features_importance_std = np.std(sample, ddof=0)
        mean_features_importanceArr.append(features_importance_mean)
        std_features_importanceArr.append(features_importance_std)

    average_pred_trainY_proba = (sum_pred_trainY_proba / abs(LoopStepMin - LoopStepMax))
    average_pred_testY_proba = (sum_pred_testY_proba / abs(LoopStepMin - LoopStepMax))
    average_pred_trainY = average_pred_trainY_proba.argmax(axis=1)
    average_pred_testY = average_pred_testY_proba.argmax(axis=1)
    ConfusionMatrix(trainY, average_pred_trainY, "average_train", screen_step)
    ConfusionMatrix(testY, average_pred_testY, "average_test", screen_step)
    mean_tpr_train = np.mean(tprs_train, axis=0)
    mean_tpr_test = np.mean(tprs_test, axis=0)
    mean_tpr_train[-1] = 1.0
    mean_tpr_test[-1] = 1.0
    mean_auc_train = roc_auc_score(trainY, average_pred_trainY)
    mean_auc_test = roc_auc_score(testY, average_pred_testY)
    std_tpr_train = np.std(tprs_train, axis=0)
    std_tpr_test = np.std(tprs_test, axis=0)
    std_auc_train = np.std(aucs_train)
    std_auc_test = np.std(aucs_test)
    ROCcurve(mean_train_fpr, mean_test_fpr, mean_tpr_train, mean_tpr_test, std_tpr_train, std_tpr_test, std_auc_train, std_auc_test,
             mean_auc_train, mean_auc_test, "average", screen_step)
    train_font = open(result_path + "predicted_train_average_" + str(screen_step) + ".dat", "a")
    test_font = open(result_path + "predicted_test_average_" + str(screen_step) + ".dat", "a")
    features_importance_font = open(result_path + "features_importance_average_" + str(screen_step) + ".dat", "a")
    for i, j in zip(average_pred_trainY_proba, average_pred_trainY):
        print(i, j, file=train_font)
    for i, j in zip(average_pred_testY_proba, average_pred_testY):
        print(i, j, file=test_font)
    for i, j, k in zip(features_name, mean_features_importanceArr, std_features_importanceArr):
        print(i, j, k, file=features_importance_font)
    train_font.close()
    test_font.close()
    features_importance_font.close()
    return average_pred_trainY, average_pred_testY, mean_features_importanceArr, std_features_importanceArr

def predict(trainX, trainY, testX, testY, predictX, screen_step):
    print("Predicting......")
    predict_font = open(result_path + "predicted_predict_average_" + str(screen_step) + ".dat", "a")

    sum_pred_predictY_proba = np.zeros(len(predictX))
    average_pred_predictY_array = []
    for i in range(LoopStepMin, LoopStepMax):
        best_model_pred_trainY_array = []; best_model_pred_testY_array = []
        model_file = result_path + "best_model_" + str(i) + "_FF.pickle"
        best_model = pickle.load(open(model_file, 'rb'))
        best_model_pred_testY = best_model.predict(testX)
        best_model_pred_trainY = best_model.predict(trainX)
        best_model_pred_predictY = best_model.predict(predictX)
        for train in best_model_pred_trainY:
            if train >= 0.5:
                new_train = 1
            else:
                new_train = 0
            best_model_pred_trainY_array.append(new_train)
        for test in best_model_pred_testY:
            if test >= 0.5:
                new_test = 1
            else:
                new_test = 0
            best_model_pred_testY_array.append(new_test)

        AUC_train, Accuracy_train, Precision_train, Recall_train = ModelEvaluationClassifier(trainY, best_model_pred_trainY_array)
        AUC_test, Accuracy_test, Precision_test, Recall_test = ModelEvaluationClassifier(testY, best_model_pred_testY_array)
        sum_pred_predictY_proba += best_model_pred_predictY

        print('AUC_train: %f, Accuracy_train: %f, Precision_train: %f, Recall_train: %f' % (
            AUC_train, Accuracy_train, Precision_train, Recall_train))
        print('AUC_test: %f, Accuracy_test: %f, Precision_test: %f, Recall_test: %f' % (
            AUC_test, Accuracy_test, Precision_test, Recall_test))

    average_pred_predictY_proba = (sum_pred_predictY_proba / abs(LoopStepMin - LoopStepMax))
    for item in average_pred_predictY_proba:
        if item >= 0.5:
            average_pred_predictY = 1
        else:
            average_pred_predictY = 0
        average_pred_predictY_array.append(average_pred_predictY)

    for i, j in zip(average_pred_predictY_proba, average_pred_predictY_array):
        print(i, j, file=predict_font)
    predict_font.close()
    return

def feature_importance(features_name, features, screen_step):
    normdata_filename = open(result_path + "norm_dataset_" + str(screen_step) +".dat", "a")
    data_filename = open(result_path + "dataset_" + str(screen_step) +".dat", "a")
    if n_fixed_features != 0:
        vector_features = features[:, n_fixed_features: np.shape(features)[1]]
        vector_features_name = list(features_name[n_fixed_features: np.shape(features)[1]])
        new_features = np.concatenate((fixed_features, vector_features), axis=1)
        norm_features, norm_label = normData(new_features, label)
        new_features_name = np.concatenate((initial_fixed_features_name, vector_features_name), axis=0)
        print("ID, formula, prototype", [name for name in new_features_name], "label", file=normdata_filename)
        for i, j, k, x, y in zip(ID, formula, prototype, norm_features, label):
            print(i, j, k, [item for item in x], y, file=normdata_filename)
        print("ID, formula, prototype", [name for name in new_features_name], "label", file=data_filename)
        for i, j, k, x, y in zip(ID, formula, prototype, features, label):
            print(i, j, k, [item for item in x], y, file=data_filename)
    else:
        vector_features = features
        vector_features_name = list(features_name)
        norm_vector_features, norm_label = normData(vector_features, label)
        norm_features = norm_vector_features
        print("ID, formula, prototype", [name for name in vector_features_name], "label", file=normdata_filename)
        for i, j, k, x, y in zip(ID, formula, prototype, norm_features, label):
            print(i, j, k, [item for item in x], y, file=normdata_filename)
        print("ID, formula, prototype", [name for name in vector_features_name], "label", file=data_filename)
        for i, j, k, x, y in zip(ID, formula, prototype, vector_features, label):
            print(i, j, k, [item for item in x], y, file=data_filename)


    train_test_data_Arr, train_test_label_Arr, predict_data_Arr, predict_label_Arr = split_train_predict(norm_features,
                                                                                                         norm_label,
                                                                                                         number_sample)
    trainX, testX, trainY, testY = splitDataHO(train_test_data_Arr, train_test_label_Arr, TestSetRatio, RandomSeed)

    average_pred_trainY, average_pred_testY, \
    mean_features_importanceArr, std_features_importanceArr \
        = train_test(trainX, testX, trainY, testY, screen_step)

    normdata_filename.close()
    data_filename.close()
    return average_pred_trainY, average_pred_testY, \
           mean_features_importanceArr, std_features_importanceArr, \
           fixed_features_name, fixed_features, vector_features_name, vector_features

def fixed_feature_engineering(mean_features_importanceArr, std_features_importanceArr,
                        fixed_features_name, fixed_features, vector_features_name,vector_features, screen_step):
    sorted_features_importance = []; sorted_features_importance_std = []
    shape_fixed_features = np.shape(fixed_features)
    fixed_features_importance = np.float(np.sum(mean_features_importanceArr[0: shape_fixed_features[1]]))
    fixed_features_importance_std = np.sum(std_features_importanceArr[0: shape_fixed_features[1]]) / \
                                    shape_fixed_features[1]
    vector_features_importance = mean_features_importanceArr[shape_fixed_features[1]: len(mean_features_importanceArr)]
    vector_features_importance_std = std_features_importanceArr[
                                     shape_fixed_features[1]: len(mean_features_importanceArr)]
    sorted_idx = np.argsort(vector_features_importance)[::-1]
    sorted_vector_features_name = np.array(vector_features_name)[sorted_idx]
    sorted_vector_features_importance = np.array(vector_features_importance)[sorted_idx]
    sorted_vector_features_importance_std = np.array(vector_features_importance_std)[sorted_idx]
    row_number = np.shape(vector_features)[0]
    vfeature_column_number = np.shape(vector_features)[1]
    sorted_vetor_features = np.zeros((row_number, vfeature_column_number))
    for i, j in zip(sorted_idx, np.arange(0, vfeature_column_number)):
        sorted_vetor_features[:, j] = vector_features[:, i]
    sorted_features = np.concatenate((fixed_features, sorted_vetor_features), axis=1)
    selected_sorted_vector_features_name = np.delete(sorted_vector_features_name, -1)
    selected_sorted_features = np.delete(sorted_features, -1, axis=1)
    sorted_features_name = np.concatenate((fixed_features_name, selected_sorted_vector_features_name), axis=0)
    sorted_features_importance.append(fixed_features_importance)
    for item in sorted_vector_features_importance:
        sorted_features_importance.append(item)
    sorted_features_importance_std.append(fixed_features_importance_std)
    for item in sorted_vector_features_importance_std:
        sorted_features_importance_std.append(item)
    font = open(result_path + "sorted_relative_feature_importance_" + str(screen_step) + ".dat", 'a')
    for i, j, k in zip(sorted_features_name, sorted_features_importance, sorted_features_importance_std):
        print(i, j, k, file=font)
    font.close()
    return sorted_features_name, selected_sorted_features

def vector_feature_engineering(mean_features_importanceArr, std_features_importanceArr,
                        vector_features_name,vector_features, screen_step):
    sorted_features_importance = []; sorted_features_importance_std = []
    vector_features_importance = mean_features_importanceArr
    vector_features_importance_std = std_features_importanceArr
    sorted_idx = np.argsort(vector_features_importance)[::-1]
    sorted_vector_features_name = np.array(vector_features_name)[sorted_idx]
    sorted_vector_features_importance = np.array(vector_features_importance)[sorted_idx]
    sorted_vector_features_importance_std = np.array(vector_features_importance_std)[sorted_idx]
    row_number = np.shape(vector_features)[0]
    vfeature_column_number = np.shape(vector_features)[1]
    sorted_vetor_features = np.zeros((row_number, vfeature_column_number))
    for i, j in zip(sorted_idx, np.arange(0, vfeature_column_number)):
        sorted_vetor_features[:, j] = vector_features[:, i]
    sorted_features = sorted_vetor_features
    selected_sorted_vector_features_name = np.delete(sorted_vector_features_name, -1)
    selected_sorted_features = np.delete(sorted_features, -1, axis=1)
    sorted_features_name = selected_sorted_vector_features_name
    for item in sorted_vector_features_importance:
        sorted_features_importance.append(item)
    for item in sorted_vector_features_importance_std:
        sorted_features_importance_std.append(item)
    font = open(result_path + "sorted_relative_feature_importance_" + str(screen_step) + ".dat", 'a')
    for i, j, k in zip(sorted_features_name, sorted_features_importance, sorted_features_importance_std):
        print(i, j, k, file=font)
    font.close()
    return sorted_features_name, selected_sorted_features

if __name__ == "__main__":
    path = os.getcwd() + "/"
    result_file = "Results"  
    result_subfile = "/Feature_Engineering_Result_"  
    workbook = "data_xitu_LOOP0.xlsx"  
    sheet = str("Sheet1")  
    n_fixed_features = 8 
    fixed_features_name = list(['Features']) 
    HyperParameter_Step = 100 
    LoopStepMin = 0
    LoopStepMax = 10
    RandomSeed = 42
    TestSetRatio = 0.2
    number_sample = 26

    ID, formula, prototype, prototype_2, features, label, features_name = data_load(workbook, sheet)


    fixed_features = features[:, 0: n_fixed_features]
    initial_fixed_features_name = features_name[0:n_fixed_features]
    ScreenStep = np.shape(features[:, n_fixed_features: np.shape(features)[1]])[1]

    if np.shape(features)[1] > n_fixed_features:
        if n_fixed_features != 0:
            for i in range(0, ScreenStep):
                screen_step = i
                result_path = path_mkdir(path, result_subfile, screen_step)
                average_pred_trainY, average_pred_testY, \
                mean_features_importanceArr, std_features_importanceArr, \
                fixed_features_name, fixed_features, vector_features_name, vector_features = feature_importance(
                    features_name,
                    features,
                    screen_step)
                selected_sorted_features_name, selected_sorted_features = \
                    fixed_feature_engineering(mean_features_importanceArr, std_features_importanceArr, fixed_features_name,
                                            fixed_features, vector_features_name,
                                            vector_features, screen_step)
                features_name = np.concatenate(
                    (initial_fixed_features_name, selected_sorted_features_name[1: len(selected_sorted_features_name)]),
                    axis=0)
                features = selected_sorted_features
        else:
            for i in range(0, ScreenStep):
                screen_step = i
                result_path = path_mkdir(path, result_subfile, screen_step)
                average_pred_trainY, average_pred_testY, \
                mean_features_importanceArr, std_features_importanceArr, \
                fixed_features_name, fixed_features, vector_features_name, vector_features = feature_importance(
                    features_name,
                    features,
                    screen_step)
                selected_sorted_features_name, selected_sorted_features = \
                    vector_feature_engineering(mean_features_importanceArr, std_features_importanceArr,
                                            vector_features_name, vector_features, screen_step)
                features_name = selected_sorted_features_name

                features = selected_sorted_features

    else:
        screen_step = "FF"
        result_path = path_mkdir(path, result_subfile, screen_step)
        norm_features, norm_label = normData(features, label)
        train_test_data_Arr, train_test_label_Arr, predict_data_Arr, predict_label_Arr = split_train_predict(norm_features,
                                                                                                            norm_label,
                                                                                                            number_sample)
        trainX, testX, trainY, testY = splitDataHO(train_test_data_Arr, train_test_label_Arr, TestSetRatio, RandomSeed)
        predictX = predict_data_Arr; predictY = predict_label_Arr
        average_pred_trainY, average_pred_testY, \
        mean_features_importanceArr, std_features_importanceArr \
            = train_test(trainX, testX, trainY, testY, screen_step)
        if number_sample != len(label):
            predict(trainX, trainY, testX, testY, predictX, screen_step)