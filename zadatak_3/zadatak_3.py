import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import confusion_matrix, make_scorer, f1_score, roc_auc_score, precision_score, recall_score, \
    accuracy_score
import os
import joblib


def create_directories():
    os.makedirs('results', exist_ok=True)
    os.makedirs('models/70-15-15_Split', exist_ok=True)
    os.makedirs('models/60-20-20_Split', exist_ok=True)


def preprocess_data(X_train, X_val, X_test, log_features, data_type):
    if data_type == 'Normal':
        return X_train, X_val, X_test

    X_train_processed = X_train.copy()
    X_val_processed = X_val.copy()
    X_test_processed = X_test.copy()

    for column in log_features:
        min_value = min(X_train_processed[column].min(), X_val_processed[column].min(), X_test_processed[column].min())
        if min_value <= 0:
            shift = -min_value + 1
            X_train_processed[column] = np.log(X_train_processed[column] + shift)
            X_val_processed[column] = np.log(X_val_processed[column] + shift)
            X_test_processed[column] = np.log(X_test_processed[column] + shift)
        else:
            X_train_processed[column] = np.log(X_train_processed[column])
            X_val_processed[column] = np.log(X_val_processed[column])
            X_test_processed[column] = np.log(X_test_processed[column])

    if data_type == 'Log':
        return X_train_processed, X_val_processed, X_test_processed
    else:  # Mixed
        X_train_mixed = X_train.copy()
        X_val_mixed = X_val.copy()
        X_test_mixed = X_test.copy()
        X_train_mixed[log_features] = X_train_processed[log_features]
        X_val_mixed[log_features] = X_val_processed[log_features]
        X_test_mixed[log_features] = X_test_processed[log_features]
        return X_train_mixed, X_val_mixed, X_test_mixed


def train_models(X, y, data_type, split_name):
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True),
        'KNN': KNeighborsClassifier(),
        'LightGBM': lgb.LGBMClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42)
    }

    for name, model in models.items():
        model.fit(X, y)
        joblib.dump(model, f'models/{split_name}/{name.replace(" ", "_")}_{data_type}.joblib')

    return models


def perform_cv(model, X, y):
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1_weighted': make_scorer(f1_score, average='weighted'),
        'f1_macro': make_scorer(f1_score, average='macro'),
        'roc_auc_ovr': make_scorer(roc_auc_score, multi_class='ovr', response_method='predict_proba'),
        'precision_weighted': make_scorer(precision_score, average='weighted'),
        'recall_weighted': make_scorer(recall_score, average='weighted')
    }

    scores = cross_validate(model, X, y, cv=5, scoring=scoring)

    return {
        'CV Accuracy': scores['test_accuracy'].mean(),
        'CV Accuracy Std': scores['test_accuracy'].std(),
        'CV F1 (Weighted)': scores['test_f1_weighted'].mean(),
        'CV F1 (Weighted) Std': scores['test_f1_weighted'].std(),
        'CV F1 (Macro)': scores['test_f1_macro'].mean(),
        'CV F1 (Macro) Std': scores['test_f1_macro'].std(),
        'CV ROC AUC': scores['test_roc_auc_ovr'].mean(),
        'CV ROC AUC Std': scores['test_roc_auc_ovr'].std(),
        'CV Precision': scores['test_precision_weighted'].mean(),
        'CV Precision Std': scores['test_precision_weighted'].std(),
        'CV Recall': scores['test_recall_weighted'].mean(),
        'CV Recall Std': scores['test_recall_weighted'].std()
    }


def plot_confusion_matrices(confusion_matrices, split_name):
    fig, axs = plt.subplots(6, 3, figsize=(15, 30))
    fig.suptitle(f'Confusion Matrices - {split_name}', fontsize=16)

    for i, (model_name, data_types) in enumerate(confusion_matrices.items()):
        for j, (data_type, cm) in enumerate(data_types.items()):
            ax = axs[i, j]
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.set_title(f'{model_name} - {data_type}')
            ax.set_xlabel('Predicted label')
            ax.set_ylabel('True label')

            thresh = cm.max() / 2.
            for i_cm, j_cm in np.ndindex(cm.shape):
                ax.text(j_cm, i_cm, format(cm[i_cm, j_cm], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i_cm, j_cm] > thresh else "black")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'results/confusion_matrices_{split_name}.png')
    plt.close(fig)


def run_experiment(X, y, train_size, val_size, test_size, log_features, split_name):
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    # Second split: separate validation set from remaining data
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size / (train_size + val_size),
                                                      stratify=y_temp, random_state=42)

    # Preprocess and scale data
    data_types = ['Normal', 'Log', 'Mixed']
    scaled_data = {}

    for data_type in data_types:
        X_train_processed, X_val_processed, X_test_processed = preprocess_data(X_train, X_val, X_test, log_features,
                                                                               data_type)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_processed)
        X_val_scaled = scaler.transform(X_val_processed)
        X_test_scaled = scaler.transform(X_test_processed)

        scaled_data[data_type] = (X_train_scaled, X_val_scaled, X_test_scaled)

    # Train models and perform cross-validation
    cv_results = []
    trained_models = {}

    for data_type in data_types:
        X_train_scaled, _, _ = scaled_data[data_type]
        models = train_models(X_train_scaled, y_train, data_type, split_name)
        trained_models[data_type] = models

        for model_name, model in models.items():
            scores = perform_cv(model, X_train_scaled, y_train)
            scores['Model'] = model_name
            scores['Data Type'] = data_type
            cv_results.append(scores)

    cv_results_df = pd.DataFrame(cv_results)
    cv_results_df = cv_results_df[['Model', 'Data Type',
                                   'CV Accuracy', 'CV Accuracy Std',
                                   'CV F1 (Weighted)', 'CV F1 (Weighted) Std',
                                   'CV F1 (Macro)', 'CV F1 (Macro) Std',
                                   'CV ROC AUC', 'CV ROC AUC Std',
                                   'CV Precision', 'CV Precision Std',
                                   'CV Recall', 'CV Recall Std']]
    cv_results_df = cv_results_df.sort_values(['Model', 'Data Type']).reset_index(drop=True)

    # Evaluate models on test data
    test_results = []
    confusion_matrices = {model_name: {} for model_name in trained_models['Normal'].keys()}

    for data_type in data_types:
        _, _, X_test_scaled = scaled_data[data_type]
        models = trained_models[data_type]

        for model_name, model in models.items():
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)

            test_results.append({
                'Model': model_name,
                'Data Type': data_type,
                'Test Accuracy': accuracy_score(y_test, y_pred),
                'Test F1 (Weighted)': f1_score(y_test, y_pred, average='weighted'),
                'Test F1 (Macro)': f1_score(y_test, y_pred, average='macro'),
                'Test ROC AUC': roc_auc_score(y_test, y_pred_proba, multi_class='ovr'),
                'Test Precision': precision_score(y_test, y_pred, average='weighted'),
                'Test Recall': recall_score(y_test, y_pred, average='weighted')
            })
            confusion_matrices[model_name][data_type] = confusion_matrix(y_test, y_pred)

    test_results_df = pd.DataFrame(test_results)
    test_results_df = test_results_df[
        ['Model', 'Data Type', 'Test Accuracy', 'Test F1 (Weighted)', 'Test F1 (Macro)', 'Test ROC AUC',
         'Test Precision', 'Test Recall']]
    test_results_df = test_results_df.sort_values(['Model', 'Data Type']).reset_index(drop=True)

    # Plot confusion matrices
    plot_confusion_matrices(confusion_matrices, split_name)

    return cv_results_df, test_results_df


# Create necessary directories
create_directories()

# Load your data
df = pd.read_csv('training_data.csv')
X = df.drop('Class', axis=1)
y = df['Class']

# Define log features
log_features = ['Blue', 'Green', 'NIR', 'Red']

# Run experiments
cv_results_70_15_15, test_results_70_15_15 = run_experiment(X, y, 0.7, 0.15, 0.15, log_features, '70-15-15_Split')
cv_results_60_20_20, test_results_60_20_20 = run_experiment(X, y, 0.6, 0.2, 0.2, log_features, '60-20-20_Split')

# Save results to CSV
cv_results_70_15_15.to_csv('results/cross_validation_multi_metric_results_70_15_15_split.csv', index=False)
test_results_70_15_15.to_csv('results/final_test_results_70_15_15_split.csv', index=False)
cv_results_60_20_20.to_csv('results/cross_validation_multi_metric_results_60_20_20_split.csv', index=False)
test_results_60_20_20.to_csv('results/final_test_results_60_20_20_split.csv', index=False)
