import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import confusion_matrix
import os
import joblib


def create_directories():
    os.makedirs('results', exist_ok=True)
    os.makedirs('models/70-15-15_Split', exist_ok=True)
    os.makedirs('models/60-20-20_Split', exist_ok=True)


def preprocess_data(X, log_features):
    X_normal = X.copy()
    X_log = X.copy()
    X_mixed = X.copy()

    for column in log_features:
        min_value = X_log[column].min()
        if min_value <= 0:
            shift = -min_value + 1
            X_log[column] = np.log(X_log[column] + shift)
            X_mixed[column] = np.log(X_mixed[column] + shift)
        else:
            X_log[column] = np.log(X_log[column])
            X_mixed[column] = np.log(X_mixed[column])

    return X_normal, X_log, X_mixed


def train_models(X, y, data_type, split_name):
    models = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42),
        'KNN': KNeighborsClassifier(),
        'LightGBM': lgb.LGBMClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42)
    }

    for name, model in models.items():
        model.fit(X, y)
        # Save the model
        joblib.dump(model, f'models/{split_name}/{name.replace(" ", "_")}_{data_type}.joblib')

    return models


def perform_cv(model, X, y):
    return cross_val_score(model, X, y, cv=5, scoring='accuracy')


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
    plt.savefig(f'results/confusion_matrices_{split_name.replace("-", "_")}.png')
    plt.close(fig)


def run_experiment(X, y, test_size_1, test_size_2, log_features, split_name):
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size_1, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size_2, stratify=y_temp,
                                                    random_state=42)

    # Preprocess data
    X_train_normal, X_train_log, X_train_mixed = preprocess_data(X_train, log_features)
    X_test_normal, X_test_log, X_test_mixed = preprocess_data(X_test, log_features)

    # Scale data
    scaler = StandardScaler()
    X_train_normal_scaled = scaler.fit_transform(X_train_normal)
    X_test_normal_scaled = scaler.transform(X_test_normal)

    scaler_log = StandardScaler()
    X_train_log_scaled = scaler_log.fit_transform(X_train_log)
    X_test_log_scaled = scaler_log.transform(X_test_log)

    scaler_mixed = StandardScaler()
    X_train_mixed_scaled = scaler_mixed.fit_transform(X_train_mixed)
    X_test_mixed_scaled = scaler_mixed.transform(X_test_mixed)

    # Train models
    models_normal = train_models(X_train_normal_scaled, y_train, 'Normal', split_name)
    models_log = train_models(X_train_log_scaled, y_train, 'Log', split_name)
    models_mixed = train_models(X_train_mixed_scaled, y_train, 'Mixed', split_name)

    # Perform cross-validation
    cv_results = []
    for data_type, X_train_scaled in [('Normal', X_train_normal_scaled), ('Log', X_train_log_scaled),
                                      ('Mixed', X_train_mixed_scaled)]:
        models = models_normal if data_type == 'Normal' else models_log if data_type == 'Log' else models_mixed
        for model_name, model in models.items():
            scores = perform_cv(model, X_train_scaled, y_train)
            cv_results.append({
                'Model': model_name,
                'Data Type': data_type,
                'Mean CV Accuracy': scores.mean(),
                'Std CV Accuracy': scores.std()
            })

    cv_df = pd.DataFrame(cv_results)
    cv_df = cv_df.sort_values('Mean CV Accuracy', ascending=False).reset_index(drop=True)

    # Evaluate all models on test data
    test_results = []
    confusion_matrices = {model_name: {} for model_name in models_normal.keys()}

    for data_type, X_test_scaled in [('Normal', X_test_normal_scaled), ('Log', X_test_log_scaled),
                                     ('Mixed', X_test_mixed_scaled)]:
        models = models_normal if data_type == 'Normal' else models_log if data_type == 'Log' else models_mixed
        for model_name, model in models.items():
            y_pred = model.predict(X_test_scaled)
            test_accuracy = (y_pred == y_test).mean()
            test_results.append({
                'Model': model_name,
                'Data Type': data_type,
                'Test Accuracy': test_accuracy
            })
            confusion_matrices[model_name][data_type] = confusion_matrix(y_test, y_pred)

    test_df = pd.DataFrame(test_results)
    test_df = test_df.sort_values('Test Accuracy', ascending=False).reset_index(drop=True)

    # Plot confusion matrices
    plot_confusion_matrices(confusion_matrices, split_name)

    return cv_df, test_df


# Create necessary directories
create_directories()

# Load your data
df = pd.read_csv('training_data.csv')
X = df.drop('Class', axis=1)
y = df['Class']

# Define log features
log_features = ['Blue', 'Green', 'NIR', 'Red']

# Run experiments
cv_results_70_15_15, test_results_70_15_15 = run_experiment(X, y, 0.3, 0.5, log_features, '70-15-15_Split')
cv_results_60_20_20, test_results_60_20_20 = run_experiment(X, y, 0.4, 0.5, log_features, '60-20-20_Split')

# Save results to CSV
cv_results_70_15_15.to_csv('results/cv_results_70_15_15.csv', index=False)
test_results_70_15_15.to_csv('results/test_results_70_15_15.csv', index=False)
cv_results_60_20_20.to_csv('results/cv_results_60_20_20.csv', index=False)
test_results_60_20_20.to_csv('results/test_results_60_20_20.csv', index=False)
