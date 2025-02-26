import os 
import pandas as pd
import numpy as np
import logging
import time
import warnings
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Ignore warnings
warnings.filterwarnings('ignore')

# Function to load data from file
def load_data(file_path):
    try:
        df = pd.read_csv(file_path, compression='gzip')
        return df
    except Exception as e:
        logging.error(f"Error loading data from file {file_path}: {e}")
        return None

# Function to load the list of selected features
def load_selected_features(selected_features_file):
    try:
        selected_features = pd.read_csv(selected_features_file)
        return selected_features.iloc[:, 0].tolist()
    except Exception as e:
        logging.error(f"Error loading feature list from file {selected_features_file}: {e}")
        return None

# Function to normalize data
def scale_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# Function to save results to CSV file
def save_results_to_csv(results_df, output_directory, filename):
    file_path = os.path.join(output_directory, filename)
    try:
        results_df.to_csv(file_path, index=False)
        logging.info(f"Results saved to file {file_path}")
    except Exception as e:
        logging.error(f"Error saving results to file {file_path}: {e}")

# Function to train and evaluate models using cross-validation, run 10 times
def train_and_evaluate_models_cv(X, y, n_runs=10):
    models = {
        'SVM': SVC(C=100000, kernel='rbf', gamma='scale', class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'XGBoost (GPU)': XGBClassifier(
            n_estimators=200, learning_rate=0.01, max_depth=3, min_child_weight=1, gamma=1,
            subsample=0.8, colsample_bytree=0.8, objective='multi:softmax', use_label_encoder=False,
            eval_metric='mlogloss', random_state=42, num_class=len(np.unique(y)),
            tree_method='gpu_hist', gpu_id=0  # Use GPU if available
        ),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    # Select cross-validation method based on sample size
    if len(y) >= 300:
        logging.info("Using 10-fold Stratified cross-validation as the sample size >= 300.")
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    else:
        logging.info("Using leave-one-out cross-validation as the sample size < 300.")
        cv = LeaveOneOut()

    # Initialize aggregated results for 10 runs
    aggregated_results = {
        'model': [], 'accuracy': [], 'f1_score': [], 'recall': [], 'precision': [], 'training_time': []
    }

    # Perform 10 runs of cross-validation
    for run in range(n_runs):
        logging.info(f"Run {run+1}")
        run_results = {
            'model': [], 'accuracy': [], 'f1_score': [], 'recall': [], 'precision': [], 'training_time': []
        }

        for model_name, model in models.items():
            fold_num = 1
            accuracy_scores = []
            precision_scores = []
            recall_scores = []
            f1_scores = []

            start_train_time = time.time()

            for train_index, test_index in cv.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

                accuracy_scores.append(accuracy)
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)

                fold_num += 1

            end_train_time = time.time()
            training_time = end_train_time - start_train_time

            run_results['model'].append(model_name)
            run_results['accuracy'].append(np.mean(accuracy_scores))
            run_results['f1_score'].append(np.mean(f1_scores))
            run_results['recall'].append(np.mean(recall_scores))
            run_results['precision'].append(np.mean(precision_scores))
            run_results['training_time'].append(training_time)

        for key in aggregated_results:
            if key != 'model':
                aggregated_results[key].extend(run_results[key])
            else:
                aggregated_results[key] = run_results[key]

    for key in aggregated_results:
        if key != 'model':
            aggregated_results[key] = np.mean(aggregated_results[key])

    for i, model_name in enumerate(aggregated_results['model']):
        summary_log_message = (f"{model_name} - Average Accuracy: {aggregated_results['accuracy'][i]*100:.2f}%, "
                               f"Average F1 Score: {aggregated_results['f1_score'][i]*100:.2f}%, "
                               f"Average Recall: {aggregated_results['recall'][i]*100:.2f}%, "
                               f"Average Precision: {aggregated_results['precision'][i]*100:.2f}%, "
                               f"Average Training Time: {aggregated_results['training_time'][i]:.2f} seconds")
        print(summary_log_message)
        logging.info(summary_log_message)

    return pd.DataFrame(aggregated_results)

# List of datasets and corresponding selected feature counts
datasets_and_n_SF = [(36895, 39)]

for dataset, n_SF in datasets_and_n_SF:
    logging.info(f"Processing dataset: {dataset} with {n_SF} selected features.")

    file_path = f'Gene Data/{dataset}/data.trn.gz'
    selected_features_file = os.path.join('1.Experiments', '1.Selected Features', 'Results',  str(dataset), f'{n_SF}_dt_selected_feature_names.csv')
    output_directory = os.path.join('1.Experiments', '1.Selected Features', 'Results',str(dataset))
    log_file = os.path.join(output_directory, 'experiment_log_with_selected_features.log')

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Starting training and evaluation with selected features for dataset {dataset}.")

    df = load_data(file_path)
    if df is not None:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        selected_features = load_selected_features(selected_features_file)
        X_selected = X[selected_features]
        X_scaled = scale_data(X_selected)
        results_df = train_and_evaluate_models_cv(X_scaled, y, n_runs=10)
        save_results_to_csv(results_df, output_directory, f'results_{dataset}_nSF_{n_SF}.csv')
    else:
        logging.error(f"Unable to load data for dataset {dataset}.")
